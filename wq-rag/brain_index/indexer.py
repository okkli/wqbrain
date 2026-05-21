"""切分 → 嵌入 → 索引到 Qdrant，支持增量。

增量逻辑:
- 每个 chunk 在 payload 里存 source_hash = sha1(source_path + mtime + size)
- 重跑时先 scroll 出现有 (point_id, source_hash) 映射
- 当前 chunk 的 source_hash 与库内一致 → 跳过
- point_id 由 uuid5(chunk_type, parent_id, chunk_index) 生成，重跑覆盖而非重复
"""
from __future__ import annotations
import hashlib
import os
import time
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

import config
import parser as p_mod
import chunker
from embedder import get_embedder, recommended_batch_size
from sources import load_forum, load_tutorials


def _source_hash(path: str | None) -> str:
    """基于路径 + mtime + size + INDEX_VERSION 的轻量增量 key。

    混入 config.INDEX_VERSION 是为了：改了 TAG_RULES / 切分阈值后，bump 版本号即可
    强制重嵌全库（未改源文件的 chunk 也会被识别为"已过期"重新走嵌入）。
    """
    if not path:
        return ""
    try:
        st = os.stat(path)
        s = f"{path}|{int(st.st_mtime)}|{st.st_size}|{config.INDEX_VERSION}"
    except FileNotFoundError:
        s = f"{path}|{config.INDEX_VERSION}"
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def _point_id(c: chunker.Chunk) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{c.chunk_type}|{c.parent_id}|{c.chunk_index}"))


def _build_embedding_text(c: chunker.Chunk) -> str:
    """嵌入前加上下文头，提升召回（BGE-M3 受益较小，但便于跨 provider 一致）。"""
    m = c.metadata
    head = []
    if m.get("title"):
        head.append(f"[主题: {m['title']}]")
    if m.get("author_id"):
        a = m["author_id"]
        if m.get("vote_count"):
            a += f", 票数: {m['vote_count']}"
        head.append(f"[作者: {a}]")
    if m.get("sa_type"):
        head.append(f"[类型: {m['sa_type']}]")
    if m.get("source") == "tutorial" and m.get("section_heading"):
        head.append(f"[小节: {m['section_heading']}]")
    return ("\n".join(head) + "\n\n" + c.text) if head else c.text


def collect_chunks(verbose: bool = True) -> list[chunker.Chunk]:
    posts = []
    for p in load_forum():
        p_mod.enrich_post(p)
        for c in p.comments:
            p_mod.enrich_comment(c)
        posts.append(p)
    tutorials = list(load_tutorials())
    for s in tutorials:
        p_mod.enrich_tutorial(s)
    chunks = chunker.collect_all(posts, tutorials)
    if verbose:
        print(f"[collect] {len(chunks)} chunks from {len(posts)} posts + {len(tutorials)} tutorial sections")
    return chunks


def _existing_hashes(client: QdrantClient) -> dict[str, str]:
    """scroll 所有现有 points 的 (point_id → source_hash)，用于增量跳过。"""
    out: dict[str, str] = {}
    offset = None
    while True:
        pts, offset = client.scroll(
            collection_name=config.QDRANT_COLLECTION,
            limit=512,
            with_payload=["source_hash"],
            with_vectors=False,
            offset=offset,
        )
        for p in pts:
            sh = (p.payload or {}).get("source_hash")
            if sh:
                out[p.id] = sh
        if offset is None:
            break
    return out


def index_all(*, incremental: bool = True, batch_size: int | None = None) -> None:
    client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
    if not client.collection_exists(config.QDRANT_COLLECTION):
        raise RuntimeError(
            f"collection '{config.QDRANT_COLLECTION}' not found — run setup_qdrant.py first"
        )

    chunks = collect_chunks()
    existing = _existing_hashes(client) if incremental else {}
    print(f"[index] existing points: {len(existing)} (incremental={incremental})")

    # 增量过滤
    to_index: list[tuple[str, str, chunker.Chunk]] = []   # (pid, sh, chunk)
    skipped = 0
    for c in chunks:
        sh = _source_hash(c.metadata.get("source_path"))
        pid = _point_id(c)
        if incremental and existing.get(pid) == sh and sh:
            skipped += 1
            continue
        to_index.append((pid, sh, c))
    print(f"[index] to_index={len(to_index)} skipped={skipped}")

    if not to_index:
        print("[index] nothing to do")
        return

    embedder = get_embedder()
    bs = batch_size or recommended_batch_size()
    print(f"[index] provider={embedder.PROVIDER} batch_size={bs}")

    t0 = time.time()
    upserted = 0
    for i in range(0, len(to_index), bs):
        batch = to_index[i:i + bs]
        texts = [_build_embedding_text(b[2]) for b in batch]
        res = embedder.embed(texts, input_type="document")

        points = []
        for (pid, sh, c), dense_vec, sparse_vec in zip(batch, res.dense, res.sparse):
            payload = {**c.metadata,
                       "chunk_type": c.chunk_type,
                       "chunk_index": c.chunk_index,
                       "body_md": c.text,
                       "source_hash": sh}
            vector = {"dense": dense_vec}
            if config.SUPPORTS_SPARSE and sparse_vec:
                vector["sparse"] = qm.SparseVector(
                    indices=sparse_vec["indices"],
                    values=sparse_vec["values"],
                )
            points.append(qm.PointStruct(id=pid, vector=vector, payload=payload))

        client.upsert(collection_name=config.QDRANT_COLLECTION, points=points, wait=False)
        upserted += len(points)
        elapsed = time.time() - t0
        rate = upserted / max(elapsed, 0.001)
        print(f"  [{upserted}/{len(to_index)}] rate={rate:.1f}/s elapsed={elapsed:.1f}s")

    print(f"[index] ✅ done. upserted={upserted} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    import sys
    incremental = "--full" not in sys.argv
    index_all(incremental=incremental)
