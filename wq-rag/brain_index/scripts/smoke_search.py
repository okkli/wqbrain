"""冒烟检索。验证索引可用 + 混合检索质量。

BGE-M3 主力时走 Qdrant Fusion (RRF) 原生混合检索：dense + sparse 各自召回 top_k*2
再用 RRF 融合 top_k。

doubao/voyage/openai 只跑 dense。
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

import config
from embedder import get_embedder


QUERIES = [
    "如何选取低相关的因子做SuperAlpha组合",
    "turnover 怎么分段降低换手率",
    "ts_corr 算子的常见用法",
    "Selection 表达式 trade_when 用法",
    "alpha submission 的流程是什么",
]


def search(client: QdrantClient, query: str, top_k: int = 5):
    res = get_embedder().embed([query], input_type="query")
    dense_q = res.dense[0]
    sparse_q = res.sparse[0] if config.SUPPORTS_SPARSE else None

    prefetch = [qm.Prefetch(query=dense_q, using="dense", limit=top_k * 2)]
    if sparse_q:
        prefetch.append(qm.Prefetch(
            query=qm.SparseVector(indices=sparse_q["indices"], values=sparse_q["values"]),
            using="sparse",
            limit=top_k * 2,
        ))

    r = client.query_points(
        collection_name=config.QDRANT_COLLECTION,
        prefetch=prefetch,
        query=qm.FusionQuery(fusion=qm.Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )
    return r.points


def main():
    client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
    if not client.collection_exists(config.QDRANT_COLLECTION):
        print(f"collection '{config.QDRANT_COLLECTION}' not found — run setup + indexer first")
        return 1

    for q in QUERIES:
        print(f"\n=== {q} ===")
        for p in search(client, q):
            md = p.payload or {}
            head = (md.get("body_md") or "")[:80].replace("\n", " ")
            tags = ",".join((md.get("tags") or [])[:3])
            print(
                f"  {p.score:.4f}  src={md.get('source'):<14} "
                f"recency={md.get('recency_tag') or '-':<8} "
                f"year={md.get('year') or '-':<5} "
                f"tags=[{tags}]  "
                f"by {md.get('author_id') or md.get('tutorial_id')}"
            )
            print(f"           ↳ {head}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
