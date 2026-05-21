"""建 Qdrant 集合。

BGE-M3 主力 → 命名双向量：
    vectors_config = {"dense": VectorParams(1024, COSINE)}
    sparse_vectors_config = {"sparse": SparseVectorParams()}

doubao/voyage/openai → 只建 dense 命名向量（保持 schema 兼容）。

集合名来自 config.QDRANT_COLLECTION（建议每个 provider 配独立集合，避免维度冲突）。
"""
from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

import config


def setup(recreate: bool = False) -> None:
    client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
    name = config.QDRANT_COLLECTION

    if client.collection_exists(name):
        if not recreate:
            print(f"[setup] collection '{name}' exists, skip (use recreate=True to drop+rebuild)")
            return
        print(f"[setup] dropping existing '{name}'")
        client.delete_collection(name)

    vectors_config = {
        "dense": qm.VectorParams(size=config.DENSE_DIM, distance=qm.Distance.COSINE),
    }
    sparse_vectors_config = None
    if config.SUPPORTS_SPARSE:
        sparse_vectors_config = {
            "sparse": qm.SparseVectorParams(
                index=qm.SparseIndexParams(on_disk=False),
            ),
        }

    client.create_collection(
        collection_name=name,
        vectors_config=vectors_config,
        sparse_vectors_config=sparse_vectors_config,
    )
    print(f"[setup] created '{name}': dense_dim={config.DENSE_DIM}, "
          f"sparse={config.SUPPORTS_SPARSE}")

    # payload 索引（按需筛选/过滤）
    for field, t in [
        ("source", qm.PayloadSchemaType.KEYWORD),
        ("post_id", qm.PayloadSchemaType.KEYWORD),
        ("comment_id", qm.PayloadSchemaType.KEYWORD),
        ("tutorial_id", qm.PayloadSchemaType.KEYWORD),
        ("author_id", qm.PayloadSchemaType.KEYWORD),
        ("chunk_type", qm.PayloadSchemaType.KEYWORD),
        ("sa_type", qm.PayloadSchemaType.KEYWORD),
        ("category", qm.PayloadSchemaType.KEYWORD),
        ("operators_mentioned", qm.PayloadSchemaType.KEYWORD),
        ("regions", qm.PayloadSchemaType.KEYWORD),
        ("tags", qm.PayloadSchemaType.KEYWORD),
        ("recency_tag", qm.PayloadSchemaType.KEYWORD),
        ("year", qm.PayloadSchemaType.KEYWORD),
        ("vote_count", qm.PayloadSchemaType.INTEGER),
        ("total_comments", qm.PayloadSchemaType.INTEGER),
        ("created_at_ts", qm.PayloadSchemaType.INTEGER),
        ("age_days", qm.PayloadSchemaType.INTEGER),
        ("reported_sharpe", qm.PayloadSchemaType.FLOAT),
        ("has_code", qm.PayloadSchemaType.BOOL),
        ("source_hash", qm.PayloadSchemaType.KEYWORD),
    ]:
        client.create_payload_index(name, field, t)
    print(f"[setup] payload indices created")


if __name__ == "__main__":
    import sys
    setup(recreate="--recreate" in sys.argv)
