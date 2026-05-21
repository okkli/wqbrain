"""统一嵌入封装。BGE-M3 为主力（dense+sparse），doubao/voyage/openai 为 dense-only 备选。

主接口 `embed(texts, input_type)` 返回 EmbedResult，对应 dense + 可选 sparse。
调用方（indexer）不关心 provider 细节。
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Literal

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

import config


SparseVec = dict   # {"indices": [int], "values": [float]}


@dataclass
class EmbedResult:
    dense: list[list[float]]
    sparse: list[SparseVec | None] = field(default_factory=list)  # 与 dense 等长；不支持 sparse 的 provider 全 None
    usage_tokens: int = 0
    provider: str = ""


# ─── BGE-M3 (vLLM pooling endpoint) ────────────────────────────────────────
class _BGEM3:
    """vLLM /pooling endpoint，单次拿 dense + sparse。"""
    PROVIDER = "bgem3"

    def __init__(self):
        self.url = f"{config.BGEM3_BASE_URL}{config.BGEM3_POOLING_PATH}"
        self.client = httpx.Client(timeout=120.0)

    def close(self):
        self.client.close()

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=30))
    def _call(self, payload: dict) -> dict:
        r = self.client.post(self.url, json=payload)
        r.raise_for_status()
        return r.json()

    def embed(self, texts: list[str], input_type: Literal["document", "query"] = "document") -> EmbedResult:
        # BGE-M3 不区分 input_type（对称模型），参数被忽略
        payload = {"task": "plugin", "data": {"input": texts, "return_tokens": False}}
        resp = self._call(payload)
        items = resp["data"]["data"]
        dense: list[list[float]] = []
        sparse: list[SparseVec] = []
        for it in items:
            dense.append(it["dense_embedding"])
            se = it.get("sparse_embedding", [])
            sparse.append({
                "indices": [int(x["token_id"]) for x in se],
                "values": [float(x["weight"]) for x in se],
            })
        return EmbedResult(dense=dense, sparse=sparse, provider=self.PROVIDER)


# ─── Doubao (volcengine /api/coding/v3/embeddings, OpenAI-compat) ──────────
class _Doubao:
    PROVIDER = "doubao"

    def __init__(self):
        if not config.DOUBAO_API_KEY:
            raise RuntimeError("DOUBAO_API_KEY not set")
        self.url = f"{config.DOUBAO_BASE_URL}/embeddings"
        self.client = httpx.Client(timeout=60.0)
        self.headers = {
            "Authorization": f"Bearer {config.DOUBAO_API_KEY}",
            "Content-Type": "application/json",
        }

    def close(self):
        self.client.close()

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=30))
    def _call(self, payload: dict) -> dict:
        r = self.client.post(self.url, headers=self.headers, json=payload)
        r.raise_for_status()
        return r.json()

    def embed(self, texts: list[str], input_type: Literal["document", "query"] = "document") -> EmbedResult:
        # 豆包 max batch = 10
        if len(texts) > 10:
            raise ValueError("doubao batch size > 10")
        payload = {
            "model": config.DOUBAO_MODEL,
            "input": texts,
            "dimensions": config.DOUBAO_DIM,
            "encoding_format": "float",
        }
        resp = self._call(payload)
        dense = [d["embedding"] for d in resp["data"]]
        return EmbedResult(
            dense=dense,
            sparse=[None] * len(dense),
            usage_tokens=int(resp.get("usage", {}).get("total_tokens", 0)),
            provider=self.PROVIDER,
        )


# ─── Voyage ────────────────────────────────────────────────────────────────
class _Voyage:
    PROVIDER = "voyage"

    def __init__(self):
        if not config.VOYAGE_API_KEY:
            raise RuntimeError("VOYAGE_API_KEY not set")
        import voyageai  # noqa: lazy
        self.client = voyageai.Client(api_key=config.VOYAGE_API_KEY)

    def close(self):
        pass

    def embed(self, texts: list[str], input_type: Literal["document", "query"] = "document") -> EmbedResult:
        r = self.client.embed(texts, model=config.VOYAGE_MODEL, input_type=input_type)
        return EmbedResult(
            dense=r.embeddings,
            sparse=[None] * len(r.embeddings),
            usage_tokens=getattr(r, "total_tokens", 0) or 0,
            provider=self.PROVIDER,
        )


# ─── OpenAI ────────────────────────────────────────────────────────────────
class _OpenAI:
    PROVIDER = "openai"

    def __init__(self):
        if not config.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY not set")
        from openai import OpenAI  # noqa: lazy
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)

    def close(self):
        pass

    def embed(self, texts: list[str], input_type: Literal["document", "query"] = "document") -> EmbedResult:
        r = self.client.embeddings.create(model=config.OPENAI_MODEL, input=texts)
        dense = [d.embedding for d in r.data]
        return EmbedResult(
            dense=dense,
            sparse=[None] * len(dense),
            usage_tokens=getattr(r.usage, "total_tokens", 0) or 0,
            provider=self.PROVIDER,
        )


# ─── 工厂 ──────────────────────────────────────────────────────────────────
_REGISTRY = {
    "bgem3": _BGEM3,
    "doubao": _Doubao,
    "voyage": _Voyage,
    "openai": _OpenAI,
}

_instance = None


def get_embedder():
    global _instance
    if _instance is None:
        cls = _REGISTRY.get(config.EMBEDDING_PROVIDER)
        if cls is None:
            raise ValueError(f"unknown provider: {config.EMBEDDING_PROVIDER}")
        _instance = cls()
    return _instance


def recommended_batch_size() -> int:
    """各 provider 实测/官方上限。"""
    return {
        "bgem3": 64,
        "doubao": 10,
        "voyage": 128,
        "openai": 256,
    }.get(config.EMBEDDING_PROVIDER, 32)
