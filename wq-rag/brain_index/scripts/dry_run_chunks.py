"""离线切分验收。

不发请求到 BGE-M3 / Qdrant，仅做 load → enrich → chunk → 统计 + 断言。

输出:
- 全量统计：chunks 总数、按 type 分布、长度分位
- 质量报告：碎片(<MIN_FRAGMENT_CHARS) 数量与样本
- 代码块断言：所有 code_block chunk 的 ``` 配对完整
- 元数据健康：有多少 chunk 命中 SA 算子 / 有 sa_type / 报了 Sharpe
- 样本打印：每类 chunk 各 1 条
"""
from __future__ import annotations
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

# 让脚本在 brain_index/ 下任何位置都能跑
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
import parser as p_mod
import chunker
from sources import load_forum, load_tutorials


def main() -> int:
    print(f"[config] {config.summary()}")
    t0 = time.time()

    posts = []
    n_comments = 0
    for p in load_forum():
        p_mod.enrich_post(p)
        for c in p.comments:
            p_mod.enrich_comment(c)
        posts.append(p)
        n_comments += len(p.comments)
    print(f"[load] forum: {len(posts)} posts, {n_comments} comments  "
          f"({time.time()-t0:.1f}s)")

    t1 = time.time()
    tutorials = list(load_tutorials())
    for s in tutorials:
        p_mod.enrich_tutorial(s)
    print(f"[load] tutorials: {len(tutorials)} sections  "
          f"({time.time()-t1:.1f}s)")

    t2 = time.time()
    chunks = chunker.collect_all(posts, tutorials)
    print(f"[chunk] total {len(chunks)} chunks  ({time.time()-t2:.1f}s)")

    by_type = Counter(c.chunk_type for c in chunks)
    by_source = Counter(c.metadata.get("source") for c in chunks)
    lens = [len(c.text) for c in chunks]
    lens_sorted = sorted(lens)
    q = statistics.quantiles(lens, n=20) if len(lens) >= 20 else lens_sorted

    print()
    print("=== Distribution ===")
    print(f"by chunk_type : {dict(by_type)}")
    print(f"by source     : {dict(by_source)}")
    print(f"length  min={min(lens)} mean={int(sum(lens)/len(lens))}"
          f" median={lens_sorted[len(lens)//2]} p95={int(q[-1])} max={max(lens)}")

    # 碎片检查
    short = [c for c in chunks
             if len(c.text) < config.MIN_FRAGMENT_CHARS
             and c.chunk_type not in ("code_block",)]
    print()
    print(f"=== Low-quality fragments (<{config.MIN_FRAGMENT_CHARS} chars, non-code) ===")
    print(f"count: {len(short)} ({100*len(short)/len(chunks):.1f}% of total)")
    for c in short[:5]:
        print(f"  {c.chunk_type} {c.parent_id} ({len(c.text)}c): {c.text!r}")

    # 代码块完整性断言
    code_chunks = [c for c in chunks if c.chunk_type == "code_block"]
    bad = [c for c in code_chunks if c.text.count("```") < 2]
    print()
    print("=== Code block integrity ===")
    print(f"code_block chunks: {len(code_chunks)} (with SA operators ≥2)")
    print(f"BROKEN (``` 不成对): {len(bad)}")
    for c in bad[:3]:
        print(f"  parent={c.parent_id} text head: {c.text[:200]!r}")
    assert not bad, "代码块完整性断言失败"

    # 元数据健康度
    n_with_op = sum(1 for c in chunks if c.metadata.get("operators_mentioned"))
    n_with_sa = sum(1 for c in chunks if c.metadata.get("sa_type"))
    n_with_sharpe = sum(1 for c in chunks if c.metadata.get("reported_sharpe") is not None)
    n_with_code = sum(1 for c in chunks if c.metadata.get("has_code"))
    print()
    print("=== Metadata coverage ===")
    print(f"chunks w/ operators_mentioned : {n_with_op} ({100*n_with_op/len(chunks):.1f}%)")
    print(f"chunks w/ sa_type             : {n_with_sa} ({100*n_with_sa/len(chunks):.1f}%)")
    print(f"chunks w/ reported_sharpe     : {n_with_sharpe} ({100*n_with_sharpe/len(chunks):.1f}%)")
    print(f"chunks w/ has_code            : {n_with_code} ({100*n_with_code/len(chunks):.1f}%)")

    # 时效性分布
    recency = Counter(c.metadata.get("recency_tag") for c in chunks)
    year_d = Counter(c.metadata.get("year") for c in chunks if c.metadata.get("year"))
    print()
    print("=== Recency ===")
    for k in ("fresh", "recent", "older", "archived", None):
        v = recency.get(k, 0)
        print(f"  {str(k):10s} : {v} ({100*v/len(chunks):.1f}%)")
    print(f"  by year: {dict(sorted(year_d.items()))}")

    # 内容标签分布
    tag_counts = Counter()
    for c in chunks:
        for t in (c.metadata.get("tags") or []):
            tag_counts[t] += 1
    n_tagged = sum(1 for c in chunks if c.metadata.get("tags"))
    print()
    print("=== Content tags ===")
    print(f"chunks w/ ≥1 tag : {n_tagged} ({100*n_tagged/len(chunks):.1f}%)")
    for tag, n in tag_counts.most_common():
        print(f"  {tag:24s} {n}")

    # 抽样输出
    print()
    print("=== Sample per type ===")
    seen = set()
    for c in chunks:
        if c.chunk_type in seen:
            continue
        seen.add(c.chunk_type)
        head = c.text[:300].replace("\n", " / ")
        print(f"-- {c.chunk_type} | parent={c.parent_id} | len={len(c.text)} --")
        print(f"   {head}")
        if len(seen) >= 6:
            break

    # 嵌入 token 估算（粗：1 字符 ≈ 0.4 token）
    total_chars = sum(lens)
    est_tokens = int(total_chars * 0.4)
    print()
    print("=== Embedding workload estimate ===")
    print(f"total chars   : {total_chars:,}")
    print(f"est tokens    : {est_tokens:,}  (假设 1 char ≈ 0.4 token，混合中英)")
    print(f"BGE-M3 估算   : ~{len(chunks)/50/60:.1f} min @ 50 req/s (batch=64)")
    print(f"Doubao 估算   : ~{len(chunks)/20/60:.1f} min @ 20 req/s (batch=10)")

    print()
    print(f"✅ dry_run done in {time.time()-t0:.1f}s — all assertions passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
