#!/usr/bin/env python3
"""Wait for the results of a tagged simulation batch, then print a short table.

Meant for a background shell, so an agent spends one turn on a batch instead of
polling get_simulation every 40 seconds (most of those polls come back RUNNING).
It reads the server's results log GET /results/<tag>.jsonl.

    python3 wqwait.py <tag> <N> [--sim S1 S2 ...] [--since K] [--max SECONDS]

N        result lines to wait for (one per item of the batch).
--sim    count only lines of these simulation ids (and items resent from them);
         without it, every line after the first K (--since) counts.
--max    give up after this many seconds (default 3000); exit code 2 then.
Server:  $WQMCP_RESULTS_URL, default http://127.0.0.1:8761.

A line counts when it is a result: it has an alpha id, or a final status
(ERROR / FAIL / CANCELLED ...). RESENT / RETRYING lines and rows without an id
or a status are not results and are skipped. Standard library only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

NOT_RESULTS = {"RESENT", "RETRYING", "RUNNING", "UNKNOWN", "QUEUED", ""}


def fetch(base: str, tag: str) -> list:
    with urllib.request.urlopen(f"{base.rstrip('/')}/results/{tag}.jsonl", timeout=30) as resp:
        text = resp.read().decode("utf-8")
    rows = []
    for line in text.splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def is_result(row: dict) -> bool:
    return bool(row.get("id")) or str(row.get("status") or "").upper() not in NOT_RESULTS


def pick(rows: list, sims: set, since: int) -> list:
    if sims:     # a simulation id, the queue id it waited under, or a batch it was resent from
        sims = set(sims)
        while True:      # follow resends of resends (a resent batch split again)
            more = {r.get("simulation_id") for r in rows if r.get("resent_from") in sims} - sims - {None}
            if not more:
                break
            sims |= more
        return [r for r in rows if is_result(r) and (r.get("simulation_id") in sims or r.get("queue_id") in sims
                                                     or r.get("resent_from") in sims)]
    return [r for r in rows[since:] if is_result(r)]


def brief(rows: list) -> str:
    cols = ("index", "label", "id", "sharpe", "fitness", "turnover", "margin_bps", "sub_sharpe", "y2_sharpe",
            "fails", "status")
    table = [[str(r.get("label") or "") if c == "label" else
              ",".join(map(str, r[c])) if isinstance(r.get(c), list) else
              ("" if c == "status" and r.get(c) == "COMPLETE" else str(r.get(c) if r.get(c) is not None else ""))
              for c in cols] for r in sorted(rows, key=lambda r: (str(r.get("simulation_id")), r.get("index", 0)))]
    keep = [i for i in range(len(cols)) if any(line[i] for line in table)]
    return "\n".join(["\t".join(cols[i] for i in keep)] + ["\t".join(line[i] for i in keep) for line in table])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("tag")
    ap.add_argument("n", type=int)
    ap.add_argument("--sim", nargs="*", default=[])
    ap.add_argument("--since", type=int, default=0)
    ap.add_argument("--max", type=float, default=3000.0)
    ap.add_argument("--interval", type=float, default=20.0)
    args = ap.parse_args()
    base = os.environ.get("WQMCP_RESULTS_URL", "http://127.0.0.1:8761")
    deadline = time.time() + args.max
    got: list = []
    while True:
        try:
            got = pick(fetch(base, args.tag), set(args.sim), args.since)
        except (urllib.error.URLError, OSError) as e:      # the server restarting: keep waiting
            print(f"# {e}", file=sys.stderr)
        if len(got) >= args.n or time.time() >= deadline:
            break
        time.sleep(args.interval)
    print(brief(got))
    if len(got) < args.n:
        print(f"# timed out: {len(got)} of {args.n} results", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
