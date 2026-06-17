#!/usr/bin/env python3
"""Deduplicate a JSONL dataset by ``uid`` (keeps the last row per uid)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def dedupe_jsonl(path: Path, in_place: bool = True) -> int:
    rows = {}
    order = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            uid = rec["uid"]
            if uid not in rows:
                order.append(uid)
            rows[uid] = rec

    out_path = path if in_place else path.with_suffix(".dedup.jsonl")
    with out_path.open("w", encoding="utf-8") as f:
        for uid in order:
            f.write(json.dumps(rows[uid], ensure_ascii=False) + "\n")
    return len(order)


def main() -> None:
    ap = argparse.ArgumentParser(description="Deduplicate JSONL examples by uid")
    ap.add_argument("jsonl", type=Path, help="Path to *.jsonl file")
    ap.add_argument("--no-in-place", action="store_true")
    args = ap.parse_args()
    n = dedupe_jsonl(args.jsonl, in_place=not args.no_in_place)
    print(f"Wrote {n} unique rows to {args.jsonl if not args.no_in_place else args.jsonl}")


if __name__ == "__main__":
    main()
