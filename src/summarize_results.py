"""把 results/ 下各 *_summary.json 汇成一张 Markdown 对比表。

用法: .venv/bin/python src/summarize_results.py results/baseline
"""
from __future__ import annotations

import glob
import json
import os
import sys


def main() -> None:
    folder = sys.argv[1] if len(sys.argv) > 1 else "results/baseline"
    files = sorted(glob.glob(os.path.join(folder, "*_summary.json")))
    if not files:
        print("no summary files found")
        return
    cols = ["policy", "success_rate", "heading_ok_rate", "collision_rate",
            "mean_pos_err_m", "mean_head_err_deg", "mean_steps", "mean_return"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for fp in files:
        with open(fp) as f:
            s = json.load(f)
        name = os.path.basename(fp).replace("_summary.json", "")
        row = [name] + [str(s.get(c, "")) for c in cols[1:]]
        print("| " + " | ".join(row) + " |")
        print(f"  failures[{name}]: {s.get('failure_breakdown')}")


if __name__ == "__main__":
    main()
