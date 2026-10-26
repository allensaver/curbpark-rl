"""确定性评测：固定场景种子集，只认 common.py 的统一口径。

用法:
  .venv/bin/python src/evaluate.py --model models/ppo_seed0 --episodes 50 --out results/baseline/eval_seed0
  .venv/bin/python src/evaluate.py --random --episodes 50 --out results/baseline/eval_random
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import classify_failure, goal_errors, is_success, make_env

MAX_STEPS = 100
EVAL_SEED_BASE = 1000  # 固定评测集：种子 1000..1000+N-1，与训练种子错开


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="SB3 模型路径（不含 .zip）")
    ap.add_argument("--random", action="store_true", help="评测随机策略")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--out", required=True, help="输出前缀（生成 .csv 与 _summary.json）")
    args = ap.parse_args()

    model = None
    if args.model:
        from stable_baselines3 import PPO
        model = PPO.load(args.model)

    env = make_env()
    rows = [run_episode(env, model, EVAL_SEED_BASE + i) for i in range(args.episodes)]
    env.close()

    n = len(rows)
    succ = [r for r in rows if r["success"]]
    summary = {
        "episodes": n,
        "success_rate": round(len(succ) / n, 4),
        "heading_ok_rate": round(sum(r["head_err_deg"] <= 15.0 for r in rows) / n, 4),
        "collision_rate": round(sum(r["crashed"] for r in rows) / n, 4),
        "mean_pos_err_m": round(float(np.mean([r["pos_err_m"] for r in rows])), 3),
        "mean_head_err_deg": round(float(np.mean([r["head_err_deg"] for r in rows])), 3),
        "mean_steps": round(float(np.mean([r["steps"] for r in rows])), 2),
        "mean_return": round(float(np.mean([r["return"] for r in rows])), 3),
        "failure_breakdown": dict(Counter(r["failure_type"] for r in rows)),
        "policy": "random" if args.random else args.model,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(args.out + "_summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def run_episode(env, model, seed: int) -> dict:
    obs, info = env.reset(seed=seed)
    steps, total_reward, crashed = 0, 0.0, False
    done = False
    while not done:
        if model is None:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1
        if info.get("crashed"):
            crashed = True
        done = terminated or truncated
    pos_err, head_err, lon, lat = goal_errors(obs)
    success = is_success(pos_err, head_err)
    return {
        "success": success,
        "failure_type": "success" if success else classify_failure(
            pos_err, head_err, lon, lat, crashed, steps, MAX_STEPS),
        "pos_err_m": round(pos_err, 3),
        "head_err_deg": round(head_err, 2),
        "longitudinal_err_m": round(lon, 3),
        "lateral_err_m": round(lat, 3),
        "crashed": crashed,
        "steps": steps,
        "return": round(total_reward, 3),
    }


if __name__ == "__main__":
    main()
