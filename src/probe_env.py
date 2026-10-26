"""环境探针：先看清观测/动作/奖励长什么样，再谈训练。

用法: .venv/bin/python src/probe_env.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import goal_errors, make_env


def main() -> None:
    env = make_env()
    print("observation_space:", env.observation_space)
    print("action_space:", env.action_space)
    obs, info = env.reset(seed=42)
    for k, v in obs.items():
        print(f"obs[{k}] shape={np.asarray(v).shape} value={np.round(v, 3)}")
    print("info at reset:", info)
    print("goal_errors at reset (pos, head_deg, lon, lat):", np.round(goal_errors(obs), 3))

    # 随机策略跑 3 回合，看 info 键与回合长度
    for ep in range(3):
        obs, info = env.reset(seed=100 + ep)
        steps, ret, done = 0, 0.0, False
        last_info = info
        while not done:
            obs, r, terminated, truncated, last_info = env.step(env.action_space.sample())
            steps += 1
            ret += float(r)
            done = terminated or truncated
        pe, he, lon, lat = goal_errors(obs)
        print(f"random ep{ep}: steps={steps} return={ret:.2f} pos_err={pe:.2f} head_err={he:.1f} "
              f"info_keys={sorted(last_info.keys())} crashed={last_info.get('crashed')}")
    env.close()


if __name__ == "__main__":
    main()
