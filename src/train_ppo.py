"""PPO 基线训练（Week 1）：默认奖励、固定种子、CPU。

用法:
  .venv/bin/python src/train_ppo.py --seed 0 --timesteps 150000 --out models/ppo_seed0
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timesteps", type=int, default=150_000)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--reward-version", default="v1", choices=["v1", "v2a", "v2b"],
                    help="奖励版本：v1 基线 / v2a 堵撞停套利 / v2b 区域奖励")
    ap.add_argument("--out", required=True, help="模型保存路径（不含 .zip）")
    args = ap.parse_args()

    import torch
    torch.set_num_threads(2)

    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.monitor import Monitor
    from common import make_env

    def _factory():
        return Monitor(make_env(reward_version=args.reward_version))

    vec_env = make_vec_env(_factory, n_envs=args.n_envs, seed=args.seed)
    model = PPO(
        "MultiInputPolicy",
        vec_env,
        seed=args.seed,
        verbose=0,
        n_steps=512,
        batch_size=512,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        learning_rate=3e-4,
        policy_kwargs=dict(net_arch=[256, 256]),
    )
    print(f"training reward_version={args.reward_version} seed={args.seed} steps={args.timesteps}", flush=True)
    model.learn(total_timesteps=args.timesteps)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    model.save(args.out)
    print(f"saved -> {args.out}.zip")


if __name__ == "__main__":
    main()
