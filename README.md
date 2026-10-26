# CurbPark-RL

靠边停车强化学习的复现与鲁棒性评测（个人学习项目）。

问题思路参考一篇 Robotaxi 强化学习方向的公开简历（哈啰出行靠边进站 PPO 项目 + 无人机鲁棒 RL 项目），
环境、代码与全部实验数据均在本仓库基于开源仿真独立完成，不使用任何公司代码或真实车辆数据。

- 环境：highway-env `parking-v0`（目标条件连续控制停车）
- 算法：PPO（Stable-Baselines3，MultiInputPolicy）
- 主线三步：①默认奖励基线 + 确定性评测 → ②区域停车（Goal Region）奖励改造消融 → ③观测噪声/扰动鲁棒性评测与 Failure Case 归因

## 快速开始

```bash
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt

.venv/bin/python src/probe_env.py                                   # 先看清环境
.venv/bin/python src/train_ppo.py --seed 0 --timesteps 150000 --out models/ppo_seed0
.venv/bin/python src/evaluate.py --model models/ppo_seed0 --episodes 50 --out results/baseline/eval_seed0
.venv/bin/python src/evaluate.py --random --episodes 50 --out results/baseline/eval_random
```

## 评测口径（全项目统一，见 src/common.py）

- 成功：终态位置误差 ≤ 1.5m 且航向误差 ≤ 15°
- 失败分型：collision 碰撞 / angled_parking 斜停 / lateral_offset 横向偏移 /
  overshoot_or_short 越位或未达 / timeout_wander 超时打转
- 评测集：固定种子 1000–1049 的 50 个场景，确定性策略

## 目录

- `src/common.py` 环境工厂、误差计算、成功判定、失败分型
- `src/train_ppo.py` PPO 训练
- `src/evaluate.py` 确定性评测（输出逐回合 CSV + 汇总 JSON）
- `src/probe_env.py` 环境探针
- `results/baseline/` 第 1 周基线结果
- `notes/` 学习笔记（GAE/PPO 推导等）
