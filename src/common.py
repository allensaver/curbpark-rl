"""CurbPark-RL 公共模块：环境工厂、成功判定、Failure Case 分型。

设计原则（与项目计划一致）：
- 评测先行：成功/失败只由这里的统一口径判定，训练脚本不得另设标准。
- 分型口径对应计划文档第五节：越位/未达、斜停、横向偏移、碰撞、超时打转。
"""
from __future__ import annotations

import math
import gymnasium as gym
import highway_env  # noqa: F401  # 注册 parking-v0
import numpy as np

ENV_ID = "parking-v0"

# 成功口径：位置误差与航向误差同时达标才算停好（数值可在 config 中调整后全项目统一）
SUCCESS_POS_TOL = 1.5   # m，终态位置误差上限
SUCCESS_HEAD_TOL_DEG = 15.0  # deg，终态航向误差上限
GOOD_POS_TOL = 2.5      # m，分型用：位置“基本到位”的上限


def make_env(seed: int | None = None, config: dict | None = None, reward_version: str = "v1"):
    """创建 parking-v0。默认使用连续动作与 Goal 观测（与 SB3 MultiInputPolicy 兼容）。

    reward_version: "v1" 基线（碰撞 -5）/ "v2a" 只改碰撞代价 -50 /
    "v2b" v2a + Goal Region 区域奖励包装器（见 reward_v2.py）/
    "v2c" 平衡调参：碰撞 -30 + 包装器配比上调（进度 0.30、精调 0.06、
    终局 +30，设计见 results/v2c/v2c设计.md）。
    """
    collision_reward = {"v1": -5, "v2a": -50, "v2b": -50, "v2c": -30}[reward_version]
    cfg = {
        "observation": {"type": "KinematicsGoal", "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"], "scales": [100, 100, 5, 5, 1, 1], "normalize": False},
        "action": {"type": "ContinuousAction"},
        "collision_reward": collision_reward,
        "reward_speed_range": [0, 0],
    }
    if config:
        cfg.update(config)
    env = gym.make(ENV_ID, config=cfg)
    # 注意：highway-env 的 duration 单位是秒（time += 1/policy_frequency），
    # 不是步数。用 TimeLimit 统一截断为 100 步，保证评测口径与分型阈值一致。
    env = gym.wrappers.TimeLimit(env, max_episode_steps=100)
    if reward_version == "v2b":
        from reward_v2 import GoalRegionRewardWrapper
        env = GoalRegionRewardWrapper(env)
    elif reward_version == "v2c":
        from reward_v2 import GoalRegionRewardWrapper
        env = GoalRegionRewardWrapper(env, progress_w=0.30, adjust_w=0.06, final_bonus=30.0)
    if seed is not None:
        env.reset(seed=seed)
    return env


def goal_errors(obs: dict) -> tuple[float, float, float, float]:
    """从 Dict 观测计算终态误差。

    返回 (pos_err, head_err_deg, longitudinal_err, lateral_err)
    纵/横向误差在目标航向坐标系下分解：纵向=沿车位朝向，横向=垂直方向。
    """
    ach = np.asarray(obs["achieved_goal"], dtype=float)
    des = np.asarray(obs["desired_goal"], dtype=float)
    # features 顺序与 make_env 中一致: [x, y, vx, vy, cos_h, sin_h]
    # 注意：x/y 在观测里被 scales=100 缩放过（探针实测 reset 时目标点为 -0.14,-0.14，
    # 对应实际 -14m,-14m），计算米制误差前必须先还原。
    POS_SCALE = 100.0
    dx, dy = (ach[0] - des[0]) * POS_SCALE, (ach[1] - des[1]) * POS_SCALE
    pos_err = float(math.hypot(dx, dy))
    ach_head = math.atan2(ach[5], ach[4])
    des_head = math.atan2(des[5], des[4])
    head_err = float(abs((ach_head - des_head + math.pi) % (2 * math.pi) - math.pi) * 180.0 / math.pi)
    # 分解到目标坐标系
    c, s = math.cos(des_head), math.sin(des_head)
    longitudinal = float(dx * c + dy * s)
    lateral = float(-dx * s + dy * c)
    return pos_err, head_err, longitudinal, lateral


def is_success(pos_err: float, head_err_deg: float) -> bool:
    return pos_err <= SUCCESS_POS_TOL and head_err_deg <= SUCCESS_HEAD_TOL_DEG


def classify_failure(pos_err: float, head_err_deg: float, longitudinal: float,
                     lateral: float, crashed: bool, steps: int, max_steps: int) -> str:
    """对未成功回合分型。分类顺序即优先级，写入报告时保持一致。"""
    if crashed:
        return "collision"          # 碰撞/贴障
    if steps >= max_steps and pos_err > GOOD_POS_TOL:
        return "timeout_wander"     # 超时/打转（没到位就耗尽步数）
    if pos_err <= GOOD_POS_TOL and head_err_deg > SUCCESS_HEAD_TOL_DEG:
        return "angled_parking"     # 斜停（位置到位、航向不达标）
    if abs(lateral) > abs(longitudinal):
        return "lateral_offset"     # 横向偏移为主
    return "overshoot_or_short"     # 越位/未达（纵向误差为主）
