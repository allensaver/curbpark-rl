"""第 2 周奖励改造（消融设计见 results/week2/奖励改造设计.md）。

版本定义（除注明外其余与 v1 基线完全一致，保证消融干净）：
- v1 ：环境默认奖励，每步 −√(加权目标距离)，碰撞 −5 并终止回合。
       已知漏洞：撞停总回报(≈−14) 优于认真尝试但失败(≈−40)，PPO 学会快速撞停。
- v2a：只堵漏洞——碰撞代价 −5 → −50。
       套利账：撞停 ≈ −0.6×15 −50 ≈ −59 < 失败慢开 ≈ −40 < 成功。撞停不再划算。
- v2b：v2a + 本模块的 Goal Region 包装器（单点奖励 → 区域奖励，对应学姐
       Goal Region / Final Adjust / Final Reward 的缩小版）：
       1) 进度塑形：每步 +0.15 × (上一步位置误差 − 本步位置误差)，米制，截断 ±0.5；
       2) 区域奖励（每回合一次性，防刷分）：首次进入区域层（位置 ≤3.0m 且航向 ≤30°）+4；
          首次进入成功层（位置 ≤1.5m 且航向 ≤15°，即评测成功口径）+8；
       3) 末端精调：位于区域层内时，每步 +0.03 × max(0, 1 − 航向误差/45°)，
          量级远小于近目标处的距离惩罚（约 −0.2/步），原地刷分不划算；
       4) 终局奖励：环境判定 is_success 终止时 +15。
       每步分量写入 info["reward_parts"]，便于事后核账。
- v2c：平衡调参轮（非单因子消融，设计见 results/v2c/v2c设计.md）。
       在 v2b 结构上只动配比：碰撞 −50 → −30（common.py）、进度权重
       0.15 → 0.30、末端精调 0.03 → 0.06、终局奖励 +15 → +30；
       区域一次性奖励（+4 / +8）不变。包装器以参数形式接收这些配比，
       默认值即 v2b，保证旧版本可复现。
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

REGION_POS, REGION_HEAD = 3.0, 30.0
SUCCESS_POS, SUCCESS_HEAD = 1.5, 15.0
REGION_BONUS, SUCCESS_ZONE_BONUS, FINAL_BONUS = 4.0, 8.0, 15.0
PROGRESS_W = 0.15
ADJUST_W = 0.03


class GoalRegionRewardWrapper(gym.Wrapper):
    """v2b 区域奖励包装器。不改变观测/动作空间，只改奖励。

    progress_w / adjust_w / final_bonus 默认取 v2b 配比；v2c 在
    common.make_env 里传入调参后的配比，其余结构与 v2b 完全一致。
    """

    def __init__(self, env, progress_w: float = PROGRESS_W,
                 adjust_w: float = ADJUST_W, final_bonus: float = FINAL_BONUS):
        super().__init__(env)
        self.progress_w = progress_w
        self.adjust_w = adjust_w
        self.final_bonus = final_bonus
        self._prev_pos_err: float | None = None
        self._got_region = False
        self._got_success_zone = False

    # goal_errors 在 common.py；延迟导入避免循环依赖
    def _errors(self, obs):
        from common import goal_errors
        pos_err, head_err, _, _ = goal_errors(obs)
        return pos_err, head_err

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_pos_err, _ = self._errors(obs)
        self._got_region = False
        self._got_success_zone = False
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        pos_err, head_err = self._errors(obs)
        parts = {"base": float(reward), "progress": 0.0, "region": 0.0,
                 "adjust": 0.0, "final": 0.0}

        if self._prev_pos_err is not None:
            progress = self.progress_w * (self._prev_pos_err - pos_err)
            parts["progress"] = float(np.clip(progress, -0.5, 0.5))
        self._prev_pos_err = pos_err

        if pos_err <= REGION_POS and head_err <= REGION_HEAD:
            if not self._got_region:
                parts["region"] += REGION_BONUS
                self._got_region = True
            parts["adjust"] = self.adjust_w * max(0.0, 1.0 - head_err / 45.0)
            if pos_err <= SUCCESS_POS and head_err <= SUCCESS_HEAD and not self._got_success_zone:
                parts["region"] += SUCCESS_ZONE_BONUS
                self._got_success_zone = True

        if terminated and info.get("is_success"):
            parts["final"] = self.final_bonus

        total = reward + parts["progress"] + parts["region"] + parts["adjust"] + parts["final"]
        info = dict(info)
        info["reward_parts"] = parts
        return obs, float(total), terminated, truncated, info
