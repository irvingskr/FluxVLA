# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Tuple

import numpy as np


class LightEffect(IntEnum):
    STATIC_RED = 1
    STATIC_GREEN = 2
    STATIC_BLUE = 3
    STATIC_CYAN = 4
    STATIC_PURPLE = 5
    STATIC_YELLOW = 6
    STATIC_WHITE = 7
    LOW_FLASH_RED = 8
    LOW_FLASH_GREEN = 9
    LOW_FLASH_BLUE = 10
    LOW_FLASH_CYAN = 11
    LOW_FLASH_PURPLE = 12
    LOW_FLASH_YELLOW = 13
    LOW_FLASH_WHITE = 14
    FAST_FLASH_RED = 15
    FAST_FLASH_GREEN = 16
    FAST_FLASH_BLUE = 17
    FAST_FLASH_CYAN = 18
    FAST_FLASH_PURPLE = 19
    FAST_FLASH_YELLOW = 20
    FAST_FLASH_WHITE = 21


@dataclass
class RobotConfig:
    ip_address: str = '10.192.1.2'
    accid: str = 'DACH_TRON2A_061'
    control_rate: int = 30
    control_horizon: int = 10
    execution_time: float = 0.07
    response_timeout: float = 2.0
    filter_ratio: float = 1.0
    arm_joint_dim: int = 14
    policy_action_dim: int = 16
    gripper_scale: float = 100.0
    gripper_speed: int = 50
    gripper_force: int = 25
    enforce_joint_limits: bool = True
    left_wrist_camera_serial: str = '260422271874'
    right_wrist_camera_serial: str = '230322270243'
    head_camera_serial: str = '338122302365'
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30


@dataclass(frozen=True)
class ActionLayout:
    """Policy layout: left 7 joints + left gripper + right 7 + right gripper."""

    left_joint_indices: Tuple[int, ...] = tuple(range(0, 7))
    left_gripper_index: int = 7
    right_joint_indices: Tuple[int, ...] = tuple(range(8, 15))
    right_gripper_index: int = 15

    def pack_state(self, left_joints: np.ndarray, right_joints: np.ndarray,
                   left_gripper: float, right_gripper: float) -> np.ndarray:
        state = np.zeros(16, dtype=np.float32)
        state[list(self.left_joint_indices)] = np.asarray(
            left_joints, dtype=np.float32)
        state[self.left_gripper_index] = np.float32(left_gripper)
        state[list(self.right_joint_indices)] = np.asarray(
            right_joints, dtype=np.float32)
        state[self.right_gripper_index] = np.float32(right_gripper)
        return state

    def split(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        action = np.asarray(action, dtype=float)
        if action.shape[-1] < 16:
            raise ValueError(
                f'Policy action must have at least 16 dims, got {action.shape}')
        joints = np.concatenate([
            action[..., list(self.left_joint_indices)],
            action[..., list(self.right_joint_indices)],
        ],
                                axis=-1)
        grippers = np.stack([
            action[..., self.left_gripper_index],
            action[..., self.right_gripper_index],
        ],
                            axis=-1)
        return joints, grippers


TRON2_MOVEJ_LOWER_LIMITS = np.array([
    -3.1416, -0.2618, -3.6652, -2.618, -1.7453, -0.7854, -1.5708,
    -3.1416, -0.2618, -3.6652, -2.618, -1.7453, -0.7854, -1.5708
],
                                    dtype=float)
TRON2_MOVEJ_UPPER_LIMITS = np.array([
    2.5994, 2.9671, 1.4835, 0.5236, 1.3963, 0.7854, 1.5708, 2.5994, 2.9671,
    1.4835, 0.5236, 1.3963, 0.7854, 1.5708
],
                                    dtype=float)
