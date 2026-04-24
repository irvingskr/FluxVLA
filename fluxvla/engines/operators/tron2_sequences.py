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

from typing import Any, Dict, Iterator, Optional

import numpy as np

from .tron2_types import RobotConfig


class MoveJSequence:
    def __init__(self,
                 config: RobotConfig,
                 joint_trajectory: np.ndarray,
                 *,
                 move_time: Optional[float] = None):
        self.config = config
        self.joint_trajectory = np.asarray(joint_trajectory, dtype=float)
        self.move_time = move_time if move_time is not None else (
            config.execution_time)
        self.current_step = 0

        if self.joint_trajectory.ndim != 2:
            raise ValueError('joint_trajectory must have shape (T, 14)')
        if self.joint_trajectory.shape[1] != config.arm_joint_dim:
            raise ValueError(
                f'Expected joint trajectory shape (T, {config.arm_joint_dim}), '
                f'got {self.joint_trajectory.shape}')

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        self.current_step = 0
        return self

    def __next__(self) -> Dict[str, Any]:
        if self.current_step >= self.joint_trajectory.shape[0]:
            raise StopIteration
        cmd = self.get_single_cmd(self.current_step)
        self.current_step += 1
        return cmd

    def get_single_cmd(self, step: int = 0) -> Dict[str, Any]:
        if step >= self.joint_trajectory.shape[0]:
            raise IndexError(
                f'Step {step} out of range {self.joint_trajectory.shape[0]}')
        return {
            'time': float(self.move_time),
            'joint': self.joint_trajectory[step].tolist(),
        }


class GripperSequence:
    def __init__(self, config: RobotConfig, gripper_trajectory: np.ndarray):
        self.config = config
        self.gripper_trajectory = np.asarray(gripper_trajectory, dtype=float)
        self.current_step = 0

        if self.gripper_trajectory.ndim != 2:
            raise ValueError('gripper_trajectory must have shape (T, 2)')
        if self.gripper_trajectory.shape[1] != 2:
            raise ValueError(
                f'Expected gripper trajectory shape (T, 2), got '
                f'{self.gripper_trajectory.shape}')

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        self.current_step = 0
        return self

    def __next__(self) -> Dict[str, Any]:
        if self.current_step >= self.gripper_trajectory.shape[0]:
            raise StopIteration
        cmd = self.get_single_cmd(self.current_step)
        self.current_step += 1
        return cmd

    def get_single_cmd(self, step: int = 0) -> Dict[str, Any]:
        if step >= self.gripper_trajectory.shape[0]:
            raise IndexError(
                f'Step {step} out of range {self.gripper_trajectory.shape[0]}')
        left_opening, right_opening = self.gripper_trajectory[step]
        return {
            'left_opening': float(np.clip(left_opening, 0, 100)),
            'left_speed': int(self.config.gripper_speed),
            'left_force': int(self.config.gripper_force),
            'right_opening': float(np.clip(right_opening, 0, 100)),
            'right_speed': int(self.config.gripper_speed),
            'right_force': int(self.config.gripper_force),
        }
