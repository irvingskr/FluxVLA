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

import logging
import time
from typing import Any, Dict, Optional

import numpy as np

from fluxvla.engines.utils.root import OPERATORS

from .tron2_sequences import GripperSequence, MoveJSequence
from .tron2_types import (TRON2_MOVEJ_LOWER_LIMITS, TRON2_MOVEJ_UPPER_LIMITS,
                          ActionLayout, LightEffect, RobotConfig)
from .tron2_websocket import WebSocketManager

logger = logging.getLogger(__name__)


@OPERATORS.register_module()
class Tron2Operator:
    """TRON2 upper-API operator backed by WebSocket requests."""

    def __init__(self,
                 config: Optional[RobotConfig] = None,
                 **robot_config_kwargs):
        self.config = config or RobotConfig(**robot_config_kwargs)
        self.ws_manager = WebSocketManager(self.config)
        if not self.ws_manager.wait_connected(timeout=5.0):
            raise TimeoutError(
                f'Could not connect to TRON2 WebSocket at '
                f'{self.config.ip_address}')
        logger.info('TRON2 upper-API controller is ready: %s',
                    self.config.accid)

    def send_request(self,
                     title: str,
                     data: Optional[Dict[str, Any]] = None,
                     *,
                     wait_response: bool = False,
                     timeout: Optional[float] = None) -> Optional[Dict[str,
                                                                      Any]]:
        return self.ws_manager.send_request(
            title, data, wait_response=wait_response, timeout=timeout)

    def get_state(self) -> Optional[Dict[str, Any]]:
        return self.get_joint_state()

    def get_joint_state(self) -> Optional[Dict[str, Any]]:
        return self._request_success_data('request_get_joint_state',
                                          'joint state')

    def get_move_pose(self) -> Optional[Dict[str, Any]]:
        return self._request_success_data('request_get_move_pose', 'move pose')

    def get_gripper_state(self) -> Optional[Dict[str, Any]]:
        return self._request_success_data('request_get_limx_2fclaw_state',
                                          'gripper state')

    def _request_success_data(self, title: str,
                              label: str) -> Optional[Dict[str, Any]]:
        response = self.send_request(title, wait_response=True)
        data = (response or {}).get('data', {})
        if data.get('result') != 'success':
            logger.warning('Failed to get %s: %s', label, data)
            return None
        return data

    def validate_joint_positions(self, joints: np.ndarray) -> bool:
        joints = np.asarray(joints, dtype=float)
        if joints.shape[-1] != self.config.arm_joint_dim:
            logger.error('Expected %d arm joints, got shape %s',
                         self.config.arm_joint_dim, joints.shape)
            return False
        below = joints < TRON2_MOVEJ_LOWER_LIMITS
        above = joints > TRON2_MOVEJ_UPPER_LIMITS
        if not (below.any() or above.any()):
            return True

        bad_indices = np.where(below | above)[0].tolist()
        logger.error('Joint command exceeds TRON2 MoveJ limits at %s: %s',
                     bad_indices, joints[bad_indices].tolist())
        return False

    def clip_joint_positions(self, joints: np.ndarray) -> np.ndarray:
        joints = np.asarray(joints, dtype=float)
        if joints.shape[-1] != self.config.arm_joint_dim:
            raise ValueError(f'Expected {self.config.arm_joint_dim} arm joints, '
                             f'got shape {joints.shape}')

        clipped = np.clip(joints, TRON2_MOVEJ_LOWER_LIMITS,
                          TRON2_MOVEJ_UPPER_LIMITS)
        changed = np.where(np.abs(clipped - joints) > 1e-9)[0]
        if changed.size > 0:
            bad_indices = changed.tolist()
            logger.warning(
                'Clipping TRON2 MoveJ joints at %s from %s to %s',
                bad_indices,
                joints[bad_indices].tolist(),
                clipped[bad_indices].tolist(),
            )
        return clipped

    def control_joint(self,
                      joint_trajectory: np.ndarray,
                      *,
                      move_time: Optional[float] = None,
                      sleep_dt: Optional[float] = None):
        sequence = MoveJSequence(
            self.config, joint_trajectory, move_time=move_time)
        dt = sleep_dt if sleep_dt is not None else self.config.execution_time
        for cmd_data in sequence:
            joints = np.asarray(cmd_data['joint'], dtype=float)
            if self.config.enforce_joint_limits:
                joints = self.clip_joint_positions(joints)
                cmd_data['joint'] = joints.tolist()
            self.send_request('request_movej', cmd_data)
            time.sleep(dt)

    def control_servoj(self,
                       q_trajectory: np.ndarray,
                       *,
                       filter_ratio: Optional[float] = None,
                       sleep_dt: Optional[float] = None):
        q_trajectory = np.asarray(q_trajectory, dtype=float)
        if q_trajectory.ndim != 2 or q_trajectory.shape[1] != 16:
            raise ValueError(
                f'ServoJ expects trajectory shape (T, 16), got '
                f'{q_trajectory.shape}')
        dt = sleep_dt if sleep_dt is not None else 1.0 / self.config.control_rate
        ratio = self.config.filter_ratio if filter_ratio is None else filter_ratio
        for q in q_trajectory:
            self.send_request(
                'request_servoj', {
                    'filter_ratio': float(ratio),
                    'q': q.tolist()
                })
            time.sleep(dt)

    def control_gripper(self, gripper_trajectory: np.ndarray):
        for cmd_data in GripperSequence(self.config, gripper_trajectory):
            self.send_request('request_set_limx_2fclaw_cmd', cmd_data)
            time.sleep(self.config.execution_time)

    def control_velocity(self,
                         velocity_sequence: np.ndarray,
                         *,
                         dt: Optional[float] = None):
        dt = dt if dt is not None else self.config.execution_time
        if dt <= 0:
            raise ValueError('dt must be positive')

        state = self.get_joint_state()
        if state is None or 'q' not in state:
            raise RuntimeError('Cannot get current joint state')

        current_q = np.asarray(state['q'][:self.config.arm_joint_dim],
                               dtype=float)
        velocity_sequence = np.asarray(velocity_sequence, dtype=float)
        if velocity_sequence.ndim != 2:
            raise ValueError('velocity_sequence must have shape (T, 14)')
        if velocity_sequence.shape[1] != self.config.arm_joint_dim:
            raise ValueError(
                f'Expected velocity dim {self.config.arm_joint_dim}, got '
                f'{velocity_sequence.shape[1]}')

        joint_trajectory = []
        for dq in velocity_sequence:
            current_q = current_q + dq * dt
            joint_trajectory.append(current_q.copy())
        self.control_joint(np.asarray(joint_trajectory), sleep_dt=dt)

    def set_robot_light(self, effect: int | LightEffect | str):
        if isinstance(effect, str):
            effect_id = int(LightEffect[effect].value)
        else:
            effect_id = int(effect)
        self.send_request(
            'request_light_effect', {'effect': effect_id}, wait_response=False)

    def emergency_stop(self):
        self.send_request('request_emgy_stop', wait_response=False)

    def execute_policy_actions(self,
                               actions: np.ndarray,
                               *,
                               layout: ActionLayout = ActionLayout(),
                               max_steps: Optional[int] = None):
        actions = np.asarray(actions, dtype=float)
        if actions.ndim != 2:
            raise ValueError(f'Policy actions must have shape (T, D), got '
                             f'{actions.shape}')
        if max_steps is not None:
            actions = actions[:max_steps]

        joint_trajectory, gripper_values = layout.split(actions)
        gripper_trajectory = np.clip(
            gripper_values * self.config.gripper_scale, 0.0, 100.0)

        for joints, grippers in zip(joint_trajectory, gripper_trajectory):
            if self.config.enforce_joint_limits:
                joints = self.clip_joint_positions(joints)
            self.send_request('request_movej', {
                'time': float(self.config.execution_time),
                'joint': joints.tolist(),
            })
            self.send_request(
                'request_set_limx_2fclaw_cmd',
                GripperSequence(self.config,
                                grippers.reshape(1, 2)).get_single_cmd(0))
            time.sleep(self.config.execution_time)

    def close(self):
        self.ws_manager.close()
