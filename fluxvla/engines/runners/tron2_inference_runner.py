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
from typing import Any, Dict, Optional

import numpy as np
import torch

from fluxvla.engines.operators.tron2_camera import (CameraProvider,
                                                    DummyCameraProvider,
                                                    RealSenseCameraProvider)
from fluxvla.engines.operators.tron2_types import ActionLayout, RobotConfig
from fluxvla.engines.utils.root import RUNNERS

from .base_inference_runner import BaseInferenceRunner

logger = logging.getLogger(__name__)


@RUNNERS.register_module()
class Tron2InferenceRunner(BaseInferenceRunner):
    """TRON2 upper-API inference runner.

    The runner owns FluxVLA model inference and converts between TRON2 sensor /
    actuator data and the model's 16-dim state/action layout. Hardware I/O is
    delegated to Tron2Operator and camera providers.
    """

    def __init__(self,
                 cfg: Dict,
                 seed: int,
                 ckpt_path: str,
                 dataset: Dict,
                 denormalize_action: Dict,
                 robot_config: Optional[RobotConfig] = None,
                 camera_provider: str | CameraProvider = 'realsense',
                 action_layout: Optional[ActionLayout] = None,
                 action_chunk: int = 10,
                 camera_names: Optional[list[str]] = None,
                 operator: Optional[Dict[str, Any]] = None,
                 task_descriptions: Optional[Dict[str, str]] = None,
                 mixed_precision_dtype: str = 'bf16',
                 enable_mixed_precision: bool = True,
                 **kwargs):
        self.robot_config = robot_config or RobotConfig()
        self.action_layout = action_layout or ActionLayout()
        self._camera_provider_spec = camera_provider
        self.camera_provider: Optional[CameraProvider] = None

        if operator is None:
            operator = {
                'type': 'Tron2Operator',
                'config': self.robot_config,
            }

        super().__init__(
            cfg=cfg,
            seed=seed,
            ckpt_path=ckpt_path,
            dataset=dataset,
            denormalize_action=denormalize_action,
            state_dim=self.robot_config.policy_action_dim,
            action_chunk=action_chunk,
            publish_rate=self.robot_config.control_rate,
            camera_names=camera_names
            or ['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
            operator=operator,
            task_descriptions=task_descriptions,
            mixed_precision_dtype=mixed_precision_dtype,
            enable_mixed_precision=enable_mixed_precision,
            **kwargs)

        try:
            self.camera_provider = self._build_camera_provider(camera_provider)
        except Exception:
            if hasattr(self.ros_operator, 'close'):
                self.ros_operator.close()
            raise

    def _build_camera_provider(
            self, camera_provider: str | CameraProvider) -> CameraProvider:
        if isinstance(camera_provider, CameraProvider):
            return camera_provider
        if camera_provider == 'dummy':
            logger.warning('Using dummy black images. Do not use this for real '
                           'closed-loop robot execution.')
            return DummyCameraProvider(self.robot_config.camera_height,
                                       self.robot_config.camera_width)
        if camera_provider == 'realsense':
            return RealSenseCameraProvider(self.robot_config)
        raise ValueError(f'Unsupported camera provider: {camera_provider}')

    def get_task_description(self, task: str) -> str:
        return self._get_task_description(task)

    def get_ros_observation(self):
        """Return a TRON2 upper-API observation tuple.

        The method name follows BaseInferenceRunner's robot-observation hook,
        although TRON2 does not use ROS topics here.
        """
        joint_state = self.ros_operator.get_joint_state()
        if joint_state is None or 'q' not in joint_state:
            raise RuntimeError('Cannot build observation without joint state')

        gripper_state = self.ros_operator.get_gripper_state()
        if gripper_state is None:
            raise RuntimeError('Cannot build observation without gripper state')

        if self.camera_provider is None:
            raise RuntimeError('Camera provider has not been initialized')
        images = self.camera_provider.get_images()
        return joint_state, gripper_state, images

    def update_observation_window(self) -> Dict:
        joint_state, gripper_state, images = self.get_ros_observation()
        arm_q = np.asarray(joint_state['q'][:self.robot_config.arm_joint_dim],
                           dtype=np.float32)
        if arm_q.shape[0] != self.robot_config.arm_joint_dim:
            raise RuntimeError(
                f'Expected at least {self.robot_config.arm_joint_dim} arm '
                f'joints, got {arm_q}')

        left_gripper = float(gripper_state.get('left_opening', 100.0))
        right_gripper = float(gripper_state.get('right_opening', 100.0))
        qpos = self.action_layout.pack_state(
            arm_q[:7],
            arm_q[7:14],
            left_gripper / self.robot_config.gripper_scale,
            right_gripper / self.robot_config.gripper_scale,
        )
        return {'qpos': qpos, **images}

    def _postprocess_actions(self, raw_action):
        denormalized = self.denormalize_action(
            dict(action=raw_action.float().cpu().numpy()))
        return np.asarray(denormalized[:self.action_chunk], dtype=float)

    def predict_actions(self, instruction: str) -> np.ndarray:
        inputs = self._preprocess(instruction)
        with torch.inference_mode():
            with torch.autocast(
                    'cuda',
                    dtype=self.mixed_precision_dtype,
                    enabled=self.enable_mixed_precision):
                raw_action = self._predict_action(inputs)
        return self._postprocess_actions(raw_action)

    def _execute_actions(self, actions: np.ndarray, rate=None):
        self.ros_operator.execute_policy_actions(
            actions, layout=self.action_layout, max_steps=actions.shape[0])
        if rate is not None:
            rate.sleep()

    def run_chunks(self,
                   instruction: str,
                   *,
                   num_chunks: int,
                   horizon: Optional[int] = None,
                   dry_run: bool = False):
        for episode_step in range(num_chunks):
            actions = self.predict_actions(instruction)
            logger.info('Chunk %d predicted actions shape=%s first=%s',
                        episode_step, actions.shape, actions[0].tolist())
            if dry_run:
                continue
            max_steps = min(horizon or actions.shape[0], actions.shape[0])
            self._execute_actions(actions[:max_steps])

    def cleanup(self):
        super().cleanup()
        if self.camera_provider is not None:
            self.camera_provider.close()
        if hasattr(self.ros_operator, 'close'):
            self.ros_operator.close()
