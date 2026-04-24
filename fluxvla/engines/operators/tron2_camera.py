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

from typing import Any, Dict

import numpy as np

from .tron2_types import RobotConfig


class CameraProvider:
    def get_images(self) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def close(self):
        pass


class DummyCameraProvider(CameraProvider):
    def __init__(self, height: int = 480, width: int = 640):
        self.height = height
        self.width = width

    def get_images(self) -> Dict[str, np.ndarray]:
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        return {
            'cam_high': image.copy(),
            'cam_left_wrist': image.copy(),
            'cam_right_wrist': image.copy(),
        }


class RealSenseCameraProvider(CameraProvider):
    def __init__(self, config: RobotConfig):
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError(
                'pyrealsense2 is required for RealSense camera capture') from exc

        self.rs = rs
        self.pipelines: Dict[str, Any] = {}
        serial_by_key = {
            'cam_high': config.head_camera_serial,
            'cam_left_wrist': config.left_wrist_camera_serial,
            'cam_right_wrist': config.right_wrist_camera_serial,
        }
        for key, serial in serial_by_key.items():
            pipeline = rs.pipeline()
            rs_config = rs.config()
            rs_config.enable_device(serial)
            rs_config.enable_stream(rs.stream.color, config.camera_width,
                                    config.camera_height, rs.format.rgb8,
                                    config.camera_fps)
            pipeline.start(rs_config)
            self.pipelines[key] = pipeline

    def get_images(self) -> Dict[str, np.ndarray]:
        images = {}
        for key, pipeline in self.pipelines.items():
            frames = pipeline.wait_for_frames()
            color = frames.get_color_frame()
            if not color:
                raise RuntimeError(f'No color frame from camera {key}')
            images[key] = np.asanyarray(color.get_data())
        return images

    def close(self):
        for pipeline in self.pipelines.values():
            pipeline.stop()
