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

import json
import logging
import socket
import threading
import time
import uuid
from typing import Any, Dict, Optional

from .tron2_types import RobotConfig

logger = logging.getLogger(__name__)


class WebSocketManager:
    def __init__(self, config: RobotConfig):
        try:
            import websocket
        except ImportError as exc:
            raise RuntimeError(
                'websocket-client is required for TRON2 upper-API control'
            ) from exc

        self._websocket = websocket
        self.config = config
        self.ws_url = f'ws://{config.ip_address}:5000'
        self.ws_client: Optional[Any] = None
        self.latest_robot_info: Dict[str, Any] = {}
        self.latest_notifications: Dict[str, Any] = {}
        self.responses: Dict[str, Dict[str, Any]] = {}
        self.is_connected = False
        self._connection_event = threading.Event()
        self._response_condition = threading.Condition()
        self._thread = threading.Thread(target=self._run_forever, daemon=True)
        self._thread.start()

    def wait_connected(self, timeout: float = 5.0) -> bool:
        return self._connection_event.wait(timeout)

    def _on_open(self, ws):
        logger.info('Connected to TRON2 WebSocket server at %s', self.ws_url)
        self.is_connected = True
        self._connection_event.set()

    def _on_message(self, ws, message: str):
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            logger.error('Failed to parse JSON message: %s', message)
            return

        title = data.get('title', '')
        guid = data.get('guid', '')
        if title == 'notify_robot_info':
            self.latest_robot_info = data.get('data', {})
            return

        if title.startswith('response_') and guid:
            with self._response_condition:
                self.responses[guid] = data
                self._response_condition.notify_all()
        elif title.startswith('notify_'):
            self.latest_notifications[title] = data

        logger.info('Received message: %s', message)

    def _on_close(self, ws, close_status_code, close_msg):
        logger.warning('WebSocket closed: %s %s', close_status_code, close_msg)
        self.is_connected = False
        self._connection_event.clear()

    def _on_error(self, ws, error):
        logger.error('WebSocket error: %s', error)

    def _run_forever(self):
        sockopt = [
            (socket.SOL_SOCKET, socket.SO_SNDBUF, 2 * 1024 * 1024),
            (socket.SOL_SOCKET, socket.SO_RCVBUF, 2 * 1024 * 1024),
            (socket.IPPROTO_TCP, socket.TCP_NODELAY, 1),
        ]
        self.ws_client = self._websocket.WebSocketApp(
            self.ws_url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_close=self._on_close,
            on_error=self._on_error)
        self.ws_client.run_forever(sockopt=sockopt)

    def send_request(self,
                     title: str,
                     data: Optional[Dict[str, Any]] = None,
                     *,
                     wait_response: bool = False,
                     timeout: Optional[float] = None) -> Optional[Dict[str,
                                                                      Any]]:
        if not self.is_connected or self.ws_client is None:
            raise RuntimeError('TRON2 WebSocket is not connected')

        guid = str(uuid.uuid4())
        message = {
            'accid': self.config.accid,
            'title': title,
            'timestamp': int(time.time() * 1000),
            'guid': guid,
            'data': data or {},
        }

        if wait_response:
            with self._response_condition:
                self.ws_client.send(json.dumps(message))
                deadline = time.time() + (
                    timeout or self.config.response_timeout)
                while guid not in self.responses:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        raise TimeoutError(
                            f'Timed out waiting for response to {title}')
                    self._response_condition.wait(remaining)
                return self.responses.pop(guid)

        self.ws_client.send(json.dumps(message))
        return None

    def close(self):
        if self.ws_client is not None:
            self.ws_client.close()
