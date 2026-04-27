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

import json
import os
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from safetensors.torch import load_file

from fluxvla.engines.utils.torch_utils import set_seed_everywhere
from ..utils import initialize_overwatch
from ..utils.name_map import str_to_dtype
from ..utils.root import RUNNERS

overwatch = initialize_overwatch(__name__)


@RUNNERS.register_module()
class OfflineParquetInferenceRunner:
    """Run offline inference on recorded Parquet data and dump predictions.

    This runner is intended for quick validation on recorded robot data
    before wiring up a real-robot ROS pipeline.
    """

    def __init__(self,
                 cfg: Dict,
                 seed: int,
                 ckpt_path: str,
                 dataset: Dict,
                 denormalize_action: Dict,
                 num_samples: int = 32,
                 start_index: int = 0,
                 sample_stride: int = 1,
                 save_path: str = 'offline_inference_predictions.jsonl',
                 mixed_precision_dtype: str = 'bf16',
                 enable_mixed_precision: bool = True):
        from fluxvla.engines import (build_dataset_from_cfg,
                                     build_transform_from_cfg,
                                     build_vla_from_cfg)

        self.cfg = cfg
        self.seed = seed
        self.ckpt_path = ckpt_path
        self.num_samples = num_samples
        self.start_index = start_index
        self.sample_stride = sample_stride
        self.mixed_precision_dtype = str_to_dtype(mixed_precision_dtype)
        self.enable_mixed_precision = enable_mixed_precision

        data_stat_path = os.path.join(
            Path(ckpt_path).resolve().parent.parent, 'dataset_statistics.json')
        assert os.path.exists(data_stat_path), (
            f'Dataset statistics file not found at {data_stat_path}!')

        with open(data_stat_path, 'r', encoding='utf-8') as f:
            self.norm_stats = json.load(f)

        dataset['dataset_statistics'] = self.norm_stats
        self.dataset = build_dataset_from_cfg(dataset)
        assert hasattr(self.dataset, '_get_item_from_global_idx'), (
            'OfflineParquetInferenceRunner expects a '
            'DistributedRepeatingDataset for deterministic indexing.')

        denormalize_action['norm_stats'] = data_stat_path
        self.denormalize_action = build_transform_from_cfg(denormalize_action)

        self.vla = build_vla_from_cfg(cfg.inference_model)
        if ckpt_path.endswith('.safetensors'):
            state_dict = load_file(ckpt_path, device='cpu')
        else:
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        self.vla.load_state_dict(state_dict, strict=True)
        self.vla.norm_stats = self.norm_stats

        save_path_obj = Path(save_path)
        if save_path_obj.is_absolute():
            self.save_path = save_path_obj
        else:
            self.save_path = Path(ckpt_path).resolve().parent.parent / save_path

    def run_setup(self):
        set_seed_everywhere(self.seed)
        self.vla.eval()
        if self.enable_mixed_precision:
            self.vla.to(device='cuda', dtype=self.mixed_precision_dtype)
        else:
            self.vla.cuda()
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        overwatch.info(
            f'Offline inference setup complete. Results will be written to '
            f'{self.save_path}')

    def _to_tensor(self, value, *, dtype=None):
        if torch.is_tensor(value):
            tensor = value
        elif isinstance(value, np.ndarray):
            tensor = torch.from_numpy(value)
        else:
            raise TypeError(f'Unsupported value type: {type(value)}')
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        return tensor.unsqueeze(0).cuda()

    def _build_batch(self, sample: Dict) -> Dict:
        batch = {
            'images': self._to_tensor(sample['images'], dtype=torch.float32),
            'img_masks': self._to_tensor(sample['img_masks'], dtype=torch.bool),
            'lang_tokens': self._to_tensor(
                sample['lang_tokens'], dtype=torch.long),
            'lang_masks': self._to_tensor(
                sample['lang_masks'], dtype=torch.bool),
            'states': self._to_tensor(sample['states'], dtype=torch.float32),
        }
        if 'embodiment_ids' in sample:
            batch['embodiment_ids'] = self._to_tensor(
                sample['embodiment_ids'], dtype=torch.int32)
        return batch

    def run(self):
        records = []
        total_len = getattr(self.dataset, 'total_len', 0)

        with torch.inference_mode():
            for offset in range(self.num_samples):
                sample_idx = self.start_index + offset * self.sample_stride
                if sample_idx >= total_len:
                    break

                sample = self.dataset._get_item_from_global_idx(sample_idx)
                batch = self._build_batch(sample)

                with torch.autocast(
                        'cuda',
                        dtype=self.mixed_precision_dtype,
                        enabled=self.enable_mixed_precision):
                    raw_action = self.vla.predict_action(**batch)

                pred_action = self.denormalize_action(
                    dict(action=raw_action.float().cpu().numpy()))
                gt_action = self.denormalize_action(
                    dict(action=np.asarray(sample['actions'])[None]))

                action_masks = sample.get('action_masks')
                if action_masks is None:
                    valid_steps = pred_action.shape[0]
                else:
                    valid_steps = int(np.asarray(action_masks).sum())
                    valid_steps = max(valid_steps, 1)

                pred_actions = np.asarray(pred_action)[:valid_steps]
                gt_actions = np.asarray(gt_action)[:valid_steps]
                action_dim = min(pred_actions.shape[-1], gt_actions.shape[-1])
                pred_actions = pred_actions[..., :action_dim]
                gt_actions = gt_actions[..., :action_dim]
                mse = float(np.mean((pred_actions - gt_actions)**2))

                record = {
                    'sample_index': sample_idx,
                    'task_description': sample.get('task_description', ''),
                    'episode_index': int(sample.get('episode_index', -1)),
                    'timestamp': float(sample.get('timestamp', 0.0)),
                    'valid_steps': valid_steps,
                    'chunk_mse': mse,
                    'pred_first_action': pred_actions[0].tolist(),
                    'gt_first_action': gt_actions[0].tolist(),
                }
                records.append(record)

        with open(self.save_path, 'w', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

        if records:
            mean_mse = sum(item['chunk_mse'] for item in records) / len(records)
            overwatch.info(
                f'Offline inference finished on {len(records)} samples. '
                f'Mean chunk MSE: {mean_mse:.6f}')
        else:
            overwatch.warning('Offline inference finished with no samples.')
