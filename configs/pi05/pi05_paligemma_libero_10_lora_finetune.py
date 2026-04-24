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

model = dict(
    type='PI05FlowMatching',
    llm_backbone=dict(
        type='ConditionGemmaModel',
        adarms_cond_dim=None,
        attention_bias=False,
        attention_dropout=0.0,
        bos_token_id=2,
        eos_token_id=1,
        head_dim=256,
        hidden_act='gelu_pytorch_tanh',
        hidden_activation='gelu_pytorch_tanh',
        hidden_size=2048,
        initializer_range=0.02,
        intermediate_size=16384,
        max_position_embeddings=8192,
        model_type='gemma',
        num_attention_heads=8,
        num_hidden_layers=18,
        num_key_value_heads=1,
        rms_norm_eps=1e-06,
        rope_theta=10000.0,
        torch_dtype='float32',
        use_cache=True,
        vocab_size=257152,
    ),
    vision_backbone=dict(
        type='SigLIPViTBackbone',
        vision_backbone_id='siglip_224',
        vision_config=dict(
            attention_dropout=0.0,
            hidden_act='gelu_pytorch_tanh',
            hidden_size=1152,
            image_size=224,
            intermediate_size=4304,
            layer_norm_eps=1e-06,
            model_type='siglip_vision_model',
            num_attention_heads=16,
            num_channels=3,
            num_hidden_layers=27,
            patch_size=14,
            projection_dim=2048,
            projector_hidden_act='gelu_fast',
            torch_dtype='float32',
            vision_use_head=False,
        ),
    ),
    projector=dict(
        type='LinearProjector',
        in_dim=1152,
        out_dim=2048,
    ),
    proj_width=1024,
    n_action_steps=10,
    action_in_proj=dict(type='LinearProjector', in_dim=32, out_dim=1024),
    action_out_proj=dict(type='LinearProjector', in_dim=1024, out_dim=32),
    time_mlp_in=dict(type='LinearProjector', in_dim=1024, out_dim=1024),
    time_mlp_out=dict(type='LinearProjector', in_dim=1024, out_dim=1024),
    max_action_dim=32,
    llm_expert=dict(
        type='ConditionGemmaModel',
        attention_bias=False,
        adarms_cond_dim=1024,
        attention_dropout=0.0,
        bos_token_id=2,
        eos_token_id=1,
        head_dim=256,
        hidden_act='gelu_pytorch_tanh',
        hidden_activation='gelu_pytorch_tanh',
        hidden_size=1024,
        initializer_range=0.02,
        intermediate_size=4096,
        max_position_embeddings=8192,
        model_type='gemma',
        num_attention_heads=8,
        num_hidden_layers=18,
        num_key_value_heads=1,
        pad_token_id=0,
        rms_norm_eps=1e-06,
        rope_theta=10000.0,
        torch_dtype='float32',
        transformers_version='4.48.1',
        use_adarms=True,
        use_cache=True,
        vocab_size=257152),
    freeze_llm_backbone=False,
    freeze_vision_backbone=False,
    pretrained_name_or_path=  # noqa: E251
    './checkpoints/pi05_libero/model.safetensors',  # noqa: E501
    name_mapping={
        'llm_backbone': 'paligemma_with_expert.paligemma.model.language_model',
        'vision_backbone.vision':
        'paligemma_with_expert.paligemma.model.vision_tower',
        'projector.projector':
        'paligemma_with_expert.paligemma.model.multi_modal_projector.linear',
        'llm_expert': 'paligemma_with_expert.gemma_expert.model',
        'time_mlp_in.projector': 'time_mlp_in',
        'time_mlp_out.projector': 'time_mlp_out',
        'action_in_proj.projector': 'action_in_proj',
        'action_out_proj.projector': 'action_out_proj',
        'llm_backbone.embed_tokens': 'paligemma_with_expert.paligemma.lm_head',
    },
    use_lora=True,
    lora_rank=256,
    lora_alpha=512,
    lora_dropout=0.0,
    ori_action_dim=16,
    lora_target_modules=[
        # Attention projections (llm_backbone & llm_expert)
        'q_proj',
        'v_proj',
        'k_proj',
        'o_proj',
        # MLP layers (llm_backbone & llm_expert)
        'gate_proj',
        'up_proj',
        'down_proj',
        # Vision projector (image -> LLM embedding)
        'projector.projector',
        # Vision backbone attention (SigLIP)
        'out_proj',
        'fc1',
        'fc2',
    ],
    modules_to_save=[
        'action_in_proj',
        'action_out_proj',
        'time_mlp_in',
        'time_mlp_out',
    ])

train_dataloader = dict(
    per_device_batch_size=16,
    per_device_num_workers=4,
    dataset=dict(
        type='DistributedRepeatingDataset',
        name_mappings={
            'observation.state': ['proprio'],
            'action': ['action']
        },
        statistic_keys=['observation.state', 'action'],
        statistic_name='libero_10_no_noops',
        datasets=dict(
            type='ParquetDataset',
            data_root_path=  # noqa: E251
            './datasets/banana_lerobot',  # noqa: E501
            transforms=[
                dict(
                    type='ProcessParquetInputs',
                    parquet_keys=[
                        'observation.state', 'timestamp', 'actions', 'info',
                        'stats', 'action_masks'
                    ],
                    video_keys=[
                        'observation.images.cam_high',
                        'observation.images.cam_left_wrist',
                        'observation.images.cam_right_wrist',
                    ],
                    name_mappings={
                        'observation.state': ['states'],
                        'actions': ['actions']
                    }),
                dict(type='ParquetPrompter', use_conversation=False),
                dict(
                    type='ProcessPrompts',
                    tokenizer=dict(type='PaligemmaTokenizer'
                                   # special_tokens={'pad_token': '<PAD>'}
                                   )),
                dict(type='ResizeImages', height=224, width=224),
                dict(
                    type='NormalizeImages',
                    means=[[123.515625, 116.04492188, 103.59375],
                           [123.515625, 116.04492188, 103.59375],
                           [123.515625, 116.04492188, 103.59375]],
                    stds=[[58.27148438, 57.02636719, 57.27539062],
                          [58.27148438, 57.02636719, 57.27539062],
                          [58.27148438, 57.02636719, 57.27539062]],
                ),
                dict(
                    type='NormalizeStatesAndActions',
                    action_dim=32,
                    state_dim=32,
                    state_key='proprio',
                    action_key='action',
                    norm_type='mean_std')
            ],
            action_window_size=10,
            action_key='action',
            use_delta=False,
            statistic_name='libero_10_no_noops',
            window_start_idx=0)))

runner = dict(
    type='DDPTrainRunner',
    max_epochs=24,
    learning_rate=5e-5,
    weight_decay=0.0,
    max_grad_norm=1.0,
    sharding_strategy='no-shard',
    collator=dict(
        type='DictCollator',
        keys=[
            'states', 'observation.eepose', 'timestamp', 'images', 'img_masks',
            'lang_tokens', 'lang_masks', 'actions', 'action_masks'
        ],
        meta_keys=['task_description', 'prompt', 'info', 'stats']),
    sampler=None,
    tokenizer=dict(type='PaligemmaTokenizer'
                   # special_tokens={'pad_token': '<PAD>'}
                   ),
    metric=dict(
        type='VLAMetric',
        active_trackers=('jsonl', 'wandb'),
        run_dir='work_dirs',
        grad_accumulation_steps=1,
        window_size=1),
    lr_scheduler_type='linear-warmup+cosine-decay',
    warmup_ratio=0.03,
    enable_gradient_checkpointing=True,
    enable_mixed_precision_training=True,
    mixed_precision_dtype='bf16',
    change_key_name=False)

eval = dict(
    type='LiberoEvalRunner',
    task_suite_name='libero_10',
    model_family='pi0',
    eval_chunk_size=10,
    resize_size=224,
    num_trials_per_task=50,
    num_steps_wait=10,
    seed=7,
    dataset=dict(
        type='LiberoParquetEvalDataset',
        transforms=[
            dict(
                type='ProcessLiberoEvalInputs',
                img_keys=['agentview_image', 'robot0_eye_in_hand_image'],
            ),
            dict(
                type='TransformImage',
                image_resize_strategy='resize-naive',
                input_sizes=[[3, 224, 224], [3, 224, 224]],
                means=[[123.515625, 116.04492188, 103.59375],
                       [123.515625, 116.04492188, 103.59375]],
                stds=[[58.27148438, 57.02636719, 57.27539062],
                      [58.27148438, 57.02636719, 57.27539062]],
            ),
            dict(
                type='LiberoPromptFromInputs',
                use_conversation=False,
                tokenizer=dict(type='PaligemmaTokenizer'
                               # special_tokens={'pad_token': '<PAD>'}
                               )),
            dict(
                type='LiberoProprioFromInputs',
                norm_type='mean_std',
                pos_key='robot0_eef_pos',
                quat_key='robot0_eef_quat',
                gripper_key='robot0_gripper_qpos',
                state_dim=32,
                out_key='states'),
        ]),
    denormalize_action=dict(
        type='DenormalizeLiberoAction',
        norm_type='mean_std',
        action_dim=7,
    ),
)

inference_model = model.copy()


# Text-only inference transforms. This matches the current training setup,
# where proprio was not injected into the PI05 prompt.
_text_only_inference_transforms = [
    dict(type='ParquetPrompter', use_conversation=False),
    dict(
        type='ProcessPrompts',
        tokenizer=dict(type='PaligemmaTokenizer')),
    dict(type='ResizeImages', height=224, width=224),
    dict(
        type='NormalizeImages',
        means=[[123.515625, 116.04492188, 103.59375],
               [123.515625, 116.04492188, 103.59375],
               [123.515625, 116.04492188, 103.59375]],
        stds=[[58.27148438, 57.02636719, 57.27539062],
              [58.27148438, 57.02636719, 57.27539062],
              [58.27148438, 57.02636719, 57.27539062]],
    ),
]

# Offline validation on recorded Parquet data. This is the default inference
# path so we can verify the checkpoint before touching a real robot.
offline_inference = dict(
    type='OfflineParquetInferenceRunner',
    seed=7,
    num_samples=32,
    start_index=0,
    sample_stride=20,
    save_path='offline_inference_predictions.jsonl',
    dataset=dict(
        type='DistributedRepeatingDataset',
        shuffle=False,
        seed=7,
        name_mappings={
            'observation.state': ['proprio'],
            'action': ['action']
        },
        statistic_keys=['observation.state', 'action'],
        statistic_name='libero_10_no_noops',
        datasets=dict(
            type='ParquetDataset',
            data_root_path='./datasets/banana2',
            transforms=[
                dict(
                    type='ProcessParquetInputs',
                    parquet_keys=[
                        'observation.state', 'timestamp', 'actions', 'info',
                        'stats', 'action_masks'
                    ],
                    video_keys=[
                        'observation.images.cam_high',
                        'observation.images.cam_left_wrist',
                        'observation.images.cam_right_wrist',
                    ],
                    name_mappings={
                        'observation.state': ['states'],
                        'actions': ['actions']
                    }),
                *_text_only_inference_transforms,
            ],
            action_window_size=10,
            action_key='action',
            use_delta=False,
            statistic_name='libero_10_no_noops',
            window_start_idx=0,
        )),
    denormalize_action=dict(
        type='DenormalizePrivateAction',
        norm_type='mean_std',
        stats_key='libero_10_no_noops',
        action_dim=16,
    ),
)

# Real-robot policy template for TRON2 upper-API deployment.
#
# This block is consumed by `scripts/tron2_inference.py infer` to reuse the
# FluxVLA model/input/denormalization pipeline on DACH_TRON2A. It is not a
# registry runner config and should not be passed to
# `scripts/inference_real_robot.py`, because the existing ALOHA/UR runners are
# ROS1-topic based.
real_robot_inference = dict(
    robot_type='DACH_TRON2A',
    control_interface='websocket_upper_api',
    action_layout='left7_gripper_right7_gripper',
    state_layout='left7_gripper_right7_gripper',
    gripper_scale=100.0,
    camera_names=['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
    task_descriptions={
        '1': 'pick up the banana from the desk and place it on the plate',
    },
    seed=7,
    dataset=dict(
        type='PrivateInferenceDataset',
        img_keys=['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
        stats_key='libero_10_no_noops',
        transforms=_text_only_inference_transforms,
    ),
    denormalize_action=dict(
        type='DenormalizePrivateAction',
        norm_type='mean_std',
        stats_key='libero_10_no_noops',
        action_dim=16,
    ),
    action_chunk=10,
)

# Use offline inference by default. For DACH_TRON2A real-robot deployment,
# keep this line unchanged and run `scripts/tron2_inference.py infer`, which
# reads `real_robot_inference` directly without constructing a ROS runner.
inference = offline_inference
