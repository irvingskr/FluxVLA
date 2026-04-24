#!/usr/bin/env python3
"""TRON2 upper-API tests and FluxVLA closed-loop inference."""

from __future__ import annotations

import argparse
import copy
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [TRON2] - %(levelname)s - %(message)s')


def build_robot_config(args: argparse.Namespace) -> RobotConfig:
    from fluxvla.engines.operators.tron2_types import RobotConfig

    return RobotConfig(
        ip_address=args.ip,
        accid=args.accid,
        control_rate=getattr(args, 'control_rate', 30),
        control_horizon=getattr(args, 'horizon', 10),
        execution_time=getattr(args, 'execution_time', 0.07),
        gripper_scale=getattr(args, 'gripper_scale', 100.0),
        head_camera_serial=getattr(args, 'head_camera_serial', '343622300603'),
        left_wrist_camera_serial=getattr(args, 'left_wrist_camera_serial',
                                         '230322270826'),
        right_wrist_camera_serial=getattr(args, 'right_wrist_camera_serial',
                                          '230422272089'),
        enforce_joint_limits=not getattr(args, 'disable_joint_limits', False),
    )


def build_runner(args: argparse.Namespace):
    from mmengine import Config

    from fluxvla.engines.runners.tron2_inference_runner import (
        Tron2InferenceRunner)

    cfg = Config.fromfile(args.config)
    inference_cfg = getattr(cfg, args.inference_key)
    robot_config = build_robot_config(args)

    return Tron2InferenceRunner(
        cfg=cfg,
        seed=int(inference_cfg.get('seed', 7)),
        ckpt_path=args.ckpt_path,
        dataset=copy.deepcopy(inference_cfg.dataset),
        denormalize_action=copy.deepcopy(inference_cfg.denormalize_action),
        robot_config=robot_config,
        camera_provider='dummy' if args.dummy_images else 'realsense',
        action_chunk=int(inference_cfg.get('action_chunk', args.horizon)),
        camera_names=list(
            inference_cfg.get('camera_names',
                              ['cam_high', 'cam_left_wrist',
                               'cam_right_wrist'])),
        task_descriptions=dict(inference_cfg.get('task_descriptions', {})),
        mixed_precision_dtype=args.dtype,
        enable_mixed_precision=not args.disable_amp,
    )


def run_policy_loop(args: argparse.Namespace):
    runner = build_runner(args)
    try:
        runner.run_setup()
        instruction = runner.get_task_description(args.task)
        logging.info('Policy instruction: %s', instruction)
        runner.run_chunks(
            instruction,
            num_chunks=args.num_chunks,
            horizon=args.horizon,
            dry_run=args.dry_run)
    finally:
        runner.cleanup()


def run_basic_tests(args: argparse.Namespace):
    import numpy as np

    from fluxvla.engines.operators.tron2_operator import Tron2Operator
    from fluxvla.engines.operators.tron2_types import LightEffect, RobotConfig

    config = RobotConfig(ip_address=args.ip, accid=args.accid)
    robot = Tron2Operator(config)
    try:
        if args.test_light:
            logging.info('Testing light effect FAST_FLASH_GREEN')
            robot.set_robot_light(LightEffect.FAST_FLASH_GREEN)
            time.sleep(1.0)

        if args.test_state:
            logging.info('Testing joint state')
            joint_state = robot.get_joint_state()
            logging.info('Joint state: %s', joint_state)
            logging.info('Testing end-effector pose')
            move_pose = robot.get_move_pose()
            logging.info('Move pose: %s', move_pose)
            logging.info('Testing gripper state')
            gripper_state = robot.get_gripper_state()
            logging.info('Gripper state: %s', gripper_state)

        if args.test_gripper:
            logging.info('Testing gripper open/close')
            # New upper-API semantics: 0 = minimum closed, 100 = maximum open.
            robot.control_gripper(np.array([[100, 100], [10, 10],
                                            [50, 50]]))

        if args.test_movej:
            logging.info('Testing conservative MoveJ hold at current position')
            joint_state = robot.get_joint_state()
            if joint_state is None:
                raise RuntimeError('Cannot test MoveJ without joint state')
            current = np.asarray(joint_state['q'][:config.arm_joint_dim],
                                 dtype=float)
            robot.control_joint(
                current.reshape(1, -1), move_time=1.0, sleep_dt=1.0)
    finally:
        robot.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ip', default='10.192.1.2')
    parser.add_argument('--accid', default='DACH_TRON2A_061')

    subparsers = parser.add_subparsers(dest='command')

    test = subparsers.add_parser('test', help='Run basic upper-API tests')
    test.add_argument('--test-light', action='store_true')
    test.add_argument('--test-state', action='store_true')
    test.add_argument('--test-gripper', action='store_true')
    test.add_argument('--test-movej', action='store_true')

    infer = subparsers.add_parser('infer', help='Run FluxVLA closed-loop chunks')
    infer.add_argument(
        '--config',
        default='configs/pi05/pi05_paligemma_libero_10_lora_finetune.py')
    infer.add_argument('--ckpt-path', required=True)
    infer.add_argument('--inference-key', default='real_robot_inference')
    infer.add_argument(
        '--task',
        default='1',
        help='Task id from config.task_descriptions, or a raw instruction')
    infer.add_argument('--num-chunks', type=int, default=100)
    infer.add_argument('--horizon', type=int, default=10)
    infer.add_argument('--control-rate', type=int, default=30)
    infer.add_argument('--execution-time', type=float, default=0.2)
    infer.add_argument('--gripper-scale', type=float, default=100.0)
    infer.add_argument('--head-camera-serial', default='338122302365')
    infer.add_argument('--left-wrist-camera-serial', default='260422271874')
    infer.add_argument('--right-wrist-camera-serial', default='230322270243')
    infer.add_argument('--dtype', default='bf16')
    infer.add_argument('--disable-amp', action='store_true')
    infer.add_argument('--disable-joint-limits', action='store_true')
    infer.add_argument('--dummy-images', action='store_true')
    infer.add_argument('--dry-run', action='store_true')

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        raise SystemExit(2)
    if args.command == 'test' and not any(
            [args.test_light, args.test_state, args.test_gripper,
             args.test_movej]):
        args.test_state = True
    return args


if __name__ == '__main__':
    parsed_args = parse_args()
    if parsed_args.command == 'test':
        run_basic_tests(parsed_args)
    elif parsed_args.command == 'infer':
        run_policy_loop(parsed_args)
