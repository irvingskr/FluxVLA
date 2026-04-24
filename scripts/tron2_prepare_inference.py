#!/usr/bin/env python3
"""Safely move TRON2 into and out of the FluxVLA inference-ready pose."""

from __future__ import annotations

import argparse
import ast
import json
import logging
import time
from typing import Iterable, Sequence

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [TRON2_PREP] - %(levelname)s - %(message)s')

ARM_JOINT_DIM = 14

# User-validated start/home pose used as the safe non-inference resting pose.
HOME_JOINTS = [
    0.0,
    0.0,
    1.48,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    -1.48,
    0.0,
    0.0,
    0.0,
    0.0,
]

# User-selected side-clear waypoint for clearing the front desk before moving
# into or out of the inference-ready pose.
SIDE_CLEAR_JOINTS = [
    0.0,
    0.0,
    1.48,
    -1.60,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    -1.48,
    -1.60,
    0.0,
    0.0,
    0.0,
]

# User-validated inference-ready pose.
INFERENCE_READY_JOINTS = [
    0.01040029525756836,
    0.23749971389770508,
    -0.0005000273231416941,
    -1.5515998601913452,
    0.2314000129699707,
    0.006600022315979004,
    -0.0009000936988741159,
    0.00839996337890625,
    -0.24019986391067505,
    0.0020999908447265625,
    -1.5484002828598022,
    -0.23440009355545044,
    0.0023000240325927734,
    0.0,
]

DEFAULT_GRIPPER_OPENINGS = [95.0, 95.0]


def parse_joint_list(raw: str) -> list[float]:
    try:
        value = ast.literal_eval(raw)
    except (ValueError, SyntaxError) as exc:
        raise argparse.ArgumentTypeError(
            f'Could not parse joint list: {raw}') from exc
    if not isinstance(value, (list, tuple)):
        raise argparse.ArgumentTypeError('Joint override must be a list/tuple')
    if len(value) != ARM_JOINT_DIM:
        raise argparse.ArgumentTypeError(
            f'Expected {ARM_JOINT_DIM} joints, got {len(value)}')
    try:
        return [float(v) for v in value]
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            'Joint override contains a non-numeric value') from exc


def parse_gripper_pair(raw: str) -> list[float]:
    parts = [part.strip() for part in raw.split(',')]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            'Gripper openings must look like "95,95"')
    try:
        return [float(parts[0]), float(parts[1])]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            'Gripper openings must be numeric') from exc


def joints_to_compact_str(joints: Sequence[float]) -> str:
    return '[' + ', '.join(f'{value:.3f}' for value in joints) + ']'


def build_robot_config(args: argparse.Namespace):
    from fluxvla.engines.operators.tron2_types import RobotConfig

    return RobotConfig(
        ip_address=args.ip,
        accid=args.accid,
        execution_time=args.command_gap,
        gripper_speed=args.gripper_speed,
        gripper_force=args.gripper_force,
    )


def build_plan(
        args: argparse.Namespace) -> tuple[str, list[list[float]], list[float]]:
    if args.command == 'go-init':
        return ('home -> side_clear -> inference_ready',
                [args.side_joints, args.ready_joints], args.grippers)
    if args.command == 'go-home':
        return ('inference_ready -> side_clear -> home',
                [args.side_joints, args.home_joints], args.grippers)
    raise ValueError(f'Unsupported command: {args.command}')


def iter_preview_payloads(
        accid: str,
        move_time: float,
        waypoints: Iterable[Sequence[float]],
        grippers: Sequence[float],
) -> list[dict]:
    timestamp = int(time.time() * 1000)
    payloads = [{
        'accid': accid,
        'title': 'request_set_limx_2fclaw_cmd',
        'timestamp': timestamp,
        'guid': 'dry-run-gripper',
        'data': {
            'left_opening': float(grippers[0]),
            'left_speed': 'from RobotConfig.gripper_speed',
            'left_force': 'from RobotConfig.gripper_force',
            'right_opening': float(grippers[1]),
            'right_speed': 'from RobotConfig.gripper_speed',
            'right_force': 'from RobotConfig.gripper_force',
        }
    }]
    for idx, joints in enumerate(waypoints, start=1):
        payloads.append({
            'accid': accid,
            'title': 'request_movej',
            'timestamp': timestamp,
            'guid': f'dry-run-movej-{idx}',
            'data': {
                'time': float(move_time),
                'joint': [float(value) for value in joints],
            }
        })
    return payloads


def preview_plan(args: argparse.Namespace, label: str,
                 waypoints: list[list[float]], grippers: list[float]):
    logging.info('Dry-run motion: %s', label)
    logging.info('Grippers: left=%.1f right=%.1f', grippers[0], grippers[1])
    for index, joints in enumerate(waypoints, start=1):
        logging.info('Waypoint %d: %s', index, joints_to_compact_str(joints))
    for payload in iter_preview_payloads(args.accid, args.move_time, waypoints,
                                         grippers):
        print(json.dumps(payload, ensure_ascii=True, indent=2))


def execute_plan(args: argparse.Namespace, label: str,
                 waypoints: list[list[float]], grippers: list[float]):
    import numpy as np

    from fluxvla.engines.operators.tron2_operator import Tron2Operator

    logging.info('Executing motion: %s', label)
    robot = Tron2Operator(build_robot_config(args))
    try:
        joint_state = robot.get_joint_state()
        if joint_state is not None and 'q' in joint_state:
            current = [float(v) for v in joint_state['q'][:ARM_JOINT_DIM]]
            logging.info('Current joints: %s', joints_to_compact_str(current))

        if not args.skip_gripper:
            logging.info('Setting grippers: left=%.1f right=%.1f',
                         grippers[0], grippers[1])
            robot.control_gripper(np.asarray([grippers], dtype=float))
            if args.gripper_wait > 0:
                time.sleep(args.gripper_wait)

        for index, joints in enumerate(waypoints, start=1):
            logging.info('MoveJ waypoint %d/%d: %s', index, len(waypoints),
                         joints_to_compact_str(joints))
            robot.control_joint(
                np.asarray([joints], dtype=float),
                move_time=args.move_time,
                sleep_dt=args.move_time + args.pause)

        if args.tail_wait > 0:
            time.sleep(args.tail_wait)
    finally:
        robot.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ip', default='10.192.1.2')
    parser.add_argument('--accid', default='DACH_TRON2A_061')
    parser.add_argument('--move-time', type=float, default=5.0)
    parser.add_argument(
        '--pause',
        type=float,
        default=0.5,
        help='Extra wait after each MoveJ, in seconds.')
    parser.add_argument(
        '--command-gap',
        type=float,
        default=0.1,
        help='RobotConfig.execution_time for gripper / command pacing.')
    parser.add_argument('--gripper-wait', type=float, default=0.5)
    parser.add_argument('--tail-wait', type=float, default=0.0)
    parser.add_argument('--gripper-speed', type=int, default=50)
    parser.add_argument('--gripper-force', type=int, default=25)
    parser.add_argument(
        '--grippers',
        type=parse_gripper_pair,
        default=DEFAULT_GRIPPER_OPENINGS,
        help='Left/right gripper openings, e.g. "95,95".')
    parser.add_argument(
        '--side-joints',
        type=parse_joint_list,
        default=SIDE_CLEAR_JOINTS,
        help='Intermediate side-clear MoveJ waypoint.')
    parser.add_argument(
        '--home-joints',
        type=parse_joint_list,
        default=HOME_JOINTS,
        help='Home/origin MoveJ waypoint.')
    parser.add_argument('--skip-gripper', action='store_true')
    parser.add_argument('--dry-run', action='store_true')

    subparsers = parser.add_subparsers(dest='command')

    go_init = subparsers.add_parser(
        'go-init', help='Move through side_clear into inference-ready pose')
    go_init.add_argument(
        '--ready-joints',
        type=parse_joint_list,
        default=INFERENCE_READY_JOINTS,
        help='Final inference-ready MoveJ waypoint.')

    subparsers.add_parser(
        'go-home', help='Move from inference-ready pose back to home/origin')

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        raise SystemExit(2)
    return args


def main():
    args = parse_args()
    label, waypoints, grippers = build_plan(args)
    logging.info('Configured ready pose reference: %s',
                 joints_to_compact_str(INFERENCE_READY_JOINTS))
    if args.dry_run:
        preview_plan(args, label, waypoints, grippers)
        return
    execute_plan(args, label, waypoints, grippers)


if __name__ == '__main__':
    main()
