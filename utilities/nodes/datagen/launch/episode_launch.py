"""
One data generation episode: sim + ego stack + frozen opponent policy + bag recorder + episode manager.
Everything is configured by the episode.json written by `ros2 run datagen mine`; the launch shuts
down when the episode manager decides the episode is over.
"""
import json
import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, EmitEvent, ExecuteProcess, GroupAction,
                            IncludeLaunchDescription, OpaqueFunction, RegisterEventHandler)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParametersFromFile


def policy_actions(policy: dict, role: str, ep: dict, ep_dir: str) -> list:
    if policy['kind'] == 'python':
        return [Node(
            package='datagen', executable='policy_runner', name='policy_runner', namespace='opp_racecar',
            output='screen',
            parameters=[{'entrypoint': policy['entrypoint'],
                         'params_json': json.dumps(policy['params']),
                         'seed': ep['policy_seed'],
                         'map_dir': ep['map_dir'],
                         'rate_hz': policy['rate_hz']}])]

    launch_file = policy['launch']['file']
    if os.sep not in launch_file:
        launch_file = os.path.join('launch', launch_file)
    include = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(os.path.join(get_package_share_directory(policy['launch']['package']), launch_file)),
        launch_arguments=policy['args'].items())
    actions = []
    if policy['params']:
        # node specific overrides, keyed by fully qualified node name, passed to every node of the policy
        params_file = os.path.join(ep_dir, f'{role}_params.yaml')
        with open(params_file, 'w') as f:
            yaml.safe_dump({node: {'ros__parameters': p} for node, p in policy['params'].items()}, f)
        actions.append(SetParametersFromFile(params_file))
    return [GroupAction(actions + [include], scoped=True)]


def launch_setup(context):
    ep_file = os.path.realpath(LaunchConfiguration('episode').perform(context))
    ep_dir = os.path.dirname(ep_file)
    with open(ep_file) as f:
        ep = json.load(f)

    sim_overrides = os.path.join(ep_dir, 'sim_overrides.yaml')
    with open(sim_overrides, 'w') as f:
        yaml.safe_dump({'bridge': {'ros__parameters': {
            'num_agent': 2,
            'ego_spawn_s': float(ep['spawn']['ego_s_m']),
            'opp_spawn_s': float(ep['spawn']['gap_m']),
            'start_gate': True,
            'kb_teleop': False,
        }}}, f)

    base_system = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(os.path.join(
            get_package_share_directory('stack_master'), 'launch', 'base_system_launch.xml')),
        launch_arguments={
            'map_name': ep['track'],
            'sim': 'True',
            'racecar_version': 'SIM',
            'rviz': LaunchConfiguration('rviz').perform(context),
            'sim_overrides': sim_overrides,
        }.items())

    recorder = ExecuteProcess(
        cmd=['ros2', 'bag', 'record', '-o', os.path.join(ep_dir, 'bag')] + ep['record_topics'],
        output='log')

    manager = Node(
        package='datagen', executable='episode_manager', name='episode_manager', output='screen',
        parameters=[{'episode_file': ep_file, 'result_file': os.path.join(ep_dir, 'result.json')}])

    return [
        recorder,
        base_system,
        *policy_actions(ep['ego_policy'], 'ego', ep, ep_dir),
        *policy_actions(ep['opp_policy'], 'opp', ep, ep_dir),
        manager,
        RegisterEventHandler(OnProcessExit(
            target_action=manager,
            on_exit=[EmitEvent(event=Shutdown(reason='episode over'))])),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('episode', description='Path to the episode.json written by `ros2 run datagen mine`'),
        DeclareLaunchArgument('rviz', default_value='False', description='True to visualize the episode in RViz'),
        OpaqueFunction(function=launch_setup),
    ])
