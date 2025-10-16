import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import dataclasses

import jax
from jax import numpy as jnp
import mediapy
import numpy as np
from tqdm import tqdm
from waymax import agents
from waymax import config as _config
from waymax import dataloader
from waymax import datatypes
from waymax import dynamics
from waymax import env as _env
from waymax import visualization
import imageio
import pickle
import tensorflow as tf
from itertools import islice
import pandas as pd
import re


def find_leading_vehicles(scenario):
    obj_types = scenario.object_metadata.object_types  # 1=vehicle, 2=pedestrian, 3=cyclist
    other_vehicle_indices = [i for i in range(len(obj_types))
                             if i != av_index and obj_types[i] == 1]

    leading_vehicles = []
    T = min(50, scenario.remaining_timesteps)

    x_av = scenario.log_trajectory.x[av_index, :T]
    y_av = scenario.log_trajectory.y[av_index, :T]
    vx_av = scenario.log_trajectory.vel_x[av_index, :T]
    vy_av = scenario.log_trajectory.vel_y[av_index, :T]
    v_av = np.stack([vx_av, vy_av], axis=-1)
    p_av = np.stack([x_av, y_av], axis=-1)
    v_av_norm = v_av / (np.linalg.norm(v_av, axis=1, keepdims=True) + 1e-6)

    for i in other_vehicle_indices:
        # 对象轨迹
        x_obj = scenario.log_trajectory.x[i, :T]
        y_obj = scenario.log_trajectory.y[i, :T]
        vx_obj = scenario.log_trajectory.vel_x[i, :T]
        vy_obj = scenario.log_trajectory.vel_y[i, :T]

        v_obj = np.stack([vx_obj, vy_obj], axis=-1)
        p_obj = np.stack([x_obj, y_obj], axis=-1)

        # 方向角
        dot = np.einsum('ij,ij->i', v_av, v_obj)
        norm = np.linalg.norm(v_av, axis=1) * np.linalg.norm(v_obj, axis=1)
        cos_angle = np.divide(dot, norm, out=np.zeros_like(dot), where=norm > 0)

        # 横向距离
        diff = p_obj - p_av
        cross = np.abs(diff[:, 0] * v_av[:, 1] - diff[:, 1] * v_av[:, 0])
        lateral_dist = cross / (np.linalg.norm(v_av, axis=1) + 1e-6)

        # 投影和纵向距离
        proj = np.einsum('ij,ij->i', diff, v_av_norm)
        longitudinal_dist = np.abs(proj)

        # 初步条件
        valid_mask = (cos_angle > 0.9) & (lateral_dist < 0.5) & (proj > 0)

        # 检查纵向距离递减连续段
        consecutive_count = 0
        max_consecutive = 0
        last_dist = None
        for t in range(T):
            if valid_mask[t] and (last_dist is None or longitudinal_dist[t] <= last_dist):
                consecutive_count += 1
                last_dist = longitudinal_dist[t]
            else:
                max_consecutive = max(max_consecutive, consecutive_count)
                consecutive_count = 0
                last_dist = None
        max_consecutive = max(max_consecutive, consecutive_count)

        # 如果连续段长度达到50%T，则加入
        if max_consecutive >= 0.5 * T:
            leading_vehicles.append(i)
    return leading_vehicles


def find_adjacent_vehicle(scenario, av_index, leading_vehicle=None):
    obj_types = scenario.object_metadata.object_types
    vehicle_indices = [i for i in range(len(obj_types)) if obj_types[i] == 1 and i != av_index]
    T = min(50, scenario.remaining_timesteps)

    # AV 轨迹
    x_av = scenario.log_trajectory.x[av_index, :T]
    y_av = scenario.log_trajectory.y[av_index, :T]
    vx_av = scenario.log_trajectory.vel_x[av_index, :T]
    vy_av = scenario.log_trajectory.vel_y[av_index, :T]
    v_av = np.stack([vx_av, vy_av], axis=-1)
    p_av = np.stack([x_av, y_av], axis=-1)
    v_av_norm = v_av / (np.linalg.norm(v_av, axis=1, keepdims=True) + 1e-6)
    speed_av = np.linalg.norm(v_av, axis=1)
    length_av = scenario.log_trajectory.length[av_index][0]

    if leading_vehicle is not None:
        vx_l = scenario.log_trajectory.vel_x[leading_vehicle, :T]
        vy_l = scenario.log_trajectory.vel_y[leading_vehicle, :T]
        speed_l = np.linalg.norm(np.stack([vx_l, vy_l], axis=-1), axis=1)
    else:
        speed_l = np.full(T, np.inf)

    candidate = None
    candidate_dist = np.inf
    candidate_side = None

    for i in vehicle_indices:
        if i == leading_vehicle:
            continue

        # 目标车辆轨迹
        p_obj = np.stack([scenario.log_trajectory.x[i, :T],
                          scenario.log_trajectory.y[i, :T]], axis=-1)
        v_obj = np.stack([scenario.log_trajectory.vel_x[i, :T],
                          scenario.log_trajectory.vel_y[i, :T]], axis=-1)

        diff = p_obj - p_av
        proj = np.einsum('ij,ij->i', diff, v_av_norm)

        # 横向偏移
        cross = diff[:, 0] * v_av[:, 1] - diff[:, 1] * v_av[:, 0]
        lateral_dist = cross / (np.linalg.norm(v_av, axis=1) + 1e-6)
        mean_lateral = np.mean(lateral_dist)

        # 判断左右车道并限制为相邻车道
        if abs(mean_lateral) < 1.5 or abs(mean_lateral) > 5.0:
            continue
        side_mask = np.sign(mean_lateral)
        if np.var(lateral_dist) > 0.5:
            continue  # 横向波动太大

        # 同向 & 速度限制
        dot = np.einsum('ij,ij->i', v_av, v_obj)
        norm = np.linalg.norm(v_av, axis=1) * np.linalg.norm(v_obj, axis=1)
        cos_angle = np.divide(dot, norm, out=np.zeros_like(dot), where=norm > 0)
        speed_obj = np.linalg.norm(v_obj, axis=1)
        heading_obj = np.arctan2(v_obj[:, 1], v_obj[:, 0])
        heading_var = np.var(np.unwrap(heading_obj))
        if heading_var > 0.05:
            continue  # 太弯的不算

        valid_mask = (
                (proj > 0) &
                (cos_angle > 0.98) &
                (speed_obj >= speed_av) &
                (speed_obj <= speed_l) &
                (proj >= 5) & (proj <= 10)
        )

        if valid_mask.mean() > 0.3:
            mean_dist = np.min(proj[valid_mask])
            # print(i, mean_dist)
            if mean_dist < candidate_dist:
                candidate = i
                candidate_dist = mean_dist
                candidate_side = side_mask
                candidate_init_time = np.argmax(valid_mask == True)

    return candidate, candidate_side, candidate_init_time


def strip_scenario_id(state):
    """Return a copy of state where scenario_id is replaced by a JAX-friendly placeholder.

    We replace scenario_id with a small scalar jnp.int32 value so the whole pytree
    contains only JAX-compatible leaves when passed into jax.jit functions.
    """
    # Use a scalar JAX integer as placeholder (shape ()). This is safe for jit.
    placeholder = jnp.array(0, dtype=jnp.int32)

    # Note: dataclasses.replace will keep other fields intact.
    clean_metadata = dataclasses.replace(
        state.object_metadata,
        scenario_id=placeholder
    )
    return dataclasses.replace(state, object_metadata=clean_metadata)

output_dir = 'docs/cutin_filtered_data'
# Config dataset:
max_num_objects = 32

tfrecord_files = tf.io.gfile.glob(
    "data/motion_v_1_3_0/uncompressed/tf_example/training/training_tfexample.tfrecord-*"
)
tfrecord_files = sorted(
    tfrecord_files,
    key=lambda x: int(re.search(r'tfrecord-(\d+)-of', x).group(1))
)
filtered_scenarios = [f[:-4] for f in os.listdir("docs/cutin_filtered_data") if f.endswith('.pkl')]

processed_scenarios = set()
pending_scenarios = set(filtered_scenarios)

for shard_idx, shard_file in enumerate(tfrecord_files):

    config = dataclasses.replace(
        _config.WOD_1_3_0_TRAIN_EX,
        path=shard_file,
        max_num_objects=max_num_objects
    )

    all_counts = sum(1 for _ in tf.data.TFRecordDataset(shard_file))
    jax.debug.print("There are {} scenarios in {}", all_counts, shard_file.split('/')[-1])

    data_iter = dataloader.simulator_state_generator(config=config)

    for scenario_idx in range(all_counts):
        scenario = next(islice(data_iter, scenario_idx, scenario_idx + 1))
        scenario_id = scenario.object_metadata.scenario_id[0].decode('utf-8')
        if (scenario_id not in filtered_scenarios) or (scenario_id in processed_scenarios):
            continue

        # jax.debug.print("Scenario id: {}", scenario_id)

        jax.debug.print("Processing scenario: {}", scenario_id)
        is_sdc_mask = scenario.object_metadata.is_sdc
        av_index = np.where(is_sdc_mask)[0][0]

        leading_vehicles_results = find_adjacent_vehicle(scenario, av_index)
        leading_index = leading_vehicles_results[0]
        lead_side = leading_vehicles_results[1]
        init_lanechange = leading_vehicles_results[2]
        # Config the multi-agent environment:
        init_steps = 11

        dynamics_model = dynamics.StateDynamics()

        # Expect users to control all valid object in the scene.
        env = _env.MultiAgentEnvironment(
            dynamics_model=dynamics_model,
            config=dataclasses.replace(
                _config.EnvironmentConfig(),
                max_num_objects=max_num_objects,
                controlled_object=_config.ObjectType.VALID,
            ),
        )

        obj_idx = jnp.arange(max_num_objects)

        actor_0 = agents.create_lane_change_actor(
            dynamics_model=dynamics_model,
            is_controlled_func=lambda state: obj_idx == leading_index,
            side=lead_side,
            duration_s=3.81,
            init_step=init_lanechange,
        )

        # controls all the other vehicles.
        actor_1 = agents.create_expert_actor(
            dynamics_model=dynamics_model,
            is_controlled_func=lambda state: (obj_idx != leading_index),
        )

        actors = [actor_0, actor_1]  # include all the vehicles you want to change
        jit_step = jax.jit(env.step)
        jit_select_action_list = [jax.jit(actor.select_action) for actor in actors]
        states = [env.reset(scenario)]

        T = max(91, states[0].remaining_timesteps)

        t = T - states[0].remaining_timesteps

        rng = jax.random.PRNGKey(0)

        actor_states = [actor.init(rng, None) for actor in actors]

        trajectories = [states[0]]

        for _ in range(t, T):
            current_state = states[-1]

            clean_state = strip_scenario_id(current_state)
            outputs = []
            new_actor_states = []
            for i, jit_select_action in enumerate(jit_select_action_list):
                out = jit_select_action({}, clean_state, actor_states[i], None)
                outputs.append(out)
                new_actor_states.append(out.actor_state)

            actor_states = new_actor_states
            action = agents.merge_actions(outputs)
            next_state = jit_step(clean_state, action)
            trajectories.append(next_state)

            if next_state.timestep < 65:
                states.append(next_state)
            
        # ========== Step 4: Save results ==========
        tensor = np.zeros((32, 4, scenario.remaining_timesteps), dtype=np.float32)

        tensor[:, 0, :] = scenario.current_sim_trajectory.x[:, :scenario.remaining_timesteps]
        tensor[:, 1, :] = scenario.current_sim_trajectory.y[:, :scenario.remaining_timesteps]
        tensor[:, 2, :] = scenario.current_sim_trajectory.vel_x[:, :scenario.remaining_timesteps]
        tensor[:, 3, :] = scenario.current_sim_trajectory.vel_y[:, :scenario.remaining_timesteps]
       
        # 保存 pkl
        pkl_file = os.path.join(output_dir, f"{scenario_id}_modified.pkl")
        with open(pkl_file, "wb") as f:
            pickle.dump(tensor, f)

        # 生成视频
        #imgs = []
        #state = scenario
        #for t in range(scenario.remaining_timesteps):
        #    imgs.append(visualization.plot_simulator_state(state, use_log_traj=True))
        #    state = datatypes.update_state_by_log(state, num_steps=1)
        #mp4_file = os.path.join(output_dir, f"{scenario_id}_modified.mp4")
        #with imageio.get_writer(mp4_file, fps=10, format='ffmpeg') as writer:
        #    for frame in imgs:
        #        writer.append_data(frame)

        print(f"Saved scenario {scenario_idx}: {scenario_id}, AV={av_index}, LV={leading_index} ")

    print(f"All scenarios processed for {shard_file.split('/')[-1]}.")
