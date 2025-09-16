import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
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
        valid_mask = (cos_angle > 0.9) & (lateral_dist < 0.5) & (proj > 0) & (longitudinal_dist <= 25)

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

def detect_collision(trajectory, av_idx, lead_idx):
    """
    Returns:
        collision_happened (bool),
        first_collision_timestep (int or None),
        speed_av_at_collision (float or None),
        speed_lead_at_collision (float or None),
        delta_v (float or None)
    """
    # Extract positions across timesteps
    x_av, y_av = trajectory.x[av_idx], trajectory.y[av_idx]
    x_lead, y_lead = trajectory.x[lead_idx], trajectory.y[lead_idx]
    vx_av = trajectory.vel_x[av_index]
    vy_av = trajectory.vel_y[av_index]
    v_av = np.stack([vx_av, vy_av], axis=-1)
    # Euclidean distance
    # dist = jnp.sqrt((x_av - x_lead)**2 + (y_av - y_lead)**2)
    p_av = np.stack([x_av, y_av], axis=-1)
    p_obj = np.stack([x_lead, y_lead], axis=-1)

    v_av_norm = v_av / (np.linalg.norm(v_av, axis=1, keepdims=True) + 1e-6)
    diff = p_obj - p_av
    proj = np.einsum('ij,ij->i', diff, v_av_norm)

    jax.debug.print('proj:{}', proj[:45].min())
    # Collision threshold
    collision_mask = proj < 0

    collision_happened = bool(jnp.any(collision_mask))
    first_collision_timestep = None
    speed_av_at_collision = None
    speed_lead_at_collision = None
    delta_v = None

    if collision_happened:
        # timestep of first collision
        first_collision_timestep = int(jnp.argmax(collision_mask))

        # Speeds at collision timestep
        vx_av = trajectory.vel_x[av_idx, first_collision_timestep]
        vy_av = trajectory.vel_y[av_idx, first_collision_timestep]
        vx_lead = trajectory.vel_x[lead_idx, first_collision_timestep]
        vy_lead = trajectory.vel_y[lead_idx, first_collision_timestep]

        speed_av_at_collision = float(jnp.sqrt(vx_av**2 + vy_av**2))
        speed_lead_at_collision = float(jnp.sqrt(vx_lead**2 + vy_lead**2))
        delta_v = abs(speed_av_at_collision - speed_lead_at_collision)

    return (
        collision_happened,
        first_collision_timestep,
        speed_av_at_collision,
        speed_lead_at_collision,
        delta_v,
    )


def compute_jerk(log_trajectory, veh_idx, dt=0.1):
    """
    Compute jerk magnitude (m/s^3) over time for a given vehicle.
    
    Args:
        log_trajectory: scenario.log_trajectory
        veh_idx: index of the vehicle (AV or leader)
        dt: timestep size in seconds (default 0.1s, adjust to your sim)
    
    Returns:
        jerk: array of jerk magnitudes at each timestep (len = T-2)
        max_jerk: maximum jerk magnitude
        mean_jerk: mean jerk magnitude
    """
    vx = log_trajectory.vel_x[veh_idx]
    vy = log_trajectory.vel_y[veh_idx]

    # Acceleration components
    ax = jnp.gradient(vx, dt)
    ay = jnp.gradient(vy, dt)

    # Jerk components
    jx = jnp.gradient(ax, dt)
    jy = jnp.gradient(ay, dt)

    # Magnitude of jerk
    jerk = jnp.sqrt(jx**2 + jy**2)

    return jerk, float(jnp.max(jerk)), float(jnp.mean(jerk))


# Config dataset:
max_num_objects = 32

tfrecord_files = tf.io.gfile.glob("data/motion_v_1_3_0/uncompressed/tf_example/training/training_tfexample.tfrecord-*")
filtered_scenarios = [f[:-4] for f in os.listdir("docs/filtered_data") if f.endswith('.pkl')]

KPIS = pd.DataFrame(index=filtered_scenarios, columns=["col_outcome", "Vs_av", "Vs_lead", "delta_Vs", "jerk_av", "max_jerk_av", "mean_jerk_av"])

for shard_idx, shard_file in enumerate(tfrecord_files):
    
    config = dataclasses.replace(
        _config.WOD_1_3_0_TRAIN_EX,
        path=shard_file,  
        max_num_objects=max_num_objects
    )
    
    all_counts = 0
    for _ in tf.data.TFRecordDataset(shard_file):
        all_counts += 1
    jax.debug.print("There are {} scenarios in {}", all_counts, shard_file.split('/')[-1])

    data_iter = dataloader.simulator_state_generator(config=config)

    output_dir = "../processed_data"
    os.makedirs(output_dir, exist_ok=True)

    for scenario_idx in range(all_counts):
        scenario = next(islice(data_iter, scenario_idx, scenario_idx+1))
        scenario_id = scenario.object_metadata.scenario_id[0].decode('utf-8')
        #jax.debug.print("Scenario id: {}", scenario_id)
        if (scenario_id in filtered_scenarios) and (KPIS.loc[scenario_id].isna().all()):
            jax.debug.print("Scenario id found: {}", scenario_id)
            is_sdc_mask = scenario.object_metadata.is_sdc
            av_index = np.where(is_sdc_mask)[0][0]
            leading_vehicles = find_leading_vehicles(scenario)
            distances = []
            for i in leading_vehicles:
                dx = scenario.log_trajectory.x[i, :91] - scenario.log_trajectory.x[av_index, :91]
                dy = scenario.log_trajectory.y[i, :91] - scenario.log_trajectory.y[av_index, :91]
                dist = np.mean(np.sqrt(dx ** 2 + dy ** 2))
                distances.append(dist)
            leading_index = leading_vehicles[np.argmin(distances)]

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

            actor_0 = agents.create_constant_acceleration_actor(
                dynamics_model=dynamics_model,
                is_controlled_func=lambda state: obj_idx == leading_index,
                acceleration=-5.884
            )

            actor_1 = agents.davis_actor(
                dynamics_model=dynamics_model,
                is_controlled_func=lambda state: obj_idx == av_index,
                av_idx=av_index,
                lead_idx=leading_index,
            )
            
            #actor_1 = agents.IDMRoutePolicy(
            #    is_controlled_func=lambda state: obj_idx == av_index,
            #)

            # controls all the other vehicles.
            actor_2 = agents.create_expert_actor(
                dynamics_model=dynamics_model,
                is_controlled_func=lambda state: (obj_idx != leading_index) & (obj_idx != av_index),
            )

            actors = [actor_0, actor_1, actor_2]  # include all the vehicles you want to change
            jit_step = jax.jit(env.step)
            jit_select_action_list = [jax.jit(actor.select_action) for actor in actors]
            states = [env.reset(scenario)]

            T = max(91, states[0].remaining_timesteps)

            t = T - states[0].remaining_timesteps

            
            for _ in range(t, T):
                current_state = states[-1]

                clean_state = strip_scenario_id(current_state)

                outputs = [
                    jit_select_action({}, clean_state, None, None)
                    for jit_select_action in jit_select_action_list
                ]

                action = agents.merge_actions(outputs)
                next_state = jit_step(clean_state, action)
                if next_state.timestep < 45:
                    states.append(next_state)
                
                    #imgs = []
                    #for state in states:
                    #    imgs.append(visualization.plot_simulator_state(state, use_log_traj=False))
                    #with imageio.get_writer(f'docs/processed_data/{scenario_id}_IDM_longtest.mp4', fps=10) as writer:
                    #    for frame in imgs:
                    #        writer.append_data(frame)
                
            # calculate KPIs and store them !
            # check whether collision occured
            collision, t_col, v_av, v_lead, delta_v = detect_collision(
                states[-1].sim_trajectory,
                av_idx=av_index,
                lead_idx=leading_index,
            )
            if collision:
                col_outcome = 0
                Vs_av, Vs_lead, delta_Vs = [v_av, v_lead, delta_v]
            else:  
                col_outcome = 1
                Vs_av, Vs_lead, delta_Vs = [jnp.nan, jnp.nan, jnp.nan]
            
            jerk_av, max_jerk_av, mean_jerk_av = compute_jerk(states[-1].log_trajectory, av_index)

            KPIS.loc[scenario_id] = pd.Series({'col_outcome':col_outcome, 
                                           'Vs_av': Vs_av, 
                                           'Vs_lead': Vs_lead, 
                                           'delta_Vs': delta_Vs, 
                                           'jerk_av': jerk_av, 
                                           'max_jerk_av': max_jerk_av, 
                                           'mean_jerk_av':mean_jerk_av})  

        else: 
            continue

KPIS.to_csv('docs/notebooks/testing_davis_breakdown_dry.csv')

