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


def find_leading_vehicles(scenario):
    obj_types = scenario.object_metadata.object_types  # 1=vehicle, 2=pedestrian, 3=cyclist
    other_vehicle_indices = [i for i in range(len(obj_types))
                             if i != av_index and obj_types[i] == 1]

    leading_vehicles = []
    T = min(30, scenario.remaining_timesteps)

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
        valid_mask = (cos_angle > 0.9) & (lateral_dist < 0.5) & (proj > 0) & (longitudinal_dist >= 25)

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
    T = min(30, scenario.remaining_timesteps)

    # AV 轨迹
    x_av = scenario.log_trajectory.x[av_index, :T]
    y_av = scenario.log_trajectory.y[av_index, :T]
    vx_av = scenario.log_trajectory.vel_x[av_index, :T]
    vy_av = scenario.log_trajectory.vel_y[av_index, :T]
    v_av = np.stack([vx_av, vy_av], axis=-1)
    p_av = np.stack([x_av, y_av], axis=-1)
    v_av_norm = v_av / (np.linalg.norm(v_av, axis=1, keepdims=True) + 1e-6)
    speed_av = np.linalg.norm(v_av, axis=1)

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
                (proj >= 10) & (proj < 25)
        )

        if valid_mask.mean() > 0.3:
            mean_dist = np.mean(proj[valid_mask])
            if mean_dist < candidate_dist:
                candidate = i
                candidate_dist = mean_dist
                candidate_side = side_mask

    return candidate, candidate_side


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


# Config dataset:
max_num_objects = 32

config = dataclasses.replace(
    _config.WOD_1_3_0_TRAIN_EX, max_num_objects=max_num_objects
)
data_iter = dataloader.simulator_state_generator(config=config)

for scenario_idx, scenario in enumerate(data_iter):
    scenario_id = scenario.object_metadata.scenario_id[0].decode('utf-8')
    #if scenario_id == '1fb44c31801c956d':
    #if scenario_id == '5130b590379b4722':
    #if scenario_id == '1ba54b48c61a6e1b':
    if scenario_id == '10f5ea040a3e4acd':
        is_sdc_mask = scenario.object_metadata.is_sdc
        av_index = np.where(is_sdc_mask)[0][0]
        leading_vehicles_results = find_adjacent_vehicle(scenario, av_index)
        leading_index = leading_vehicles_results[0]
        jax.debug.print('leading_index:{}', leading_index)
        #distances = []
        #for i in leading_vehicles:
        #    dx = scenario.log_trajectory.x[i, :91] - scenario.log_trajectory.x[av_index, :91]
        #    dy = scenario.log_trajectory.y[i, :91] - scenario.log_trajectory.y[av_index, :91]
        #    dist = np.mean(np.sqrt(dx ** 2 + dy ** 2))
        #    distances.append(dist)
        #leading_index = leading_vehicles[np.argmin(distances)]
        break

# Config the multi-agent environment:
#init_steps = leading_vehicles_results[np.argmin(distances), 1]
#jax.debug.print("initial timestep={} for vehicle id: {}", init_steps, leading_index)
#side = leading_vehicles_results[np.argmin(distances), 2]
lead_side = leading_vehicles_results[1]
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

# An actor that doesn't move, controlling all objects with index > 4
obj_idx = jnp.arange(max_num_objects)

# leading vehicle control
actor_0 = agents.create_lane_change_actor(
    dynamics_model=dynamics_model,
    is_controlled_func=lambda state: obj_idx == leading_index,
    side = lead_side
)

# IDM actor/policy controlling both object 0 and 1.
# Note IDM policy is an actor hard-coded to use dynamics.StateDynamics().
#actor_1 = agents.IDMRoutePolicy(
    #is_controlled_func=lambda state: (obj_idx == 0) | (obj_idx == 1)
#    is_controlled_func=lambda state: obj_idx == av_index
#)

actor_1 = agents.MPC_2D_actor(
    dynamics_model=dynamics_model,
    is_controlled_func=lambda state: obj_idx == av_index,
    av_idx=av_index,
    obs_idx=leading_index,
)

#actor_1 = agents.MPC_actor(
#    dynamics_model=dynamics_model,
#    is_controlled_func=lambda state: obj_idx == av_index,
#    av_idx=av_index,
#    lead_idx=leading_index,
#)

actors = [actor_0, actor_1]  # include all the vehicles you want to change
jit_step = jax.jit(env.step)
jit_select_action_list = [jax.jit(actor.select_action) for actor in actors]
states = [env.reset(scenario)]

T = max(91, states[0].remaining_timesteps)
# tensor = np.zeros((2, 4, T), dtype=np.float32)

t = T - states[0].remaining_timesteps

rng = jax.random.PRNGKey(0)

actor_states = [actor.init(rng, None) for actor in actors]

for _ in range(t, T):
    current_state = states[-1]

    clean_state = strip_scenario_id(current_state)
    outputs = []
    new_actor_states = []

    for i, jit_select_action in enumerate(jit_select_action_list):
        out = jit_select_action({}, clean_state, actor_states[i], None)
        #out = jit_select_action({}, clean_state, actor_states[i], None)
        outputs.append(out)
        new_actor_states.append(out.actor_state)

    actor_states = new_actor_states
    action = agents.merge_actions(outputs)
    next_state = jit_step(clean_state, action)

    if next_state.timestep < 55:

        states.append(next_state)

        imgs = []
        for state in states:
            imgs.append(visualization.plot_simulator_state(state, use_log_traj=False))
        with imageio.get_writer(f'docs/processed_data/{scenario_id}_MPC2d_lanechange.mp4', fps=10) as writer:
            for frame in imgs:
                writer.append_data(frame)
    else:
        break
    

# with open(f"../processed_data/{scenario_id}_m.pkl", "wb") as f:
#     pickle.dump(tensor, f)
# print(f"Saved successfully: {scenario_id}_m.pkl, shape = {tensor.shape}")


