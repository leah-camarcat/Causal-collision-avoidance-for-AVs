import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..", "..")))
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

# def generate_random_accel_heading_sequence(
#     timesteps: int = 80,
#     accel_range: tuple = (-2.0, 2.0),
#     heading_range: tuple = (-jnp.pi, jnp.pi),
#     smooth_factor: float = 0.8,
#     seed: int = 42
# ) -> jnp.ndarray:
#
#     rng = np.random.default_rng(seed)
#     accel = rng.uniform(low=accel_range[0], high=accel_range[1], size=timesteps)
#     heading = rng.uniform(low=heading_range[0], high=heading_range[1], size=timesteps)
#
#
#     def smooth(seq, alpha):
#         smoothed = np.zeros_like(seq)
#         smoothed[0] = seq[0]
#         for t in range(1, len(seq)):
#             smoothed[t] = alpha * smoothed[t - 1] + (1 - alpha) * seq[t]
#         return smoothed
#
#     accel_smooth = smooth(accel, smooth_factor)
#     heading_smooth = smooth(heading, smooth_factor)
#
#     return jnp.stack([accel_smooth, heading_smooth], axis=0)

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

    jax.debug.print('proj:{}', proj)
    # Collision threshold
    collision_mask = (proj < (trajectory.length[av_index][0]/2 + trajectory.length[lead_idx][0]/2)) & (proj != 0)

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

# Config dataset:
max_num_objects = 32

config = dataclasses.replace(
    _config.WOD_1_3_0_TRAIN_EX, max_num_objects=max_num_objects
)
data_iter = dataloader.simulator_state_generator(config=config)

for scenario_idx, scenario in enumerate(data_iter):
    scenario_id = scenario.object_metadata.scenario_id[0].decode('utf-8')
    #if scenario_id == '1fb44c31801c956d':
    #if scenario_id == 'cd7a6be74ecc125a':
    if scenario_id == 'dddbf8db0b149f17':
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
        break

# Config the multi-agent environment:
init_steps = 11

# Set the dynamics model the environment is using.
# Note each actor interacting with the environment needs to provide action
# compatible with this dynamics model.
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
# Setup a few actors, see visualization below for how each actor behaves.

# An actor that doesn't move, controlling all objects with index > 4
obj_idx = jnp.arange(max_num_objects)

# static_actor = agents.create_constant_speed_actor(
#     speed=0.0,
#     dynamics_model=dynamics_model,
#     is_controlled_func=lambda state: obj_idx > 3,
# )

# control_sequence = generate_random_accel_heading_sequence()

# actor_0 = agents.create_acceleration_heading_actor(
#     dynamics_model=dynamics_model,
#     is_controlled_func=lambda state: obj_idx == 2,
#     control_array=control_sequence,
# )

# leading vehicle control
actor_0 = agents.create_constant_acceleration_actor(
    dynamics_model=dynamics_model,
    is_controlled_func=lambda state: obj_idx == leading_index,
    acceleration=-5.884
)

# IDM actor/policy controlling both object 0 and 1.
# Note IDM policy is an actor hard-coded to use dynamics.StateDynamics().
actor_1 = agents.IDMRoutePolicy(
    is_controlled_func=lambda state: obj_idx == av_index
)

#actor_1 = agents.MPC_actor(
#    dynamics_model=dynamics_model,
#    is_controlled_func=lambda state: obj_idx == av_index,
#    av_idx=av_index,
#    lead_idx=leading_index,
#)

#actor_1 = agents.davis_actor(
#    dynamics_model=dynamics_model,
#    is_controlled_func=lambda state: obj_idx == av_index,
#    av_idx=av_index,
#    lead_idx=leading_index,
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
# tensor = np.zeros((2, 4, T), dtype=np.float32)

t = T - states[0].remaining_timesteps
# tensor[0, 0, :t] = scenario.log_trajectory.x[av_index, :t]
# tensor[0, 1, :t] = scenario.log_trajectory.y[av_index, :t]
# tensor[0, 2, :t] = scenario.log_trajectory.vel_x[av_index, :t]
# tensor[0, 3, :t] = scenario.log_trajectory.vel_y[av_index, :t]
#
# tensor[1, 0, :t] = scenario.log_trajectory.x[leading_index, :t]
# tensor[1, 1, :t] = scenario.log_trajectory.y[leading_index, :t]
# tensor[1, 2, :t] = scenario.log_trajectory.vel_x[leading_index, :t]
# tensor[1, 3, :t] = scenario.log_trajectory.vel_y[leading_index, :t]


# create a single random key
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

    # tensor[0, 0, _] = next_state.sim_trajectory.x[av_index, _]
    # tensor[0, 1, _] = next_state.sim_trajectory.y[av_index, _]
    # tensor[0, 2, _] = next_state.sim_trajectory.vel_x[av_index, _]
    # tensor[0, 3, _] = next_state.sim_trajectory.vel_y[av_index, _]
    #
    # tensor[1, 0, _] = next_state.sim_trajectory.x[leading_index, _]
    # tensor[1, 1, _] = next_state.sim_trajectory.y[leading_index, _]
    # tensor[1, 2, _] = next_state.sim_trajectory.vel_x[leading_index, _]
    # tensor[1, 3, _] = next_state.sim_trajectory.vel_y[leading_index, _]
    imgs = []
    for state in states:
        imgs.append(visualization.plot_simulator_state(state, use_log_traj=False))
    with imageio.get_writer(f'docs/processed_data/{scenario_id}_IDM_Rtrigg1_5.mp4', fps=10) as writer:
        for frame in imgs:
            writer.append_data(frame)

collision, t_col, v_av, v_lead, delta_v = detect_collision(
                states[-1].sim_trajectory,
                av_idx=av_index,
                lead_idx=leading_index,
            )

print(f'collision: {collision}')

# with open(f"../processed_data/{scenario_id}_m.pkl", "wb") as f:
#     pickle.dump(tensor, f)
# print(f"Saved successfully: {scenario_id}_m.pkl, shape = {tensor.shape}")


