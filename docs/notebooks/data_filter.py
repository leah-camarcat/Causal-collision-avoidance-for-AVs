import os
import imageio
from itertools import islice
import dataclasses
from waymax import config as _config
from waymax import dataloader
from waymax import datatypes
from waymax import visualization
import numpy as np
import pickle

config = dataclasses.replace(_config.WOD_1_3_0_TRAIN_EX, max_num_objects=32)

data_iter = dataloader.simulator_state_generator(config=config)

# Loop all the scenarios
for scenario_idx, scenario in enumerate(data_iter):
    scenario_id = scenario.object_metadata.scenario_id[0].decode('utf-8')
    # Find AV
    is_sdc_mask = scenario.object_metadata.is_sdc
    av_index = np.where(is_sdc_mask)[0][0]  # AV's index

    # Filter out vehicles
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

    if len(leading_vehicles) == 0:
        continue

    # 计算每个leading vehicle与AV在整个时间段的平均距离
    distances = []
    for i in leading_vehicles:
        dx = scenario.log_trajectory.x[i, :T] - scenario.log_trajectory.x[av_index, :T]
        dy = scenario.log_trajectory.y[i, :T] - scenario.log_trajectory.y[av_index, :T]
        dist = np.mean(np.sqrt(dx ** 2 + dy ** 2))
        distances.append(dist)
    closest_idx = leading_vehicles[np.argmin(distances)]

    # 构建 (2, 4, T) 的tensor
    tensor = np.zeros((2, 4, scenario.remaining_timesteps), dtype=np.float32)
    # AV first
    tensor[0, 0, :] = scenario.log_trajectory.x[av_index, :scenario.remaining_timesteps]
    tensor[0, 1, :] = scenario.log_trajectory.y[av_index, :scenario.remaining_timesteps]
    tensor[0, 2, :] = scenario.log_trajectory.vel_x[av_index, :scenario.remaining_timesteps]
    tensor[0, 3, :] = scenario.log_trajectory.vel_y[av_index, :scenario.remaining_timesteps]
    # leading vehicle second
    tensor[1, 0, :] = scenario.log_trajectory.x[closest_idx, :scenario.remaining_timesteps]
    tensor[1, 1, :] = scenario.log_trajectory.y[closest_idx, :scenario.remaining_timesteps]
    tensor[1, 2, :] = scenario.log_trajectory.vel_x[closest_idx, :scenario.remaining_timesteps]
    tensor[1, 3, :] = scenario.log_trajectory.vel_y[closest_idx, :scenario.remaining_timesteps]

    # 保存 pkl
    pkl_file = f"../filtered_data/{scenario_id}.pkl"
    with open(pkl_file, "wb") as f:
        pickle.dump(tensor, f)
    # video generation
    imgs = []
    state = scenario
    for t in range(scenario.remaining_timesteps):
        imgs.append(visualization.plot_simulator_state(state, use_log_traj=True))
        state = datatypes.update_state_by_log(state, num_steps=1)

    output_file = f"../filtered_data/{scenario_id}.mp4"
    with imageio.get_writer(output_file, fps=10, format='ffmpeg') as writer:
        for frame in imgs:
            writer.append_data(frame)
    print(f"Saved video for scenario {scenario_idx}, leading vehicles: {leading_vehicles}")