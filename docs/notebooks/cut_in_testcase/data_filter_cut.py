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
from waymo_open_dataset import dataset_pb2 as open_dataset
import tensorflow as tf

def find_leading_vehicles(scenario, av_index, T):
    obj_types = scenario.object_metadata.object_types  # 1=vehicle, 2=pedestrian, 3=cyclist
    other_vehicle_indices = [i for i in range(len(obj_types))
                             if i != av_index and obj_types[i] == 1]

    leading_vehicles = []

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


tfrecord_files = tf.io.gfile.glob(
    "../../data/motion_v_1_3_0/uncompressed/tf_example/training/training_tfexample.tfrecord-*"
)

scenario_file_mapping = []

# ---------------- Main processing loop ----------------
for shard_idx, shard_file in enumerate(tfrecord_files):
    print(f"Processing shard {shard_idx + 1}/{len(tfrecord_files)}: {shard_file.split('/')[-1]}")
    config = dataclasses.replace(_config.WOD_1_3_0_TRAIN_EX, path=shard_file, max_num_objects=32)
    all_counts = sum(1 for _ in tf.data.TFRecordDataset(shard_file))
    print(f"There are {all_counts} scenarios in {shard_file.split('/')[-1]}")

    data_iter = dataloader.simulator_state_generator(config=config)
    output_dir = "../adjacent_filtered_data"
    os.makedirs(output_dir, exist_ok=True)

    for scenario_idx in range(all_counts):
        scenario = next(islice(data_iter, scenario_idx, scenario_idx+1))
        scenario_id = scenario.object_metadata.scenario_id[0].decode('utf-8')
        mp4_file = os.path.join(output_dir, f"{scenario_id}.mp4")
        if os.path.exists(mp4_file):
            print(f"Skipping scenario {scenario_idx}: {scenario_id}, already exists.")
            continue

        # ---------- Step 1: Find AV ----------
        is_sdc_mask = scenario.object_metadata.is_sdc
        av_index = np.where(is_sdc_mask)[0][0]
        obj_types = scenario.object_metadata.object_types
        vehicle_indices = [i for i in range(len(obj_types)) if obj_types[i] == 1 and i != av_index]
        T = min(30, scenario.remaining_timesteps)

        x_av = scenario.log_trajectory.x[av_index, :T]
        y_av = scenario.log_trajectory.y[av_index, :T]
        vx_av = scenario.log_trajectory.vel_x[av_index, :T]
        vy_av = scenario.log_trajectory.vel_y[av_index, :T]
        v_av = np.stack([vx_av, vy_av], axis=-1)
        p_av = np.stack([x_av, y_av], axis=-1)
        v_av_norm = v_av / (np.linalg.norm(v_av, axis=1, keepdims=True) + 1e-6)
        speed_av = np.linalg.norm(v_av, axis=1)

        # ---------- Step 2: Find leading vehicle ----------
        leading_vehicle = None
        leading_vehicles = find_leading_vehicles(scenario, av_index, T)
        if len(leading_vehicles) > 0:
            # 挑最近的
            distances = [np.mean(np.linalg.norm(
                np.stack([scenario.log_trajectory.x[i, :T],
                          scenario.log_trajectory.y[i, :T]], axis=-1) - p_av, axis=1))
                for i in leading_vehicles]
            leading_vehicle = leading_vehicles[np.argmin(distances)]

        if leading_vehicle is not None:
            vx_l = scenario.log_trajectory.vel_x[leading_vehicle, :T]
            vy_l = scenario.log_trajectory.vel_y[leading_vehicle, :T]
            speed_l = np.linalg.norm(np.stack([vx_l, vy_l], axis=-1), axis=1)
        else:
            speed_l = np.full(T, np.inf)  # 没有leader时不限制上界

        # ---------- Step 3: Find adjacent vehicle ----------
        candidate = None
        candidate_dist = np.inf
        candidate_side = None
        for i in vehicle_indices:
            if i == leading_vehicle:
                continue
            p_obj = np.stack([scenario.log_trajectory.x[i, :T],
                              scenario.log_trajectory.y[i, :T]], axis=-1)
            v_obj = np.stack([scenario.log_trajectory.vel_x[i, :T],
                              scenario.log_trajectory.vel_y[i, :T]], axis=-1)
            diff = p_obj - p_av
            proj = np.einsum('ij,ij->i', diff, v_av_norm)

            cross = diff[:, 0] * v_av[:, 1] - diff[:, 1] * v_av[:, 0]
            lateral_dist = cross / (np.linalg.norm(v_av, axis=1) + 1e-6)
            mean_lateral = np.mean(lateral_dist)

            side_mask = np.sign(mean_lateral)
            if abs(mean_lateral) < 1.5 or abs(mean_lateral) > 5.0:
                continue  # <1.5: 同车道, >5: 不是相邻车道

            lat_var = np.var(lateral_dist)
            if lat_var > 0.5:
                continue  # 横向波动太大，可能在转弯，不算相邻车道

            dot = np.einsum('ij,ij->i', v_av, v_obj)
            norm = np.linalg.norm(v_av, axis=1) * np.linalg.norm(v_obj, axis=1)
            cos_angle = np.divide(dot, norm, out=np.zeros_like(dot), where=norm > 0)
            speed_obj = np.linalg.norm(v_obj, axis=1)

            heading_obj = np.arctan2(v_obj[:, 1], v_obj[:, 0])
            heading_var = np.var(np.unwrap(heading_obj))
            if heading_var > 0.05:
                continue

            # dist_to_lead = np.linalg.norm(p_obj - p_av, axis=1)
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

        if candidate is None:
            continue

        # ========== Step 4: Save results ==========
        tensor = np.zeros((3, 4, scenario.remaining_timesteps), dtype=np.float32)
        # AV
        tensor[0, 0, :] = scenario.log_trajectory.x[av_index, :scenario.remaining_timesteps]
        tensor[0, 1, :] = scenario.log_trajectory.y[av_index, :scenario.remaining_timesteps]
        tensor[0, 2, :] = scenario.log_trajectory.vel_x[av_index, :scenario.remaining_timesteps]
        tensor[0, 3, :] = scenario.log_trajectory.vel_y[av_index, :scenario.remaining_timesteps]
        # Adjacent vehicle
        tensor[1, 0, :] = scenario.log_trajectory.x[candidate, :scenario.remaining_timesteps]
        tensor[1, 1, :] = scenario.log_trajectory.y[candidate, :scenario.remaining_timesteps]
        tensor[1, 2, :] = scenario.log_trajectory.vel_x[candidate, :scenario.remaining_timesteps]
        tensor[1, 3, :] = scenario.log_trajectory.vel_y[candidate, :scenario.remaining_timesteps]

        scenario_file_mapping.append(f"{scenario_id}\t{shard_file.split('/')[-1]}")

        # 保存 pkl
        pkl_file = os.path.join(output_dir, f"{scenario_id}.pkl")
        with open(pkl_file, "wb") as f:
            pickle.dump(tensor, f)

        # 生成视频
        imgs = []
        state = scenario
        for t in range(scenario.remaining_timesteps):
            imgs.append(visualization.plot_simulator_state(state, use_log_traj=True))
            state = datatypes.update_state_by_log(state, num_steps=1)

        with imageio.get_writer(mp4_file, fps=10, format='ffmpeg') as writer:
            for frame in imgs:
                writer.append_data(frame)

        print(f"Saved scenario {scenario_idx}: {scenario_id}, AV={av_index}, LV={leading_vehicle}, AVj={candidate}, side={'left' if candidate_side > 0 else 'right'}")

    print(f"All scenarios processed for {shard_file.split('/')[-1]}.")


mapping_file = os.path.join(output_dir, "scenario_file_mapping.txt")
with open(mapping_file, "w") as f:
    f.write("\n".join(scenario_file_mapping))

print(f"Scenario -> TFRecord mapping saved to {mapping_file}")
