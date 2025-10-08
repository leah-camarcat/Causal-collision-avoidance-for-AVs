"""
Improved ground truth risk grid using Modified Time-To-Collision (MTTC).
Replaces simplistic Gaussian zones with physically-motivated risk metrics.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..",  "..", "..")))
import jax
import jax.numpy as jnp
from waymax import datatypes
import numpy as np 

# ============================================================================
# MTTC-BASED GROUND TRUTH RISK GRID
# ============================================================================


def compute_mttc_vectorized(
    state: datatypes.SimulatorState,
    ego_idx: int,
    max_timesteps: int = 91,
    critical_ttc: float = 1,
    safe_ttc: float = 5.0
) -> jnp.ndarray:
    """
    Vectorized computation of Modified Time-To-Collision for ALL agents.
    
    Uses proper MTTC formula: MTTC = (-ΔV ± √(ΔV² + 2·Δa·D)) / Δa
    where ΔV is relative velocity, Δa is relative acceleration, D is distance.
    
    Args:
        state: Current simulator state
        ego_idx: Ego vehicle index
        max_timesteps: Maximum timesteps to consider
        critical_ttc: Time threshold for critical risk
        safe_ttc: Time threshold for safe operation
    
    Returns:
        mttc_risk: (num_objects,) array with risk values [0, 1] for each object
    """
    num_objects = state.log_trajectory.x.shape[0]
    traj_length = state.log_trajectory.x.shape[1]
    T = jnp.minimum(max_timesteps, traj_length)
    
    time_mask = jnp.arange(traj_length) < T
    
    lengths_all = state.log_trajectory.length
    widths_all = state.log_trajectory.width

    # === GET EGO TRAJECTORY ===
    x_ego = jnp.where(time_mask, state.log_trajectory.x[ego_idx, :], 0.0)
    y_ego = jnp.where(time_mask, state.log_trajectory.y[ego_idx, :], 0.0)
    vx_ego = jnp.where(time_mask, state.log_trajectory.vel_x[ego_idx, :], 0.0)
    vy_ego = jnp.where(time_mask, state.log_trajectory.vel_y[ego_idx, :], 0.0)
    
    v_ego = jnp.stack([vx_ego, vy_ego], axis=-1)  # (T, 2)
    p_ego = jnp.stack([x_ego, y_ego], axis=-1)    # (T, 2)
    heading_ego = jnp.arctan2(v_ego[..., 1], v_ego[..., 0])
    heading_ego = jnp.unwrap(heading_ego, axis=0)
    
    # Compute ego acceleration (finite differences)
    a_ego = jnp.zeros_like(v_ego)
    a_ego = a_ego.at[1:].set((v_ego[1:] - v_ego[:-1]) / 0.1)  # dt = 0.1s
    
    # === GET ALL OBJECT TRAJECTORIES ===
    x_all = jnp.where(time_mask[None, :], state.log_trajectory.x, 0.0)
    y_all = jnp.where(time_mask[None, :], state.log_trajectory.y, 0.0)
    vx_all = jnp.where(time_mask[None, :], state.log_trajectory.vel_x, 0.0)
    vy_all = jnp.where(time_mask[None, :], state.log_trajectory.vel_y, 0.0)
    
    v_all = jnp.stack([vx_all, vy_all], axis=-1)  # (num_objects, T, 2)
    p_all = jnp.stack([x_all, y_all], axis=-1)    # (num_objects, T, 2)
    heading_all = jnp.arctan2(v_all[..., 1], v_all[..., 0])
    heading_all = jnp.unwrap(heading_all, axis=0)

    # Compute all object accelerations
    a_all = jnp.zeros_like(v_all)
    a_all = a_all.at[:, 1:, :].set((v_all[:, 1:, :] - v_all[:, :-1, :]) / 0.1)
    
    # === COMPUTE RELATIVE QUANTITIES ===
    rel_pos = p_all - p_ego[None, :, :]  # (num_objects, T, 2)
    rel_vel = v_all - v_ego[None, :, :]  # (num_objects, T, 2)
    rel_acc = a_all - a_ego[None, :, :]  # (num_objects, T, 2)
    
    # === PROJECT TO 1D (LONGITUDINAL) ===
    # Use ego velocity direction as longitudinal axis
    ego_speed = jnp.linalg.norm(v_ego, axis=1, keepdims=True) + 1e-6  # (T, 1)
    ego_dir = v_ego / ego_speed  # (T, 2) - normalized direction
    
    # Project relative quantities onto ego direction
    D = jnp.einsum('kti,ti->kt', rel_pos, ego_dir)  # (num_objects, T) - relative distance
    delta_V = jnp.einsum('kti,ti->kt', rel_vel, ego_dir)  # (num_objects, T) - relative velocity
    delta_a = jnp.einsum('kti,ti->kt', rel_acc, ego_dir)  # (num_objects, T) - relative acceleration
    
    # === COMPUTE MTTC USING PROPER FORMULA ===
    # MTTC = (-ΔV ± √(ΔV² + 2·Δa·D)) / Δa
    
    # Discriminant
    discriminant = delta_V**2 + 2 * delta_a * np.abs(D)  # (num_objects, T)
    
    # Only valid if discriminant >= 0
    valid_discriminant = discriminant >= 0
    
    # Compute both solutions
    sqrt_term = jnp.sqrt(jnp.maximum(discriminant, 0.0))
    
    # Two possible MTTC values
    mttc_1 = (-delta_V + sqrt_term) / (delta_a + 1e-9)  # Add small epsilon to avoid division by zero
    mttc_2 = (-delta_V - sqrt_term) / (delta_a + 1e-9)
    
    # Select smallest positive value
    # Mark negative or invalid values as inf
    mttc_1 = jnp.where((mttc_1 > 0) & valid_discriminant, mttc_1, jnp.inf)
    mttc_2 = jnp.where((mttc_2 > 0) & valid_discriminant, mttc_2, jnp.inf)
    
    mttc_per_timestep = jnp.minimum(mttc_1, mttc_2)  # (num_objects, T)
    
    # === SPECIAL CASE: Δa ≈ 0 (constant velocity) ===
    # When Δa is very small, use TTC = D / (-ΔV)
    small_accel = jnp.abs(delta_a) < 1e-3
    ttc_constant_vel = D / (-delta_V + 1e-9)
    ttc_constant_vel = jnp.where(
        (ttc_constant_vel > 0) & (delta_V < 0),  # Approaching
        ttc_constant_vel,
        jnp.inf
    )
    
    # Use constant velocity TTC when acceleration is negligible
    mttc_per_timestep = jnp.where(small_accel, ttc_constant_vel, mttc_per_timestep)
    
    # === CHECK APPROACHING CONDITION ===
    def valid_mttc_condition(pos_rel, vel_rel, v_ego, v_all, dist, ego_heading, fov_angle=180, min_dist=1.0):
        """
        Determine if MTTC is applicable based on spatial and directional constraints.

        Args:
            pos_rel: (N,2) relative position vectors (x,y)
            vel_rel: (N,2) relative velocity vectors (x,y)
            ego_heading: ego vehicle heading unit vector (2,)
            fov_angle: field-of-view (degrees) in which MTTC applies (e.g. 180 for front, 360 for global)
            min_dist: minimum distance threshold to avoid noise near ego

        Returns:
            mask: boolean (N,) array where MTTC is valid
        """
        # Normalize vectors
        pos_norm = pos_rel / (jnp.linalg.norm(pos_rel, axis=2, keepdims=True) + 1e-6)
        vel_norm = vel_rel / (jnp.linalg.norm(vel_rel, axis=2, keepdims=True) + 1e-6)

        # Relative motion direction: cosine of angle between position and relative velocity
        cos_theta = jnp.sum(pos_norm * vel_norm, axis=2)
        
        ego_heading_vec = jnp.stack([jnp.cos(ego_heading), jnp.sin(ego_heading)], axis=1)  # (T, 2)

        # Cosine of angle between position vector and ego heading direction
        cos_heading = jnp.einsum('ntc,tc->nt', pos_norm, ego_heading_vec)

        # Condition 1: moving toward ego (cos_theta < 0)
        moving_toward = cos_theta < -0.1

        # Condition 2: within ego field of view
        # within_fov = cos_heading > jnp.cos(jnp.deg2rad(fov_angle / 2))

        # Condition 3: object direction consistent with ego direction (not opposite)
        dot = jnp.einsum('ij,kij->ki', v_ego, v_all)  # (num_objects, T)
        norm_av = jnp.linalg.norm(v_ego, axis=1)      # (T,)
        norm_all = jnp.linalg.norm(v_all, axis=2)    # (num_objects, T)
        norm_product = norm_av[None, :] * norm_all   # (num_objects, T)
        cos_angle = jnp.where(norm_product > 0, dot / norm_product, 0.0)
        same_direction = cos_angle > 0.9

        valid_mask = moving_toward & same_direction
        return valid_mask
    
    valid = valid_mttc_condition(rel_pos, rel_vel, v_ego, v_all, D, heading_ego, heading_all)
    mttc_per_timestep = jnp.where(valid, mttc_per_timestep, jnp.inf)

    # === AGGREGATE: Minimum MTTC across trajectory ===
    min_mttc_per_object = jnp.min(mttc_per_timestep, axis=1)  # (num_objects,)
    
    # === CONVERT MTTC TO RISK ===
    def mttc_to_risk_vectorized(mttc_val):
        mttc_val = jnp.clip(mttc_val, 1e-3, None)
        decay_rate = jnp.log(2) / (safe_ttc - critical_ttc)
        risk = jnp.exp(-decay_rate * (mttc_val - critical_ttc))
        risk = jnp.clip(risk, 0.0, 1.0)
        return risk
    
    risk_values = jax.vmap(mttc_to_risk_vectorized)(min_mttc_per_object)
    
    # === MASK OUT EGO AND INVALID OBJECTS ===
    valid_mask = state.object_metadata.is_valid
    ego_mask = jnp.arange(num_objects) == ego_idx
    
    # Set ego and invalid objects to zero risk
    risk_values = jnp.where(valid_mask & ~ego_mask, risk_values, 0.0)
    
    return risk_values  # (num_objects,)


def create_mttc_risk_grid(
    state: datatypes.SimulatorState,
    ego_idx: int,
    grid_size: int = 64,
    grid_range_long: float = 50.0,  # Longitudinal: ±50m (100m total)
    grid_range_lat: float = 15.0,   # Lateral: ±15m (30m total) - road width
    prediction_horizon: float = 5.0,
    num_prediction_steps: int = 3
) -> jnp.ndarray:
    """
    Create ground truth risk grid based on vectorized MTTC computation.
    
    Grid is rectangular to match road geometry:
    - Longitudinal (forward/back): ±50m (need long prediction horizon)
    - Lateral (left/right): ±15m (typical 3-4 lane road width)
    
    Accounts for actual vehicle dimensions using length/width attributes.
    
    Args:
        state: Current simulator state
        ego_idx: Ego vehicle index
        grid_size: Spatial resolution (applies to both dimensions)
        grid_range_long: Longitudinal range in meters
        grid_range_lat: Lateral range in meters
        prediction_horizon: How far to predict forward (seconds)
        num_prediction_steps: Number of prediction steps
    
    Returns:
        risk_grid: (1, grid_size, grid_size, 1) with MTTC-based risk
                   Note: grid is rectangular in world space but square in grid space
    """
    # Compute MTTC risk for all objects (vectorized)
    risk_per_object = compute_mttc_vectorized(state, ego_idx)  # (num_objects,)
    
    # Get current positions and velocities
    current_timestep = state.timestep
    positions = jnp.stack([
        state.sim_trajectory.x[:, current_timestep],
        state.sim_trajectory.y[:, current_timestep]
    ], axis=-1)  # (num_objects, 2)  # (num_objects, 2)
    velocities = jnp.stack([
        state.sim_trajectory.vel_x[:, current_timestep],
        state.sim_trajectory.vel_y[:, current_timestep]
    ], axis=-1)  # (num_objects, 2)
    
    # Get vehicle dimensions
    lengths = state.sim_trajectory.length[:, current_timestep]  # (num_objects,)
    widths = state.sim_trajectory.width[:, current_timestep]    # (num_objects,)
    yaws = state.sim_trajectory.yaw[:, current_timestep]        # (num_objects,)
    
    ego_pos = positions[ego_idx]
    ego_vel = velocities[ego_idx]
    ego_yaw = yaws[ego_idx]
    
    # Get ego's forward direction (longitudinal axis)
    ego_forward = jnp.array([jnp.cos(ego_yaw), jnp.sin(ego_yaw)])
    ego_lateral = jnp.array([-jnp.sin(ego_yaw), jnp.cos(ego_yaw)])  # perpendicular
    
    # Initialize grid
    risk_grid = jnp.zeros((grid_size, grid_size), dtype=jnp.float32)
    
    # Cell sizes (rectangular cells in world space)
    cell_size_long = 2 * grid_range_long / grid_size  # ~1.56m per cell longitudinally
    cell_size_lat = 2 * grid_range_lat / grid_size    # ~0.47m per cell laterally
    
    max_vehicle_cells_long = int(jnp.ceil(10.0 / cell_size_long).item()) + 1
    max_vehicle_cells_lat = int(jnp.ceil(3.0 / cell_size_lat).item()) + 1

    dt = prediction_horizon / num_prediction_steps
    
    dx_range = jnp.arange(-max_vehicle_cells_long, max_vehicle_cells_long + 1)
    dy_range = jnp.arange(-max_vehicle_cells_lat, max_vehicle_cells_lat + 1)
    dx_grid, dy_grid = jnp.meshgrid(dx_range, dy_range, indexing='ij')
    dx_flat = dx_grid.flatten()
    dy_flat = dy_grid.flatten()

    # Vectorized grid creation
    def add_object_risk_to_grid(carry, obj_idx):
        grid_accum = carry
        
        # Get object info
        obj_risk = risk_per_object[obj_idx]
        obj_pos = positions[obj_idx]
        obj_vel = velocities[obj_idx]
        obj_length = lengths[obj_idx]
        obj_width = widths[obj_idx]
        obj_yaw = yaws[obj_idx]
        
        # Check if object is valid and has non-zero risk
        is_valid = state.object_metadata.is_valid[obj_idx] & (obj_idx != ego_idx) & (obj_risk > 0.01)
        
        def project_object():
            # Project future positions
            time_steps = jnp.arange(num_prediction_steps) * dt
            
            # Predicted positions (constant velocity)
            pred_obj_pos = obj_pos[None, :] + obj_vel[None, :] * time_steps[:, None]  # (steps, 2)
            
            # Transform to ego-centric frame
            rel_pos_world = pred_obj_pos - ego_pos[None, :]  # (steps, 2)
            
            # Project onto ego's longitudinal and lateral axes
            rel_long = jnp.einsum('si,i->s', rel_pos_world, ego_forward)  # (steps,)
            rel_lat = jnp.einsum('si,i->s', rel_pos_world, ego_lateral)   # (steps,)
            
            # Convert to grid coordinates (rectangular grid)
            grid_x = ((rel_long + grid_range_long) / cell_size_long).astype(jnp.int32)
            grid_y = ((rel_lat + grid_range_lat) / cell_size_lat).astype(jnp.int32)
            
            # Calculate footprint size based on vehicle dimensions
            # How many cells does this vehicle occupy?
            obj_cells_long = jnp.maximum(jnp.ceil(obj_length / cell_size_long).astype(jnp.int32), 1)
            obj_cells_lat = jnp.maximum(jnp.ceil(obj_width / cell_size_lat).astype(jnp.int32), 1)
            
            # Create risk contribution for each prediction step
            def add_prediction_step(grid_inner, step_idx):
                center_x = grid_x[step_idx]
                center_y = grid_y[step_idx]
                
                # Check if center is in bounds
                in_bounds = (center_x >= 0) & (center_x < grid_size) & \
                           (center_y >= 0) & (center_y < grid_size)
                
                # Time decay
                time_decay = 1.0 - (step_idx / num_prediction_steps) * 0.3
                decayed_risk = obj_risk * time_decay
                
                def spread_vehicle_footprint(grid_spread):
                    # Use PRE-COMPUTED static offset arrays
                    # (dx_flat and dy_flat are defined outside, not traced)
                    
                    # Compute neighbor coordinates for all offsets at once
                    ngx_all = center_x + dx_flat  # shape: (num_neighbors,)
                    ngy_all = center_y + dy_flat  # shape: (num_neighbors,)
                    
                    # Check bounds for all neighbors
                    in_neighbor_bounds = (ngx_all >= 0) & (ngx_all < grid_size) & \
                                        (ngy_all >= 0) & (ngy_all < grid_size)
                    
                    # Actual vehicle footprint (dynamic)
                    half_long = obj_cells_long / 2.0 + 1.0
                    half_lat = obj_cells_lat / 2.0 + 1.0
                    
                    # Distance from vehicle center (vectorized)
                    # Only apply decay within actual vehicle footprint
                    dist_long = jnp.abs(dx_flat.astype(jnp.float32)) / (half_long + 1e-6)
                    dist_lat = jnp.abs(dy_flat.astype(jnp.float32)) / (half_lat + 1e-6)
                    dist = jnp.sqrt(dist_long**2 + dist_lat**2)
                    
                    # Gaussian-like decay (zero outside vehicle footprint)
                    within_footprint = (dist <= 2.0)  # 2.0 gives some margin
                    spatial_decay = jnp.where(within_footprint, jnp.exp(-dist**2 / 0.5), 0.0)
                    cell_risks = decayed_risk * spatial_decay
                    
                    # Update grid using scatter_max for all neighbors at once
                    # Filter to only valid neighbors
                    valid_mask = in_neighbor_bounds & (cell_risks > 1e-4)  # Skip negligible risk
                    valid_ngx = jnp.where(valid_mask, ngx_all, 0)
                    valid_ngy = jnp.where(valid_mask, ngy_all, 0)
                    valid_risks = jnp.where(valid_mask, cell_risks, 0.0)
                    
                    # Convert 2D indices to 1D for scatter operation
                    flat_indices = valid_ngx * grid_size + valid_ngy
                    
                    # Flatten current grid
                    flat_grid = grid_spread.flatten()
                    
                    # Use scatter_max to update (takes maximum risk value)
                    updated_flat = flat_grid.at[flat_indices].max(valid_risks)
                    
                    # Reshape back to 2D
                    return updated_flat.reshape(grid_size, grid_size)
                
                return jax.lax.cond(
                    in_bounds,
                    spread_vehicle_footprint,
                    lambda g: g,
                    grid_inner
                )
            
            # Apply all prediction steps
            grid_with_object = jax.lax.fori_loop(
                0, num_prediction_steps,
                lambda i, g: add_prediction_step(g, i),
                grid_accum
            )
            
            return grid_with_object
        
        # Only add if valid object
        new_grid = jax.lax.cond(
            is_valid,
            project_object,
            lambda: grid_accum
        )
        
        return new_grid, None
    
    # Process all objects
    final_grid, _ = jax.lax.scan(
        add_object_risk_to_grid,
        risk_grid,
        jnp.arange(state.sim_trajectory.num_objects)
    )
    
    return final_grid[None, ..., None]  # Add batch and channel dims

# ============================================================================
# MULTI-AGENT OBSERVATION EXTRACTION
# ============================================================================

def extract_multi_agent_observations(
    state: datatypes.SimulatorState,
    ego_idx: int,
    history_length: int = 10,
    max_agents: int = 8
) -> jnp.ndarray:
    """
    Extract observations for multiple surrounding agents (not just lead vehicle).
    
    Returns observations for the K nearest agents to ego vehicle.
    
    Args:
        state: Current state
        ego_idx: Ego vehicle index
        history_length: Number of past timesteps
        max_agents: Maximum number of agents to consider
    
    Returns:
        observations: (history_length, max_agents * 6) array
                     For each agent: [rel_x, rel_y, rel_vx, rel_vy, speed, distance]
    """
    current_timestep = state.timestep
    ego_pos_current = state.sim_trajectory.xy[ego_idx, current_timestep]
    
    # Find K nearest valid agents
    distances = []
    valid_indices = []
    
    for obj_idx in range(state.sim_trajectory.num_objects):
        if obj_idx == ego_idx:
            continue
        if not state.object_metadata.is_valid[obj_idx]:
            continue
        
        obj_pos = state.sim_trajectory.xy[obj_idx, current_timestep]
        dist = jnp.linalg.norm(obj_pos - ego_pos_current)
        
        distances.append(dist)
        valid_indices.append(obj_idx)
    
    # Sort by distance and take closest max_agents
    if len(distances) > 0:
        sorted_idx = jnp.argsort(jnp.array(distances))
        nearest_indices = [valid_indices[i] for i in sorted_idx[:max_agents]]
    else:
        nearest_indices = []
    
    # Pad if fewer than max_agents
    while len(nearest_indices) < max_agents:
        nearest_indices.append(-1)  # Invalid index
    
    observations = []
    
    for t in range(max(0, current_timestep - history_length + 1), current_timestep + 1):
        ego_pos = state.sim_trajectory.xy[ego_idx, t]
        ego_vel = jnp.array([
            state.sim_trajectory.vel_x[ego_idx, t],
            state.sim_trajectory.vel_y[ego_idx, t]
        ])
        
        timestep_features = []
        
        for agent_idx in nearest_indices:
            if agent_idx == -1:  # Padded (no agent)
                # Zero features
                agent_features = jnp.zeros(6)
            else:
                agent_pos = state.sim_trajectory.xy[agent_idx, t]
                agent_vel = jnp.array([
                    state.sim_trajectory.vel_x[agent_idx, t],
                    state.sim_trajectory.vel_y[agent_idx, t]
                ])
                
                rel_pos = agent_pos - ego_pos
                rel_vel = agent_vel - ego_vel
                agent_speed = jnp.linalg.norm(agent_vel)
                distance = jnp.linalg.norm(rel_pos)
                
                agent_features = jnp.array([
                    rel_pos[0],
                    rel_pos[1],
                    rel_vel[0],
                    rel_vel[1],
                    agent_speed,
                    distance
                ])
            
            timestep_features.append(agent_features)
        
        # Flatten all agent features for this timestep
        obs = jnp.concatenate(timestep_features)
        observations.append(obs)
    
    # Pad history if needed
    while len(observations) < history_length:
        observations.insert(0, observations[0])
    
    return jnp.stack(observations)


# ============================================================================
# UPDATED MODEL ARCHITECTURE FOR MULTI-AGENT
# ============================================================================

def update_model_for_multi_agent():
    """
    Pseudo-code showing how to update the CNN model for true multi-agent input.
    """
    
    # OLD: Single lead vehicle
    # obs_features = 6  # [rel_x, rel_y, rel_vx, rel_vy, ego_speed, lead_speed]
    
    # NEW: Multiple agents
    max_agents = 8
    obs_features = max_agents * 6  # 6 features per agent
    
    # Model architecture stays the same, just different input dimension
    # The temporal attention will learn to focus on relevant agents
    
    return obs_features


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def example_mttc_ground_truth():
    """Example of creating MTTC-based ground truth."""
    
    # During training:
    risk_grid_mttc = create_mttc_risk_grid(
        state,
        ego_idx,
        grid_size=64,
        grid_range=50.0
    )
    
    # Or potential field approach:
    risk_grid_potential = create_potential_field_risk_grid(
        state,
        ego_idx,
        grid_size=64,
        grid_range=50.0
    )
    
    # Use for training
    batch = {
        'observations': observations[None, ...],
        'risk_labels': risk_grid_mttc  # or risk_grid_potential
    }


def example_multi_agent_observations():
    """Example of extracting multi-agent observations."""
    
    # Extract observations from K nearest agents
    observations = extract_multi_agent_observations(
        state,
        ego_idx,
        history_length=10,
        max_agents=8  # Consider 8 nearest vehicles
    )
    
    # Shape: (history_length, 8 * 6) = (10, 48)
    
    # No need for lead_idx anymore!
    # The model learns which agents are most relevant


if __name__ == "__main__":
    print(__doc__)
    print("\nKey improvements:")
    print("1. MTTC-based ground truth (physically motivated)")
    print("2. Considers ALL agents (truly multi-agent)")
    print("3. No need for lead_idx - model learns relevant agents")
    print("4. Includes predicted trajectories in risk assessment")