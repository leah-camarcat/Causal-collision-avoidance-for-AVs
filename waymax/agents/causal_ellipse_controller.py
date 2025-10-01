from typing import Callable
import jax
import jax.numpy as jnp
from waymax import datatypes, dynamics
from waymax.agents import actor_core

def causal_ellipse_actor_old(
    dynamics_model: dynamics.DynamicsModel,
    is_controlled_func: Callable[[datatypes.SimulatorState], jax.Array],
    av_idx: int,
    neigh_idx: int,
) -> actor_core.WaymaxActorCore:
    """Actor that uses an oriented safety ellipse with counterfactual reasoning
    to evaluate collision risk for both longitudinal and lateral evasive maneuvers.
    """

    # Parameters for ellipse
    m_long = 0.8
    m_lat = 0.5
    kappa_a = 1.0   # time headway scaling (s)
    kappa_b = 0.05  # lateral margin scaling

    # Candidate accelerations and steering angles to search
    NUM_ACCEL_CANDIDATES = 21
    NUM_STEER_CANDIDATES = 11
    MAX_BRAKE = 4.0
    MAX_ACCEL = 2.0
    MAX_STEER_ANGLE = jnp.pi / 6  # 30 degrees max steering
    
    # Counterfactual simulation parameters
    PREDICTION_HORIZON = 3.0  # seconds to look ahead
    NUM_PREDICTION_STEPS = int(PREDICTION_HORIZON / datatypes.TIME_INTERVAL)
    
    # Risk assessment parameters
    IMMINENT_COLLISION_TIME = 1.0  # seconds - only intervene if collision within this time
    IMMINENT_STEPS = int(IMMINENT_COLLISION_TIME / datatypes.TIME_INTERVAL)
    SAFETY_BUFFER = 0.01  # additional safety margin for intervention trigger

    def compute_axes(v_long: jax.Array, L:jax.Array, W:jax.Array):
        """Compute ellipse semi-axes given forward velocity."""
        a = (L / 2 + m_long) + kappa_a * jnp.maximum(v_long, 0.0)
        b = (W / 2 + m_lat) + kappa_b * jnp.maximum(v_long, 0.0)
        return a, b

    def signed_ellipse_value(rel_pos: jax.Array, psi: jax.Array, a: jax.Array, b: jax.Array):
        """Compute g(r) = (rx/a)^2 + (ry/b)^2 - 1 in ego's frame."""
        c = jnp.cos(psi)
        s = jnp.sin(psi)
        Rm = jnp.array([[c, s], [-s, c]])
        r = Rm @ rel_pos
        return (r[0] / a) ** 2 + (r[1] / b) ** 2 - 1.0

    def predict_neighbor_trajectory(pos_N_init, vel_N_init):
        """Predict neighbor trajectory assuming constant velocity model.
        In practice, this could be replaced with a more sophisticated predictor.
        """
        def step_neighbor(carry, _):
            pos, vel = carry
            new_pos = pos + vel * datatypes.TIME_INTERVAL
            return (new_pos, vel), new_pos
        
        _, neighbor_positions = jax.lax.scan(
            step_neighbor, 
            (pos_N_init, vel_N_init), 
            jnp.arange(NUM_PREDICTION_STEPS)
        )
        # Ensure shape is (NUM_PREDICTION_STEPS, 2)
        return jnp.reshape(neighbor_positions, (NUM_PREDICTION_STEPS, 2))

    def simulate_ego_trajectory(pos_E_init, vel_E_init, psi_init, accel_cmd, steer_cmd):
        """Simulate ego trajectory under counterfactual control inputs.
        Uses simple kinematic bicycle model.
        """
        def step_ego(carry, _):
            pos, vel, psi, speed = carry
            
            # Update heading with steering command
            new_psi = psi + steer_cmd
            
            # Update speed with acceleration command  
            new_speed = jnp.maximum(speed + accel_cmd * datatypes.TIME_INTERVAL, 0.0)
            
            # Update velocity in global frame
            new_vel = jnp.array([jnp.cos(new_psi), jnp.sin(new_psi)]) * new_speed
            
            # Update position
            new_pos = pos + new_vel * datatypes.TIME_INTERVAL
            
            return (new_pos, new_vel, new_psi, new_speed), (new_pos, new_vel, new_psi, new_speed)
        
        initial_speed = jnp.linalg.norm(vel_E_init)
        _, trajectory = jax.lax.scan(
            step_ego,
            (pos_E_init, vel_E_init, psi_init, initial_speed),
            jnp.arange(NUM_PREDICTION_STEPS)
        )
        
        ego_positions, ego_velocities, ego_headings, ego_speeds = trajectory
        # Ensure consistent shapes: (NUM_PREDICTION_STEPS, 2), (NUM_PREDICTION_STEPS,), etc.
        ego_positions = jnp.reshape(ego_positions, (NUM_PREDICTION_STEPS, 2))
        ego_velocities = jnp.reshape(ego_velocities, (NUM_PREDICTION_STEPS, 2))
        ego_headings = jnp.reshape(ego_headings, (NUM_PREDICTION_STEPS,))
        ego_speeds = jnp.reshape(ego_speeds, (NUM_PREDICTION_STEPS,))
        return ego_positions, ego_velocities, ego_headings, ego_speeds
        
    def assess_baseline_collision_risk(pos_E, pos_N, vel_E, vel_N, psi, L, W):
        """Assess collision risk if ego continues current behavior (no evasive action).
        Returns whether immediate intervention is needed.
        """
        # Simulate baseline trajectory (constant velocity)
        ego_positions_baseline, _, ego_headings_baseline, ego_speeds_baseline = simulate_ego_trajectory(
            pos_E, vel_E, psi, accel_cmd=0.0, steer_cmd=0.0  # No evasive action
        )
        neighbor_positions_baseline = predict_neighbor_trajectory(pos_N, vel_N)
        
        # Check for collision in the imminent future (not full horizon)
        def check_imminent_collision(step_idx):
            ego_pos = ego_positions_baseline[step_idx]
            neighbor_pos = neighbor_positions_baseline[step_idx]
            ego_speed = ego_speeds_baseline[step_idx]
            ego_psi = ego_headings_baseline[step_idx]
            
            rel_pos = neighbor_pos - ego_pos
            a, b = compute_axes(ego_speed, L, W)
            g_val = signed_ellipse_value(rel_pos, ego_psi, a, b)
            
            return g_val
        
        # Only check imminent time window, not full prediction horizon
        imminent_steps = jnp.minimum(IMMINENT_STEPS, NUM_PREDICTION_STEPS)
        g_values_imminent = jax.vmap(check_imminent_collision)(jnp.arange(IMMINENT_STEPS))
        #jax.debug.print('ellipse values: {}', g_values_imminent)
        # Determine if intervention is needed
        min_g_imminent = jnp.min(g_values_imminent)
        jax.debug.print('min ellipse values: {}', min_g_imminent)
        intervention_needed = min_g_imminent < SAFETY_BUFFER
        jax.debug.print('intervention_needed: {}', intervention_needed)
        
        return intervention_needed, min_g_imminent

    def evaluate_counterfactual_collision(pos_E, pos_N, vel_E, vel_N, psi, accel_cmd, steer_cmd, L, W):
        """Counterfactual reasoning: What if ego applies (accel_cmd, steer_cmd)?
        
        Causal chain:
        Initial conditions → Evasive maneuver → Future trajectories → Collision outcome
        """
        
        # STEP 1: Predict future trajectories under counterfactual action
        ego_positions, ego_velocities, ego_headings, ego_speeds = simulate_ego_trajectory(
            pos_E, vel_E, psi, accel_cmd, steer_cmd
        )
        neighbor_positions = predict_neighbor_trajectory(pos_N, vel_N)
        
        # STEP 2: Evaluate collision risk at each future timestep
        def check_collision_at_step(step_idx):
            ego_pos = ego_positions[step_idx]  # shape: (2,)
            ego_vel = ego_velocities[step_idx]  # shape: (2,)
            ego_psi = ego_headings[step_idx]    # shape: ()
            ego_speed = ego_speeds[step_idx]    # shape: ()
            neighbor_pos = neighbor_positions[step_idx]  # shape: (2,)
            
            # Compute relative position in future scenario
            rel_pos_future = neighbor_pos - ego_pos  # shape: (2,)
            
            # Compute ellipse parameters based on ego's future speed
            a, b = compute_axes(ego_speed, L, W)
            
            # Evaluate ellipse constraint in ego's future frame
            g_val = signed_ellipse_value(rel_pos_future, ego_psi, a, b)
            
            return g_val

        # Check collision over prediction horizon
        g_values = jax.vmap(check_collision_at_step)(jnp.arange(NUM_PREDICTION_STEPS))
        
        # STEP 3: Determine collision outcome
        # Collision occurs if ellipse constraint is violated at any future time
        min_g_value = jnp.min(g_values)
        collision_risk = min_g_value < 0.0
        
        return min_g_value, collision_risk

    def actor_init(rng, init_state):
        return {"reaction_timer": jnp.array(0, dtype=jnp.int32)}

    def select_action(params, state: datatypes.SimulatorState, actor_state=None, rng=None):
        is_controlled = is_controlled_func(state)

        # Extract current states
        traj_t0 = datatypes.dynamic_index(state.sim_trajectory, state.timestep, axis=-1, keepdims=True)
        traj_prev = datatypes.dynamic_index(
            state.sim_trajectory,
            jnp.maximum(state.timestep - 1, 0),
            axis=-1,
            keepdims=True,
        )

        # Current ego state
        pos_E = jnp.array([traj_t0.x[av_idx, 0], traj_t0.y[av_idx, 0]])
        vel_E = jnp.array([traj_t0.vel_x[av_idx, 0], traj_t0.vel_y[av_idx, 0]])
        speed_E = jnp.linalg.norm(vel_E)
        psi = jnp.where(speed_E > 1e-3, jnp.arctan2(vel_E[1], vel_E[0]), 0.0)
        L = traj_t0.length[av_idx]
        W = traj_t0.width[av_idx]
        # Current neighbor state  
        pos_N = jnp.array([traj_t0.x[neigh_idx, 0], traj_t0.y[neigh_idx, 0]])
        vel_N = jnp.array([traj_t0.vel_x[neigh_idx, 0], traj_t0.vel_y[neigh_idx, 0]])

        # Current ego acceleration (for preference)
        vel_E_prev = jnp.array([traj_prev.vel_x[av_idx, 0], traj_prev.vel_y[av_idx, 0]])
        acc_E_current = (jnp.linalg.norm(vel_E) - jnp.linalg.norm(vel_E_prev)) / datatypes.TIME_INTERVAL

        # FIRST: Assess if intervention is needed at all
        intervention_needed, baseline_risk = assess_baseline_collision_risk(
            pos_E, pos_N, vel_E, vel_N, psi, L, W
        )
        
        def select_evasive_action():
            """Only called when intervention is actually needed."""
            # Generate candidate control actions
            accel_candidates = jnp.linspace(MAX_ACCEL, -MAX_BRAKE, NUM_ACCEL_CANDIDATES)
            steer_candidates = jnp.linspace(-MAX_STEER_ANGLE, MAX_STEER_ANGLE, NUM_STEER_CANDIDATES)
            
            # Create all combinations of actions
            accel_grid, steer_grid = jnp.meshgrid(accel_candidates, steer_candidates, indexing='ij')
            accel_flat = accel_grid.flatten()
            steer_flat = steer_grid.flatten()

            def eval_candidate_pair(accel_cmd, steer_cmd):
                """Counterfactual evaluation: What if we apply this (accel, steer) pair?"""
                min_g_val, collision_risk = evaluate_counterfactual_collision(
                    pos_E, pos_N, vel_E, vel_N, psi, accel_cmd, steer_cmd, L, W
                )
                return min_g_val

            # Evaluate all candidate actions using counterfactual reasoning
            safety_margins = jax.vmap(eval_candidate_pair)(accel_flat, steer_flat)
            safe_mask = safety_margins >= 0.0
            any_safe = jnp.any(safe_mask)

            def choose_safe_action():
                # Find all safe action pairs
                safe_indices = jnp.nonzero(safe_mask, size=len(accel_flat))[0]
                safe_accels = accel_flat[safe_indices]
                safe_steers = steer_flat[safe_indices]
                safe_margins = safety_margins[safe_indices]
                
                # Preference for actions close to current behavior
                accel_preferences = jnp.abs(safe_accels - acc_E_current)
                steer_preferences = jnp.abs(safe_steers)  # prefer straight
                
                # Combined preference score (lower is better)
                w_accel = 1.0
                w_steer = 2.0
                w_safety = -0.5  # prefer higher safety margins
                
                preference_scores = (w_accel * accel_preferences + 
                                   w_steer * steer_preferences + 
                                   w_safety * safe_margins)
                
                best_idx = jnp.argmin(preference_scores)
                return safe_accels[best_idx], safe_steers[best_idx]

            def choose_emergency_action():
                # Emergency: maximum braking, no steering
                return -MAX_BRAKE, 0.0

            return jax.lax.cond(any_safe, choose_safe_action, choose_emergency_action)
        
        def maintain_current_behavior():
            """Continue current behavior - no intervention needed."""
            # Clamp current acceleration to reasonable bounds
            clamped_accel = jnp.clip(acc_E_current, -MAX_BRAKE, MAX_ACCEL)
            return clamped_accel, 0.0  # maintain current acceleration, no steering
        
        # Only apply evasive maneuver if intervention is actually needed
        accel_selected, steer_selected = jax.lax.cond(
            intervention_needed,
            select_evasive_action,
            maintain_current_behavior
        )

        # Apply selected action for one timestep
        new_psi = psi + steer_selected
        new_speed = jnp.maximum(speed_E + accel_selected * datatypes.TIME_INTERVAL, 0.0)
        new_vel = jnp.array([jnp.cos(new_psi), jnp.sin(new_psi)]) * new_speed
        new_pos = pos_E + new_vel * datatypes.TIME_INTERVAL

        # Update trajectory
        traj_t1 = traj_t0.replace(
            x=traj_t0.x.at[av_idx].set(new_pos[0]),
            y=traj_t0.y.at[av_idx].set(new_pos[1]),
            vel_x=traj_t0.vel_x.at[av_idx].set(new_vel[0]),
            vel_y=traj_t0.vel_y.at[av_idx].set(new_vel[1]),
            valid=is_controlled[..., None] & traj_t0.valid,
            timestamp_micros=traj_t0.timestamp_micros + datatypes.TIMESTEP_MICROS_INTERVAL,
        )

        traj_combined = jax.tree_util.tree_map(
            lambda x, y: jnp.concatenate([x, y], axis=-1), traj_t0, traj_t1
        )
        actions = dynamics_model.inverse(traj_combined, state.object_metadata, timestep=0)

        return actor_core.WaymaxActorOutput(
            actor_state=actor_state,
            action=actions,
            is_controlled=is_controlled,
        )

    return actor_core.actor_core_factory(
        init=actor_init,
        select_action=select_action,
        name="causal_ellipse_actor",
    )


def causal_ellipse_actor(
    dynamics_model: dynamics.DynamicsModel,
    is_controlled_func: Callable[[datatypes.SimulatorState], jax.Array],
    av_idx: int,
    neigh_idx: int,
) -> actor_core.WaymaxActorCore:
    """Actor using counterfactual reasoning with distinct longitudinal and lateral primitives.
    
    Key improvements:
    1. Separate evaluation of braking vs lane-change maneuvers
    2. Lane-change uses target lateral offset, not continuous steering
    3. Road-aware lateral bounds checking
    4. Clear intervention hierarchy
    """

    # Ellipse parameters
    m_long = 0.8
    m_lat = 0.5
    kappa_a = 1.0
    kappa_b = 0.05

    # Longitudinal action space
    NUM_BRAKE_CANDIDATES = 15
    MAX_BRAKE = 4.0  # m/s² - emergency braking
    MAX_ACCEL = 2.0
    COMFORT_BRAKE = 3.0  # m/s² - comfortable deceleration

    # Lateral action space (target offsets, not steering angles)
    LANE_WIDTH = 3.7  # meters - standard US lane
    LATERAL_OFFSETS = jnp.array([-LANE_WIDTH, 0.0, LANE_WIDTH])  # left lane, stay, right lane
    LANE_CHANGE_DURATION = 3.0  # seconds for a lane change
    
    # Prediction parameters
    PREDICTION_HORIZON = 4.0  # seconds
    NUM_PREDICTION_STEPS = int(PREDICTION_HORIZON / datatypes.TIME_INTERVAL)
    
    # Intervention thresholds
    IMMINENT_COLLISION_TIME = 2.0  # seconds
    #IMMINENT_STEPS = int(IMMINENT_COLLISION_TIME / datatypes.TIME_INTERVAL)
    # At the top of the function
    IMMINENT_STEPS = min(int(IMMINENT_COLLISION_TIME / datatypes.TIME_INTERVAL), NUM_PREDICTION_STEPS)
    SAFETY_MARGIN = 0.0
    
    # Road boundaries (simplified - should come from map in production)
    MAX_LATERAL_DEVIATION = 2.0 * LANE_WIDTH  # can move 2 lanes each direction

    def compute_axes(v_long: jax.Array, L: jax.Array, W: jax.Array):
        a = (L / 2 + m_long) + kappa_a * jnp.maximum(v_long, 0.0)
        b = (W / 2 + m_lat) + kappa_b * jnp.maximum(v_long, 0.0)
        return a, b

    def signed_ellipse_value(rel_pos: jax.Array, psi: jax.Array, a: jax.Array, b: jax.Array):
        c = jnp.cos(psi)
        s = jnp.sin(psi)
        Rm = jnp.array([[c, s], [-s, c]])
        r = Rm @ rel_pos
        return (r[0] / a) ** 2 + (r[1] / b) ** 2 - 1.0

    def get_road_aligned_frame(pos, vel, speed):
        """Get position in road-aligned coordinate system.
        Returns: (s, d, psi) where s=longitudinal, d=lateral, psi=heading
        """
        psi = jnp.where(speed > 1e-3, jnp.arctan2(vel[1], vel[0]), 0.0)
        # In full implementation, would project onto road centerline
        # For now, use current position as origin
        return pos[0], pos[1], psi

    def predict_neighbor_trajectory(pos_N_init, vel_N_init):
        """Constant velocity prediction for neighbor."""
        def step(carry, _):
            pos, vel = carry
            new_pos = pos + vel * datatypes.TIME_INTERVAL
            return (new_pos, vel), new_pos
        
        _, positions = jax.lax.scan(
            step, 
            (pos_N_init, vel_N_init), 
            jnp.arange(NUM_PREDICTION_STEPS)
        )
        return positions

    def simulate_longitudinal_maneuver(pos_E, vel_E, psi, lateral_offset_current, accel_long):
        """Simulate pure longitudinal control (braking/acceleration) with no lane change.
        
        Maintains current lateral position while adjusting speed.
        """
        def step(carry, _):
            pos, vel, speed, psi_curr, lat_offset = carry
            
            # Update speed with longitudinal acceleration
            new_speed = jnp.maximum(speed + accel_long * datatypes.TIME_INTERVAL, 0.0)
            
            # Velocity direction stays aligned with heading
            direction = jnp.array([jnp.cos(psi_curr), jnp.sin(psi_curr)])
            new_vel = direction * new_speed
            
            # Position update - maintain lateral offset
            new_pos = pos + new_vel * datatypes.TIME_INTERVAL
            
            return (new_pos, new_vel, new_speed, psi_curr, lat_offset), (new_pos, new_vel, psi_curr, new_speed)
        
        speed_init = jnp.linalg.norm(vel_E)
        _, trajectory = jax.lax.scan(
            step,
            (pos_E, vel_E, speed_init, psi, lateral_offset_current),
            jnp.arange(NUM_PREDICTION_STEPS)
        )
        
        positions, velocities, headings, speeds = trajectory
        return positions, velocities, headings, speeds

    def simulate_lane_change_maneuver(pos_E, vel_E, psi, lateral_offset_current, target_lateral_offset):
        """Simulate lane change to target lateral offset.
        
        Uses smooth lateral interpolation over LANE_CHANGE_DURATION.
        Maintains speed (could add optional deceleration during lane change).
        """
        total_steps = NUM_PREDICTION_STEPS
        lane_change_steps = int(LANE_CHANGE_DURATION / datatypes.TIME_INTERVAL)
        
        def step(carry, t_idx):
            pos, vel, speed, psi_curr, lat_offset = carry
            
            # Smooth lateral transition using sigmoid-like profile
            progress = jnp.minimum(t_idx / lane_change_steps, 1.0)
            # Use smooth step function: 3*progress^2 - 2*progress^3
            smooth_progress = 3 * progress**2 - 2 * progress**3
            
            new_lat_offset = lat_offset + (target_lateral_offset - lateral_offset_current) * smooth_progress
            
            # Compute lateral velocity needed for this transition
            def lat_vel_needed():
                return (target_lateral_offset - lateral_offset_current) / LANE_CHANGE_DURATION
            
            def lat_vel_not_needed():
                return 0.0
            
            lat_vel_needed = jax.lax.cond(
                t_idx < lane_change_steps, 
                lat_vel_needed,
                lat_vel_not_needed
                )
            
            # Heading adjusts to accommodate lateral motion
            # For small angles: tan(psi) ≈ v_lat / v_long
            speed_safe = jnp.maximum(speed, 0.1)  # avoid division by zero
            heading_adjustment = jnp.arctan2(lat_vel_needed, speed_safe)
            new_psi = psi + heading_adjustment
            
            # Velocity combines longitudinal and lateral components
            long_dir = jnp.array([jnp.cos(psi), jnp.sin(psi)])
            lat_dir = jnp.array([-jnp.sin(psi), jnp.cos(psi)])  # perpendicular to heading
            
            new_vel = long_dir * speed + lat_dir * lat_vel_needed
            new_speed = jnp.linalg.norm(new_vel)
            
            # Update position
            new_pos = pos + new_vel * datatypes.TIME_INTERVAL
            
            return (new_pos, new_vel, new_speed, new_psi, new_lat_offset), (new_pos, new_vel, new_psi, new_speed)
        
        speed_init = jnp.linalg.norm(vel_E)
        _, trajectory = jax.lax.scan(
            step,
            (pos_E, vel_E, speed_init, psi, lateral_offset_current),
            jnp.arange(total_steps)
        )
        
        positions, velocities, headings, speeds = trajectory
        return positions, velocities, headings, speeds

    def assess_baseline_risk(pos_E, pos_N, vel_E, vel_N, psi, L, W):
        """Assess collision risk if ego continues current behavior (no evasive action).
        
        Returns the minimum ellipse value over the imminent horizon.
        If < SAFETY_MARGIN, intervention is needed.
        """
        # Simulate baseline trajectory (no acceleration change)
        ego_positions, ego_velocities, ego_headings, ego_speeds = simulate_longitudinal_maneuver(
            pos_E, vel_E, psi, 0.0, accel_long=0.0  # maintain current speed
        )
        
        neighbor_positions = predict_neighbor_trajectory(pos_N, vel_N)
        
        # Check ellipse constraint over imminent horizon
        def check_ellipse_at_step(t_idx):
            rel_pos = neighbor_positions[t_idx] - ego_positions[t_idx]
            a, b = compute_axes(ego_speeds[t_idx], L, W)
            g_val = signed_ellipse_value(rel_pos, ego_headings[t_idx], a, b)
            return g_val
        
        # Only check imminent time window
        #g_values_imminent = jax.vmap(check_ellipse_at_step)(jnp.arange(imminent_steps))
        g_values_all = jax.vmap(check_ellipse_at_step)(jnp.arange(NUM_PREDICTION_STEPS))
        g_values_imminent = g_values_all[:IMMINENT_STEPS]
        min_g_imminent = jnp.min(g_values_imminent)
        
        return min_g_imminent
    
    def evaluate_maneuver(pos_E, pos_N, vel_E, vel_N, psi, L, W, 
                         maneuver_type, maneuver_param, lateral_offset_current):
        """Counterfactual evaluation of a specific maneuver.
        
        Args:
            maneuver_type: 0 for longitudinal (brake/accel), 1 for lane change
            maneuver_param: accel value for type 0, target lateral offset for type 1
        """
        
        # Simulate ego trajectory under this maneuver
        ego_positions, ego_velocities, ego_headings, ego_speeds = jax.lax.cond(
            maneuver_type == 0,
            lambda: simulate_longitudinal_maneuver(pos_E, vel_E, psi, lateral_offset_current, maneuver_param),
            lambda: simulate_lane_change_maneuver(pos_E, vel_E, psi, lateral_offset_current, maneuver_param)
        )
        
        # Predict neighbor trajectory
        neighbor_positions = predict_neighbor_trajectory(pos_N, vel_N)
        
        # Check collision at each timestep
        def check_collision(t_idx):
            rel_pos = neighbor_positions[t_idx] - ego_positions[t_idx]
            a, b = compute_axes(ego_speeds[t_idx], L, W)
            g_val = signed_ellipse_value(rel_pos, ego_headings[t_idx], a, b)
            return g_val
        
        g_values = jax.vmap(check_collision)(jnp.arange(NUM_PREDICTION_STEPS))
        
        min_g = jnp.min(g_values)
        collision_free = min_g >= 0.0
        
        # Time to first violation (if any)
        violations = g_values < 0.0
        time_to_violation = jnp.where(
            jnp.any(violations),
            jnp.argmax(violations) * datatypes.TIME_INTERVAL,
            jnp.inf
        )
        
        return collision_free, min_g, time_to_violation

    def check_lateral_feasibility(lateral_offset_current, target_lateral_offset):
        """Check if lane change is feasible given road boundaries."""
        within_bounds = jnp.abs(target_lateral_offset) <= MAX_LATERAL_DEVIATION
        not_current = jnp.abs(target_lateral_offset - lateral_offset_current) > 0.1
        return within_bounds & not_current

    def actor_init(rng, init_state):
        return {
            "reaction_timer": jnp.array(0, dtype=jnp.int32),
            "lateral_offset": jnp.array(0.0, dtype=jnp.float32)  # track current lateral position
        }

    def select_action(params, state: datatypes.SimulatorState, actor_state=None, rng=None):
        is_controlled = is_controlled_func(state)

        # Extract current states
        traj_t0 = datatypes.dynamic_index(state.sim_trajectory, state.timestep, axis=-1, keepdims=True)
        traj_prev = datatypes.dynamic_index(
            state.sim_trajectory,
            jnp.maximum(state.timestep - 1, 0),
            axis=-1,
            keepdims=True,
        )

        pos_E = jnp.array([traj_t0.x[av_idx, 0], traj_t0.y[av_idx, 0]])
        vel_E = jnp.array([traj_t0.vel_x[av_idx, 0], traj_t0.vel_y[av_idx, 0]])
        speed_E = jnp.linalg.norm(vel_E)
        psi = jnp.where(speed_E > 1e-3, jnp.arctan2(vel_E[1], vel_E[0]), 0.0)
        L = traj_t0.length[av_idx]
        W = traj_t0.width[av_idx]
        
        pos_N = jnp.array([traj_t0.x[neigh_idx, 0], traj_t0.y[neigh_idx, 0]])
        vel_N = jnp.array([traj_t0.vel_x[neigh_idx, 0], traj_t0.vel_y[neigh_idx, 0]])

        vel_E_prev = jnp.array([traj_prev.vel_x[av_idx, 0], traj_prev.vel_y[av_idx, 0]])
        acc_E_current = (jnp.linalg.norm(vel_E) - jnp.linalg.norm(vel_E_prev)) / datatypes.TIME_INTERVAL

        # Track lateral offset (in production, get from map)
        lateral_offset_current = actor_state.get("lateral_offset", 0.0)

        # === INTERVENTION DECISION ===
        # Assess baseline collision risk using ellipse constraint
        baseline_min_g = assess_baseline_risk(pos_E, pos_N, vel_E, vel_N, psi, L, W)
        
        # Intervention needed if ellipse constraint will be violated
        intervention_needed = baseline_min_g < SAFETY_MARGIN
        
        jax.debug.print('Baseline min ellipse value: {}', baseline_min_g)
        jax.debug.print('Intervention needed: {}', intervention_needed)

        def select_evasive_maneuver():
            """Hierarchical maneuver selection: prefer braking, then lane change."""
            
            # === PHASE 1: Try longitudinal control (braking) ===
            brake_candidates = jnp.linspace(-COMFORT_BRAKE, -MAX_BRAKE, NUM_BRAKE_CANDIDATES)
            
            def eval_brake(accel_cmd):
                safe, margin, ttv = evaluate_maneuver(
                    pos_E, pos_N, vel_E, vel_N, psi, L, W,
                    maneuver_type=0,
                    maneuver_param=accel_cmd,
                    lateral_offset_current=lateral_offset_current
                )
                return safe, margin, accel_cmd
            
            brake_results = jax.vmap(eval_brake)(brake_candidates)
            brake_safe, brake_margins, brake_accels = brake_results
            
            any_brake_safe = jnp.any(brake_safe)
            
            def choose_brake():
                # Select least invasive safe braking
                safe_indices = jnp.nonzero(brake_safe, size=NUM_BRAKE_CANDIDATES)[0]
                safe_margins = brake_margins[safe_indices]
                safe_accels = brake_accels[safe_indices]
                
                # Prefer less severe braking (closer to 0)
                invasiveness = jnp.abs(safe_accels)
                best_idx = jnp.argmin(invasiveness)
                
                return safe_accels[best_idx], lateral_offset_current  # no lane change
            
            # === PHASE 2: Try lane changes if braking insufficient ===
            def try_lane_changes():
                def eval_lane_change(target_offset):
                    feasible = check_lateral_feasibility(lateral_offset_current, target_offset)
                    
                    safe, margin, ttv = jax.lax.cond(
                        feasible,
                        lambda: evaluate_maneuver(
                            pos_E, pos_N, vel_E, vel_N, psi, L, W,
                            maneuver_type=1,
                            maneuver_param=target_offset,
                            lateral_offset_current=lateral_offset_current
                        ),
                        lambda: (False, -jnp.inf, 0.0)
                    )
                    
                    return safe & feasible, margin, target_offset
                
                lc_results = jax.vmap(eval_lane_change)(LATERAL_OFFSETS)
                lc_safe, lc_margins, lc_offsets = lc_results
                
                any_lc_safe = jnp.any(lc_safe)
                
                def choose_lane_change():
                    # Select lane change with best safety margin
                    safe_indices = jnp.nonzero(lc_safe, size=len(LATERAL_OFFSETS))[0]
                    safe_margins = lc_margins[safe_indices]
                    safe_offsets = lc_offsets[safe_indices]
                    
                    best_idx = jnp.argmax(safe_margins)
                    
                    # Combine with moderate braking during lane change
                    return -COMFORT_BRAKE, safe_offsets[best_idx]
                
                def emergency_brake():
                    return -MAX_BRAKE, lateral_offset_current
                
                return jax.lax.cond(any_lc_safe, choose_lane_change, emergency_brake)
            
            return jax.lax.cond(any_brake_safe, choose_brake, try_lane_changes)
        
        def maintain_behavior():
            # No intervention needed - maintain current state
            return jnp.clip(acc_E_current, -COMFORT_BRAKE, MAX_ACCEL), lateral_offset_current
        
        # Select action based on intervention need
        accel_cmd, target_lateral_offset = jax.lax.cond(
            intervention_needed,
            select_evasive_maneuver,
            maintain_behavior
        )

        # === EXECUTE SELECTED ACTION ===
        # Determine if we're in a lane change maneuver
        lane_changing = jnp.abs(target_lateral_offset - lateral_offset_current) > 0.1
        
        # Apply action for this timestep
        new_speed = jnp.maximum(speed_E + accel_cmd * datatypes.TIME_INTERVAL, 0.0)
        
        # Update heading and velocity based on lane change status
        def apply_lane_change():
            # Simplified: move laterally at constant rate
            lat_vel = (target_lateral_offset - lateral_offset_current) / LANE_CHANGE_DURATION
            lat_dir = jnp.array([-jnp.sin(psi), jnp.cos(psi)])
            long_dir = jnp.array([jnp.cos(psi), jnp.sin(psi)])
            return long_dir * new_speed + lat_dir * lat_vel
        
        def maintain_heading():
            direction = jnp.where(speed_E > 1e-3, vel_E / speed_E, jnp.array([1.0, 0.0]))
            return direction * new_speed
        
        new_vel = jax.lax.cond(lane_changing, apply_lane_change, maintain_heading)
        new_pos = pos_E + new_vel * datatypes.TIME_INTERVAL
        
        # Update lateral offset tracking
        new_lateral_offset = jnp.where(
            lane_changing,
            lateral_offset_current + (target_lateral_offset - lateral_offset_current) * datatypes.TIME_INTERVAL / LANE_CHANGE_DURATION,
            lateral_offset_current
        )

        # Update trajectory
        traj_t1 = traj_t0.replace(
            x=traj_t0.x.at[av_idx].set(new_pos[0]),
            y=traj_t0.y.at[av_idx].set(new_pos[1]),
            vel_x=traj_t0.vel_x.at[av_idx].set(new_vel[0]),
            vel_y=traj_t0.vel_y.at[av_idx].set(new_vel[1]),
            valid=is_controlled[..., None] & traj_t0.valid,
            timestamp_micros=traj_t0.timestamp_micros + datatypes.TIMESTEP_MICROS_INTERVAL,
        )

        traj_combined = jax.tree_util.tree_map(
            lambda x, y: jnp.concatenate([x, y], axis=-1), traj_t0, traj_t1
        )
        actions = dynamics_model.inverse(traj_combined, state.object_metadata, timestep=0)

        # Update actor state
        new_actor_state = {"lateral_offset": new_lateral_offset, "reaction_timer": jnp.array(0)}

        return actor_core.WaymaxActorOutput(
            actor_state=new_actor_state,
            action=actions,
            is_controlled=is_controlled,
        )

    return actor_core.actor_core_factory(
        init=actor_init,
        select_action=select_action,
        name="causal_ellipse_actor",
    )