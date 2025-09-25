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
    """Actor that uses an oriented safety ellipse to evaluate collision risk
    and select evasive longitudinal acceleration.
    """

    # Parameters for ellipse
    m_long = 2.0
    m_lat = 1.0
    kappa_a = 1.0   # time headway scaling (s)
    kappa_b = 0.05  # lateral margin scaling

    # Candidate accelerations to search
    NUM_CANDIDATES = 41
    MAX_BRAKE = 4.0
    MAX_ACCEL = 2.0

    def compute_axes(v_long: jax.Array, L: int, W: int):
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

    def actor_init(rng, init_state):
        return {"reaction_timer": jnp.array(0, dtype=jnp.int32)}

    def select_action(params, state: datatypes.SimulatorState, actor_state=None, rng=None):
        is_controlled = is_controlled_func(state)

        # Extract ego and neighbor states at t0 and t-1
        traj_t0 = datatypes.dynamic_index(state.sim_trajectory, state.timestep, axis=-1, keepdims=True)
        traj_prev = datatypes.dynamic_index(
            state.sim_trajectory,
            jnp.maximum(state.timestep - 1, 0),
            axis=-1,
            keepdims=True,
        )

        posE = jnp.array([traj_t0.x[av_idx, 0], traj_t0.y[av_idx, 0]])
        velE = jnp.array([traj_t0.vel_x[av_idx, 0], traj_t0.vel_y[av_idx, 0]])
        posN = jnp.array([traj_t0.x[neigh_idx, 0], traj_t0.y[neigh_idx, 0]])
        L = traj_t0.length[av_idx]
        W = traj_t0.width[av_idx]
        velE_prev = jnp.array([traj_prev.vel_x[av_idx, 0], traj_prev.vel_y[av_idx, 0]])
        speedE = jnp.linalg.norm(velE)
        accE = (jnp.linalg.norm(velE) - jnp.linalg.norm(velE_prev)) / datatypes.TIME_INTERVAL

        # Ego heading from velocity (fallback: 0 if very small speed)
        psi = jnp.where(speedE > 1e-3, jnp.arctan2(velE[1], velE[0]), 0.0)

        # Relative position
        rel_pos = posN - posE

        # Candidate accelerations
        candidates = jnp.linspace(MAX_ACCEL, -MAX_BRAKE, NUM_CANDIDATES)

        def eval_candidate(aF):
            # Predict new forward speed (no prediction of neighbor)
            v_long_after = jnp.maximum(speedE + aF * datatypes.TIME_INTERVAL, 0.0)
            a, b = compute_axes(v_long_after, L, W)
            g_val = signed_ellipse_value(rel_pos, psi, a, b)
            return g_val

        g_vals = jax.vmap(eval_candidate)(candidates)
        safe_mask = g_vals >= 0.0
        any_safe = jnp.any(safe_mask)

        def choose_safe():
            safe_indices = jnp.nonzero(safe_mask, size=NUM_CANDIDATES)[0]
            safe_candidates = candidates[safe_indices]
            diffs = jnp.abs(safe_candidates - accE)
            idx = jnp.argmin(diffs)
            return safe_candidates[idx]

        def choose_emergency():
            return -MAX_BRAKE

        accE_try = jax.lax.cond(any_safe, choose_safe, choose_emergency)

        # Update AV trajectory one step
        direction = jnp.where(speedE > 1e-3, velE / speedE, jnp.zeros_like(velE))
        new_speed = jnp.maximum(speedE + accE_try * datatypes.TIME_INTERVAL, 0.0)
        new_vel = direction * new_speed

        traj_t1 = traj_t0.replace(
            x=traj_t0.x.at[av_idx].set(posE[0] + new_vel[0] * datatypes.TIME_INTERVAL),
            y=traj_t0.y.at[av_idx].set(posE[1] + new_vel[1] * datatypes.TIME_INTERVAL),
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
        name="ellipse_actor",
    )

def causal_ellipse_actor(
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
        name="causal_ellipse_actor_with_steering",
    )

