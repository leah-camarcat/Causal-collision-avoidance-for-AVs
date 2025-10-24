"""
Causal CNN Actor: Collision avoidance using learned spatial risk model.

Replaces deterministic ellipse with learned CNN risk assessment while
maintaining the same counterfactual reasoning structure.
"""

from typing import Callable, Tuple
import jax
import jax.numpy as jnp
from waymax import datatypes, dynamics
from waymax.agents import actor_core
import pickle
from waymax.agents.causal_cnn.causal_cnn_model import CausalRiskCNN
from waymax.agents.causal_cnn.ground_truth_mttc import create_mttc_risk_grid

def causal_cnn_actor(
    dynamics_model: dynamics.DynamicsModel,
    is_controlled_func: Callable[[datatypes.SimulatorState], jax.Array],
    av_idx: int,
    scenario_id: str,
) -> actor_core.WaymaxActorCore:
    """
    Actor using CNN-based spatial risk assessment with counterfactual reasoning.
    
    Key features:
    1. Learned spatial risk field (replaces ellipse)
    2. Multi-agent awareness (no need for neigh_idx)
    3. Counterfactual evaluation of maneuvers
    4. Hierarchical action selection (brake → lane change → emergency)
    
    Args:
        dynamics_model: Waymax dynamics model
        is_controlled_func: Function determining which agents are controlled
        av_idx: Index of autonomous vehicle (ego)
    
    Note: Must call load_trained_risk_model() before using this actor!
    """

    # Longitudinal action space
    NUM_BRAKE_CANDIDATES = 15
    MAX_BRAKE = 4.0  # m/s²
    MAX_ACCEL = 2.0
    COMFORT_BRAKE = 3.0

    # Lateral action space
    LANE_WIDTH = 3.7  # meters
    LATERAL_OFFSETS = jnp.array([-LANE_WIDTH, 0.0, LANE_WIDTH])
    LANE_CHANGE_DURATION = 3.0  # seconds
    
    # Prediction parameters
    PREDICTION_HORIZON = 4.0  # seconds
    NUM_PREDICTION_STEPS = int(PREDICTION_HORIZON / datatypes.TIME_INTERVAL)
    
    # Intervention thresholds
    IMMINENT_COLLISION_TIME = 2.0  # seconds
    IMMINENT_STEPS = min(int(IMMINENT_COLLISION_TIME / datatypes.TIME_INTERVAL), NUM_PREDICTION_STEPS)
    RISK_THRESHOLD = 0.5  # CNN risk threshold for intervention (0-1 scale)
    
    # Road boundaries
    MAX_LATERAL_DEVIATION = 2.0 * LANE_WIDTH

    
    def load_model(path, rng=None):
        """
        Load model parameters and config from checkpoint.
        
        Args:
            path: checkpoint path (".pkl")
            rng: optional JAX PRNGKey for re-init (e.g. jax.random.PRNGKey(0))
        
        Returns:
            model: CausalRiskCNN instance with config restored
            params: trained parameters (frozen dict)
            losses: training loss curve (if available)
        """
        with open(path, "rb") as f:
            state = pickle.load(f)
        
        cfg = state["config"]
        model = CausalRiskCNN()
            #grid_size=cfg["grid_size"],
            #grid_range=cfg["grid_range"],
            #history_length=cfg["history_length"]
        #)
        
        params = state["params"]
        losses = state.get("training_losses", None)  # Updated key name
        
        #print(f"✓ Model loaded from {path}")
        #print(f"  Grid: {cfg['grid_size']} | Range: ±{cfg['grid_range']}m | "
        #    f"History: {cfg['history_length']} | Agents: {cfg.get('max_agents', 'N/A')}")
        
        return model, params, losses


    def load_trained_risk_model(model_path: str):
        """
        Load the trained CNN risk model (call once at initialization).
        Uses your custom load_model function.
        
        Args:
            model_path: Path to the trained model pickle file
        """
        _RISK_MODEL = None
        _RISK_PARAMS = None
        _MODEL_CONFIG = None
        
        #print(f"Loading trained risk model from: {model_path}")
        
        # Use your load_model function
        _RISK_MODEL, _RISK_PARAMS, losses = load_model(model_path)
        
        # Extract config from the loaded checkpoint
        with open(model_path, 'rb') as f:
            checkpoint = pickle.load(f)
        _MODEL_CONFIG = checkpoint['config']
        
        #print(f"✓ Model loaded successfully")
        #if losses is not None:
            #print(f"  Training samples: {len(losses)}")
            #print(f"  Final loss: {losses[-1]:.4f}" if len(losses) > 0 else "  No loss history")
        return _RISK_MODEL, _RISK_PARAMS, _MODEL_CONFIG


    def extract_multi_agent_observations(
        state: datatypes.SimulatorState,
        ego_idx: int,
        history_length: int = 10,
        max_agents: int = 8
    ) -> jnp.ndarray:
        """
        JAX-compatible multi-agent observation extraction.
        
        Returns:
            observations: (history_length, max_agents, 6) array
            For each agent: [rel_x, rel_y, rel_vx, rel_vy, speed, distance]
        """
        current_timestep = state.timestep
        num_objects = state.sim_trajectory.num_objects
        
        # Get current ego position
        ego_pos_current = state.sim_trajectory.xy[ego_idx, current_timestep]
        
        # Compute distances to all agents at current timestep
        all_positions = state.sim_trajectory.xy[:, current_timestep]  # (num_objects, 2)
        distances = jnp.linalg.norm(all_positions - ego_pos_current, axis=-1)  # (num_objects,)
        
        # Create validity mask: valid objects that aren't ego
        is_not_ego = jnp.arange(num_objects) != ego_idx
        is_valid = state.object_metadata.is_valid & is_not_ego
        
        # Set invalid/ego distances to infinity so they sort last
        distances = jnp.where(is_valid, distances, jnp.inf)
        
        # Get indices of K nearest agents
        sorted_indices = jnp.argsort(distances)
        nearest_indices = sorted_indices[:max_agents]  # (max_agents,)
        
        # Create mask for valid nearest agents (not padded with inf distance)
        nearest_valid = distances[nearest_indices] < jnp.inf  # (max_agents,)
        
        def extract_features_for_timestep(t):
            """Extract features for all agents at timestep t."""
            ego_pos = state.sim_trajectory.xy[ego_idx, t]
            ego_vel = jnp.stack([
                state.sim_trajectory.vel_x[ego_idx, t],
                state.sim_trajectory.vel_y[ego_idx, t]
            ])
            
            def extract_agent_features(agent_idx, is_valid_agent):
                """Extract features for a single agent."""
                # Get agent data
                agent_pos = state.sim_trajectory.xy[agent_idx, t]
                agent_vel = jnp.stack([
                    state.sim_trajectory.vel_x[agent_idx, t],
                    state.sim_trajectory.vel_y[agent_idx, t]
                ])
                
                # Compute relative features
                rel_pos = agent_pos - ego_pos
                rel_vel = agent_vel - ego_vel
                agent_speed = jnp.linalg.norm(agent_vel)
                distance = jnp.linalg.norm(rel_pos)
                
                features = jnp.array([
                    rel_pos[0],
                    rel_pos[1],
                    rel_vel[0],
                    rel_vel[1],
                    agent_speed,
                    distance
                ])
                
                # Return zeros if agent is invalid/padded
                return jnp.where(is_valid_agent, features, jnp.zeros(6))
            
            # Vectorize over all nearest agents
            agent_features = jax.vmap(extract_agent_features)(
                nearest_indices, 
                nearest_valid
            )  # (max_agents, 6)
            
            return agent_features
        
        # Create indices for history lookback (relative to current_timestep)
        # This avoids the traced value issue
        lookback_indices = jnp.arange(history_length)  # [0, 1, 2, ..., history_length-1]
        
        # Compute actual timesteps, clamping to valid range
        # t = current_timestep - (history_length - 1 - i)
        timesteps = current_timestep - (history_length - 1 - lookback_indices)
        timesteps = jnp.maximum(timesteps, 0)  # Clamp to 0 if negative
        
        # For padding: if timestep would be < 0, use timestep 0
        # This naturally handles the history padding
        
        # Vectorize over all timesteps
        observations = jax.vmap(extract_features_for_timestep)(timesteps)  # (history_length, max_agents, 6)
        
        return observations


    def assess_baseline_risk(
        state: datatypes.SimulatorState,
        ego_idx: int,
        return_grid: bool = False
        ) -> Tuple[float, jnp.ndarray]:
        """
        Assess baseline collision risk using trained Causal CNN.
        
        *** NOW MULTI-AGENT: No lead_idx needed! ***
        The model automatically considers all nearby vehicles.
        
        Args:
            state: Current simulator state
            ego_idx: Index of ego vehicle
            return_grid: If True, return full risk grid for visualization
        
        Returns:
            risk_score: Scalar risk value [0, 1] for ego vehicle's vicinity
            risk_grid: (grid_size, grid_size) spatial risk field (if return_grid=True)
        """
        _RISK_MODEL, _RISK_PARAMS, _MODEL_CONFIG = load_trained_risk_model('waymax/agents/causal_cnn/trained_risk_model_v3.pkl')
        
        # Check if model is loaded
        if _RISK_MODEL is None or _RISK_PARAMS is None:
            raise RuntimeError(
                "Risk model not loaded! Call load_trained_risk_model() first."
            )
        
        # Extract MULTI-AGENT observation history
        observations = extract_multi_agent_observations(
            state,
            ego_idx,
            _MODEL_CONFIG['history_length'],
            _MODEL_CONFIG.get('max_agents', 8)
        )
        
        # Add batch dimension
        observations = observations[None, ...]
        
        # Run inference
        risk_grid, attention_maps = _RISK_MODEL.apply(
            _RISK_PARAMS,
            observations,
            training=False
        )
        
        # Extract risk score for ego vehicle vicinity
        grid_size = _MODEL_CONFIG['grid_size']
        center = grid_size // 2
        vicinity_size = 5  # 5x5 cells around ego
        
        # Extract risk in vicinity
        risk_vicinity = risk_grid[
            0,
            center - vicinity_size : center + vicinity_size,
            center - vicinity_size : center + vicinity_size,
            0
        ]
        
        # Compute risk score (max for conservative estimate)
        risk_score = jnp.max(risk_vicinity)
        
        if return_grid:
            return risk_score, risk_grid[0, :, :, 0]
        else:
            return risk_score, None


    def assess_baseline_risk_gt(state: datatypes.SimulatorState,
        ego_idx: int,
        scenario_id: str,
        return_grid: bool = False
        ) -> Tuple[float, jnp.ndarray]:

        risk_grid = create_mttc_risk_grid(
            state, ego_idx, state.timestep, 
            'docs/cutin_filtered_data_modified/', 
            scenario_id
        )
        risk_score = jnp.maximum(risk_grid)
        return risk_score

    def simulate_ego_state_forward(pos, vel, psi, speed, accel_long, lateral_offset, 
                                   target_lateral_offset, num_steps):
        """
        Simulate ego vehicle forward in time.
        
        Args:
            pos: Current position [x, y]
            vel: Current velocity [vx, vy]
            psi: Current heading (radians)
            speed: Current speed (m/s)
            accel_long: Longitudinal acceleration command
            lateral_offset: Current lateral offset from centerline
            target_lateral_offset: Target lateral offset (for lane changes)
            num_steps: Number of prediction steps
        
        Returns:
            Trajectory of (positions, velocities, headings, speeds)
        """
        lane_changing = jnp.abs(target_lateral_offset - lateral_offset) > 0.1
        lane_change_steps = int(LANE_CHANGE_DURATION / datatypes.TIME_INTERVAL)
        
        def step(carry, t_idx):
            pos_curr, vel_curr, speed_curr, psi_curr, lat_offset_curr = carry
            
            # Update speed with longitudinal acceleration
            new_speed = jnp.maximum(speed_curr + accel_long * datatypes.TIME_INTERVAL, 0.0)
            
            # Handle lateral motion if lane changing
            def apply_lane_change():
                # Smooth lateral transition
                progress = jnp.minimum(t_idx / lane_change_steps, 1.0)
                smooth_progress = 3 * progress**2 - 2 * progress**3
                
                # Lateral velocity needed
                lat_vel = (target_lateral_offset - lateral_offset) / LANE_CHANGE_DURATION
                
                # Combined velocity
                long_dir = jnp.array([jnp.cos(psi_curr), jnp.sin(psi_curr)])
                lat_dir = jnp.array([-jnp.sin(psi_curr), jnp.cos(psi_curr)])
                combined_vel = long_dir * new_speed + lat_dir * lat_vel
                
                # Update heading
                new_psi = psi_curr + jnp.arctan2(lat_vel, jnp.maximum(new_speed, 0.1))
                
                return combined_vel, new_psi
            
            def maintain_heading():
                direction = jnp.array([jnp.cos(psi_curr), jnp.sin(psi_curr)])
                return direction * new_speed, psi_curr
            
            new_vel, new_psi = jax.lax.cond(
                lane_changing & (t_idx < lane_change_steps),
                apply_lane_change,
                maintain_heading
            )
            
            # Update position
            new_pos = pos_curr + new_vel * datatypes.TIME_INTERVAL
            
            return (new_pos, new_vel, new_speed, new_psi, lat_offset_curr), \
                   (new_pos, new_vel, new_psi, new_speed)
        
        _, trajectory = jax.lax.scan(
            step,
            (pos, vel, speed, psi, lateral_offset),
            jnp.arange(num_steps)
        )
        
        return trajectory  # (positions, velocities, headings, speeds)

    def create_counterfactual_state(base_state, ego_trajectory, timestep_idx):
        """
        Create counterfactual state by updating ego position/velocity.
        
        Args:
            base_state: Current simulator state
            ego_trajectory: Predicted ego trajectory (positions, velocities, headings, speeds)
            timestep_idx: Which timestep to extract
        
        Returns:
            Modified state with ego at predicted position/velocity
        """
        positions, velocities, headings, speeds = ego_trajectory
        
        # Extract predicted ego state at this timestep
        pred_pos = positions[timestep_idx]
        pred_vel = velocities[timestep_idx]
        
        # Create modified trajectory
        modified_traj = base_state.sim_trajectory.replace(
            x=base_state.sim_trajectory.x.at[av_idx, base_state.timestep].set(pred_pos[0]),
            y=base_state.sim_trajectory.y.at[av_idx, base_state.timestep].set(pred_pos[1]),
            vel_x=base_state.sim_trajectory.vel_x.at[av_idx, base_state.timestep].set(pred_vel[0]),
            vel_y=base_state.sim_trajectory.vel_y.at[av_idx, base_state.timestep].set(pred_vel[1])
        )
        
        # Return modified state
        return base_state.replace(sim_trajectory=modified_traj)

    def evaluate_maneuver_cnn(state, pos_E, vel_E, psi, speed_E, 
                             accel_long, lateral_offset_current, target_lateral_offset):
        """
        Counterfactual evaluation: What if ego applies this maneuver?
        
        Uses CNN to assess risk at future timesteps under counterfactual trajectory.
        
        Args:
            state: Current simulator state
            pos_E, vel_E, psi, speed_E: Current ego state
            accel_long: Longitudinal acceleration command
            lateral_offset_current: Current lateral position
            target_lateral_offset: Target lateral position
        
        Returns:
            collision_free: Boolean, whether maneuver avoids collision
            max_risk: Maximum risk encountered along trajectory
            avg_risk: Average risk along trajectory
        """
        # Simulate ego trajectory under this maneuver
        ego_trajectory = simulate_ego_state_forward(
            pos_E, vel_E, psi, speed_E, accel_long, 
            lateral_offset_current, target_lateral_offset, NUM_PREDICTION_STEPS
        )
        
        positions, velocities, headings, speeds = ego_trajectory
        
        # Evaluate CNN risk at each future timestep
        def eval_risk_at_step(t_idx):
            # Create counterfactual state with ego at predicted position
            cf_state = create_counterfactual_state(state, ego_trajectory, t_idx)
            
            # Assess risk using CNN
            risk, _ = assess_baseline_risk(cf_state, av_idx)
            return risk
        
        # Check risk over prediction horizon
        risk_trajectory = jax.vmap(eval_risk_at_step)(jnp.arange(NUM_PREDICTION_STEPS))
        
        # Aggregate risk metrics
        max_risk = jnp.max(risk_trajectory)
        avg_risk = jnp.mean(risk_trajectory)
        
        # Consider collision-free if max risk stays below threshold
        collision_free = max_risk < RISK_THRESHOLD
        
        return collision_free, max_risk, avg_risk

    def check_lateral_feasibility(lateral_offset_current, target_lateral_offset):
        """Check if lane change is feasible given road boundaries."""
        within_bounds = jnp.abs(target_lateral_offset) <= MAX_LATERAL_DEVIATION
        not_current = jnp.abs(target_lateral_offset - lateral_offset_current) > 0.1
        return within_bounds & not_current

    def actor_init(rng, init_state):
        return {
            "lateral_offset": jnp.array(0.0, dtype=jnp.float32),
            "reaction_timer": jnp.array(0.0, dtype=jnp.float32),
            "has_reacted": jnp.array(False)
        }

    def select_action(params, state: datatypes.SimulatorState, actor_state=None, rng=None):
        is_controlled = is_controlled_func(state)

        REACTION_STEPS = int(0.25 / datatypes.TIME_INTERVAL)
        reaction_timer = actor_state["reaction_timer"]
        has_reacted = actor_state["has_reacted"]
        
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

        vel_E_prev = jnp.array([traj_prev.vel_x[av_idx, 0], traj_prev.vel_y[av_idx, 0]])
        acc_E_current = (jnp.linalg.norm(vel_E) - jnp.linalg.norm(vel_E_prev)) / datatypes.TIME_INTERVAL

        # Track lateral offset
        lateral_offset_current = actor_state.get("lateral_offset", 0.0) if actor_state else 0.0

        # === INTERVENTION DECISION ===
        # Assess baseline risk using CNN
        baseline_risk, _ = assess_baseline_risk(state, av_idx, return_grid=False)

        # Intervention needed if risk exceeds threshold
        intervention_needed = baseline_risk > RISK_THRESHOLD
        can_start_reaction = (~has_reacted) & (reaction_timer == 0) & intervention_needed
        jax.debug.print('CNN Baseline Risk: {}', baseline_risk)
        jax.debug.print('Intervention Needed: {}', intervention_needed)

        def select_evasive_maneuver():
            """
            Hierarchical maneuver selection using CNN risk assessment.
            
            1. Try pure braking (maintain lane)
            2. If insufficient, try lane changes
            3. Emergency brake as last resort
            """
            
            # === PHASE 1: Evaluate Braking Maneuvers ===
            brake_candidates = jnp.linspace(-COMFORT_BRAKE, -MAX_BRAKE, NUM_BRAKE_CANDIDATES)
            
            def eval_brake(accel_cmd):
                safe, max_risk, avg_risk = evaluate_maneuver_cnn(
                    state, pos_E, vel_E, psi, speed_E,
                    accel_long=accel_cmd,
                    lateral_offset_current=lateral_offset_current,
                    target_lateral_offset=lateral_offset_current  # no lane change
                )
                return safe, max_risk, accel_cmd
            
            brake_results = jax.vmap(eval_brake)(brake_candidates)
            brake_safe, brake_risks, brake_accels = brake_results
            
            any_brake_safe = jnp.any(brake_safe)
            
            def choose_brake():
                # Select least invasive safe braking
                safe_indices = jnp.nonzero(brake_safe, size=NUM_BRAKE_CANDIDATES)[0]
                safe_risks = brake_risks[safe_indices]
                safe_accels = brake_accels[safe_indices]
                
                # Prefer less severe braking with lowest risk
                # Combine risk (lower better) and invasiveness (closer to 0 better)
                score = safe_risks + 0.1 * jnp.abs(safe_accels)
                best_idx = jnp.argmin(score)
                
                return safe_accels[best_idx], lateral_offset_current
            
            # === PHASE 2: Evaluate Lane Changes ===
            def try_lane_changes():
                def eval_lane_change(target_offset):
                    feasible = check_lateral_feasibility(lateral_offset_current, target_offset)
                    
                    safe, max_risk, avg_risk = jax.lax.cond(
                        feasible,
                        lambda: evaluate_maneuver_cnn(
                            state, pos_E, vel_E, psi, speed_E,
                            accel_long=-COMFORT_BRAKE,  # brake during lane change
                            lateral_offset_current=lateral_offset_current,
                            target_lateral_offset=target_offset
                        ),
                        lambda: (False, 1.0, 1.0)  # infeasible = max risk
                    )
                    
                    return safe & feasible, max_risk, target_offset
                
                lc_results = jax.vmap(eval_lane_change)(LATERAL_OFFSETS)
                lc_safe, lc_risks, lc_offsets = lc_results
                
                any_lc_safe = jnp.any(lc_safe)
                
                def choose_lane_change():
                    # Select lane change with lowest risk
                    safe_indices = jnp.nonzero(lc_safe, size=len(LATERAL_OFFSETS))[0]
                    safe_risks = lc_risks[safe_indices]
                    safe_offsets = lc_offsets[safe_indices]
                    
                    best_idx = jnp.argmin(safe_risks)
                    
                    return -COMFORT_BRAKE, safe_offsets[best_idx]
                
                def emergency_brake():
                    # No safe maneuver found - maximum braking
                    return -MAX_BRAKE, lateral_offset_current
                
                return jax.lax.cond(any_lc_safe, choose_lane_change, emergency_brake)
            
            return jax.lax.cond(any_brake_safe, choose_brake, try_lane_changes)
        
        def maintain_behavior():
            # No intervention needed - continue current behavior
            return jnp.clip(acc_E_current, -COMFORT_BRAKE, MAX_ACCEL), lateral_offset_current
        
        # Select action based on CNN risk assessment
        accel_cmd, target_lateral_offset = jax.lax.cond(
            can_start_reaction,
            select_evasive_maneuver,
            maintain_behavior
        )

        # === EXECUTE SELECTED ACTION ===
        lane_changing = jnp.abs(target_lateral_offset - lateral_offset_current) > 0.1
        
        # Apply action for this timestep
        new_speed = jnp.maximum(speed_E + accel_cmd * datatypes.TIME_INTERVAL, 0.0)
        
        def apply_lane_change():
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
        new_actor_state = {**actor_state,
                           "lateral_offset": new_lateral_offset,
                            "reaction_timer": reaction_timer,
                           "has_reacted": has_reacted}

        return actor_core.WaymaxActorOutput(
            actor_state=new_actor_state,
            action=actions,
            is_controlled=is_controlled,
        )

    return actor_core.actor_core_factory(
        init=actor_init,
        select_action=select_action,
        name="causal_cnn_actor",
    )