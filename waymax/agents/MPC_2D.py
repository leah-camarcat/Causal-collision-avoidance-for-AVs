import jax
import jax.numpy as jnp
from waymax import datatypes, dynamics
from waymax.agents import actor_core
from functools import partial
from jaxopt import OSQP

def MPC_2D_actor(
    dynamics_model: dynamics.DynamicsModel,
    is_controlled_func,
    av_idx: int,
    obs_idx: int, 
    horizon: int = 10,
    dt: float = 0.1,
) -> actor_core.WaymaxActorCore:
    """2D MPC Actor with longitudinal and lateral collision avoidance."""
    
    def actor_init(rng, init_state):
        return {
            "reaction_timer": jnp.array(0, dtype=jnp.int32),
            "has_reacted": jnp.array(False),
            "prev_accel_x": jnp.array(0.0, dtype=jnp.float32),
            "prev_accel_y": jnp.array(0.0, dtype=jnp.float32),
        }

    def select_action(params, 
                      state: datatypes.SimulatorState, 
                      actor_state=None, 
                      rng=None
    ) -> actor_core.WaymaxActorOutput:
        
        is_controlled = is_controlled_func(state)
        
        # Extract current trajectory data
        traj_t0 = datatypes.dynamic_index(
            state.sim_trajectory, state.timestep, axis=-1, keepdims=True
        )

        jax.debug.print("timestep={}", state.timestep)

        def get_prev(_):
            return datatypes.dynamic_index(state.sim_trajectory, state.timestep - 1, axis=-1, keepdims=True)
        
        def get_same(_):
            return traj_t0
        
        traj_prev = jax.lax.cond(
            state.timestep > 0, get_prev, get_same, operand=None,
        )

        # AV state extraction
        av_pos = jnp.array([traj_t0.x[av_idx, 0], traj_t0.y[av_idx, 0]])
        av_vel = jnp.array([traj_t0.vel_x[av_idx, 0], traj_t0.vel_y[av_idx, 0]])
        av_vel_prev = jnp.array([traj_prev.vel_x[av_idx, 0], traj_prev.vel_y[av_idx, 0]])
        speed_av = jnp.linalg.norm(av_vel)
        speed_av_prev = jnp.linalg.norm(av_vel_prev)
        acc_av = (speed_av - speed_av_prev) / dt
        # Current acceleration (from velocity difference)
        av_accel = (av_vel - av_vel_prev) / dt

        av_length = traj_t0.length[av_idx, 0]
        av_width = traj_t0.width[av_idx, 0]
        
        # Extract obstacle data
        obs_pos = jnp.array([traj_t0.x[obs_idx, 0], traj_t0.y[obs_idx, 0]])
        obs_vel = jnp.array([traj_t0.vel_x[obs_idx, 0], traj_t0.vel_y[obs_idx, 0]])
        obs_vel_prev = jnp.array([traj_prev.vel_x[obs_idx, 0], traj_prev.vel_y[obs_idx, 0]])
        obs_accel = (obs_vel - obs_vel_prev) / dt
        obs_length = traj_t0.length[obs_idx, 0]
        obs_width = traj_t0.width[obs_idx, 0]
        
        def make_difference_matrix(N):
            """Build difference matrix for jerk regularization."""
            J = jnp.zeros((N, N), dtype=jnp.float32)
            J = J.at[0, 0].set(1.0)
            for k in range(1, N):
                J = J.at[k, k].set(1.0)
                J = J.at[k, k-1].set(-1.0)
            return J

        def build_2d_qp_matrices_fixed(
            N, dt, av_pos, av_vel, av_accel_prev, obs_pos, obs_vel, 
            obs_length, obs_width, av_length, av_width,
            beta=1.0, gamma=10.0, alpha_slack=1e4, a_max=3.0, safety_margin=2.0,
        ):
            """Build QP matrices with fixed JAX-compatible structure."""
            
            # Ensure all inputs are JAX arrays
            dt = jnp.asarray(dt, dtype=jnp.float32)
            av_pos = jnp.asarray(av_pos, dtype=jnp.float32)
            av_vel = jnp.asarray(av_vel, dtype=jnp.float32) 
            av_accel_prev = jnp.asarray(av_accel_prev, dtype=jnp.float32)
            obs_pos = jnp.asarray(obs_pos, dtype=jnp.float32)
            obs_vel = jnp.asarray(obs_vel, dtype=jnp.float32)
            
            n_accel_vars = 2 * N
            n_collision_constraints = N
            n_slack_vars = n_collision_constraints
            Z = n_accel_vars + n_slack_vars
            
            # Build Q matrix
            Q = jnp.zeros((Z, Z), dtype=jnp.float32)
            
            # Acceleration cost
            for k in range(N):
                ax_idx = 2 * k
                ay_idx = 2 * k + 1
                Q = Q.at[ax_idx, ax_idx].set(2.0 * beta)
                Q = Q.at[ay_idx, ay_idx].set(2.0 * beta)
            
            # Jerk regularization
            J = make_difference_matrix(N)
            JTJ = J.T @ J 

            for i in range(N):
                for j in range(N):
                    ax_i_idx = 2 * i
                    ax_j_idx = 2 * j
                    ay_i_idx = 2 * i + 1
                    ay_j_idx = 2 * j + 1
                    Q = Q.at[ax_i_idx, ax_j_idx].add(2.0 * gamma * JTJ[i, j])
                    Q = Q.at[ay_i_idx, ay_j_idx].add(2.0 * gamma * JTJ[i, j])
            
            # Slack penalty
            for s in range(n_slack_vars):
                s_idx = n_accel_vars + s
                Q = Q.at[s_idx, s_idx].set(2.0 * alpha_slack)
            
            # Linear cost
            c = jnp.zeros((Z,), dtype=jnp.float32)
            b_x = jnp.zeros((N,), dtype=jnp.float32).at[0].set(av_accel_prev[0])
            b_y = jnp.zeros((N,), dtype=jnp.float32).at[0].set(av_accel_prev[1])
            
            jerk_linear_x = -2.0 * gamma * (J.T @ b_x)
            jerk_linear_y = -2.0 * gamma * (J.T @ b_y)
            
            for k in range(N):
                c = c.at[2 * k].set(jerk_linear_x[k])
                c = c.at[2 * k + 1].set(jerk_linear_y[k])
            
            # Pre-allocate constraint matrices with fixed size
            # We know we'll have: 4*N (box constraints) + N (collision) + N (slack >= 0)
            max_constraints = 4*N + N + N
            G = jnp.zeros((max_constraints, Z), dtype=jnp.float32)
            h = jnp.zeros((max_constraints,), dtype=jnp.float32)
            
            constraint_idx = 0
            
            # Box constraints on acceleration
            for k in range(N):
                # ax_k <= a_max
                G = G.at[constraint_idx, 2*k].set(1.0)
                h = h.at[constraint_idx].set(a_max)
                constraint_idx += 1
                
                # -ax_k <= a_max
                G = G.at[constraint_idx, 2*k].set(-1.0)
                h = h.at[constraint_idx].set(a_max)
                constraint_idx += 1
                
                # ay_k <= a_max
                G = G.at[constraint_idx, 2*k+1].set(1.0)
                h = h.at[constraint_idx].set(a_max)
                constraint_idx += 1
                
                # -ay_k <= a_max
                G = G.at[constraint_idx, 2*k+1].set(-1.0)
                h = h.at[constraint_idx].set(a_max)
                constraint_idx += 1
            
            # Collision avoidance constraints
            safe_dist = (av_length + obs_length) / 2.0 + safety_margin
            
            for k in range(N):
                # Predicted positions
                av_pred_pos = av_pos + av_vel * (k + 1) * dt
                obs_pred_pos = obs_pos + obs_vel * (k + 1) * dt
                rel_pos = av_pred_pos - obs_pred_pos
                rel_dist = jnp.linalg.norm(rel_pos) + 1e-8  # avoid division by zero
                
                # Unit vector
                unit_vec = rel_pos / rel_dist
                
                # Build constraint row
                for i in range(k + 1):
                    coeff = dt * dt * (k - i + 0.5)
                    G = G.at[constraint_idx, 2*i].set(-coeff * unit_vec[0])
                    G = G.at[constraint_idx, 2*i+1].set(-coeff * unit_vec[1])
                
                # Slack variable
                G = G.at[constraint_idx, n_accel_vars + k].set(-1.0)
                h = h.at[constraint_idx].set(rel_dist - safe_dist)
                constraint_idx += 1
            
            # Slack >= 0 constraints
            for s in range(n_slack_vars):
                G = G.at[constraint_idx, n_accel_vars + s].set(-1.0)
                h = h.at[constraint_idx].set(0.0)
                constraint_idx += 1
            
            # Return only the used part of matrices
            G_final = G[:constraint_idx]
            h_final = h[:constraint_idx]
            
            # numerical regularizer to ensure positive definiteness / conditioning
            eps = jnp.array(1e-6, dtype=jnp.float32)
            Q = Q + eps * jnp.eye(Z, dtype=jnp.float32)
            
            return Q, c, G_final, h_final

        def solve_2d_mpc_simple(av_pos, av_vel, av_accel_prev, obs_pos, obs_vel, 
                               obs_length, obs_width, av_length, av_width):
            """Simplified MPC solver."""
            
            N = horizon
            Q, c, G, h = build_2d_qp_matrices_fixed(
                N, dt, av_pos, av_vel, av_accel_prev, obs_pos, obs_vel,
                obs_length, obs_width, av_length, av_width
            )
            
            solver = OSQP()
            z, state  = solver.run(params_obj=(Q, c), params_ineq=(G, h))
            
            # Extract solution
            n_accel_vars = 2 * N
            accel_plan = z.primal[:n_accel_vars] if hasattr(z, "primal") else z[:n_accel_vars]
            slack_plan = z.primal[n_accel_vars:] if hasattr(z, "primal") else z[n_accel_vars:]
            
            # Return first acceleration command
            a0 = jnp.array(accel_plan[:2]) # (2,) array: [ax_0, ay_0]
            
            info = {
                "accel_plan": accel_plan,
                "slack_plan": slack_plan, 
                "status": state,  # KKTSolution doesn't have a status field typically
            }
            
            return a0, info

        # Initialize actor state if needed
        if actor_state is None:
            actor_state = {
                "reaction_timer": 0,
                "has_reacted": jnp.array(False),
                "prev_accel_x": 0.0,
                "prev_accel_y": 0.0,
            }

        # Reaction logic
        SUDDEN_BRAKE_THRESHOLD = -2.0
        REACTION_STEPS = int(0.25 / dt)
        
        reaction_timer = actor_state["reaction_timer"] 
        has_reacted = actor_state["has_reacted"]
        
        obstacle_decel = -jnp.linalg.norm(obs_accel)
        sudden_brake_detected = obstacle_decel < SUDDEN_BRAKE_THRESHOLD
        
        start_reaction = (~has_reacted) & (reaction_timer == 0) & sudden_brake_detected
        
        reaction_timer = jnp.where(
            start_reaction,
            jnp.array(REACTION_STEPS, dtype=jnp.int32),
            jnp.where(reaction_timer > 0, reaction_timer - 1, jnp.array(0, dtype=jnp.int32)),
        )
        has_reacted = jnp.where(start_reaction, True, has_reacted)
        
        new_actor_state = {
            **actor_state, 
            "reaction_timer": reaction_timer,
            "has_reacted": has_reacted,
            "prev_accel_x": av_accel[0],
            "prev_accel_y": av_accel[1],
        }
        jax.debug.print("speed_av: {}", speed_av)
        jax.debug.print("acc_av: {}", acc_av)
        jax.debug.print("time interval: {}", datatypes.TIME_INTERVAL)

        def during_reaction(_):
            """During reaction - maintain current motion."""
            direction_av = jnp.where(speed_av > 1e-3, av_vel / speed_av, jnp.zeros_like(av_vel))
            new_speed_av = jnp.maximum(speed_av + acc_av * datatypes.TIME_INTERVAL, 0.0)
            new_velocity_av = direction_av * new_speed_av
            jax.debug.print("new velocity during reaction: {}", new_velocity_av)
            return new_velocity_av
        
        def after_reaction(_):
            """Use MPC after reaction."""
            # Get 2D acceleration from MPC
            a0_2d, info = solve_2d_mpc_simple(
                av_pos, av_vel, av_accel, 
                obs_pos, obs_vel, obs_length, obs_width, av_length, av_width
            )
            
            # Convert to velocity update (using magnitude for compatibility)
            accel_magnitude = jnp.linalg.norm(a0_2d)
            direction_av = jnp.where(speed_av > 1e-3, av_vel / speed_av, jnp.array([1.0, 0.0]))
            new_speed_av = jnp.maximum(speed_av + accel_magnitude * datatypes.TIME_INTERVAL, 0.0)
            new_velocity_av = direction_av * new_speed_av
            return new_velocity_av
        
        new_velocity_av = jax.lax.cond(reaction_timer > 0, during_reaction, after_reaction, operand=None)
        jax.debug.print("reaction timer: {}", reaction_timer)
        jax.debug.print("new velocity: {}", new_velocity_av)
        
        traj_t1 = traj_t0.replace(
            x=traj_t0.x.at[av_idx].set(av_pos[0] + new_velocity_av[0] * datatypes.TIME_INTERVAL),
            y=traj_t0.y.at[av_idx].set(av_pos[1] + new_velocity_av[1] * datatypes.TIME_INTERVAL),
            vel_x=traj_t0.vel_x.at[av_idx].set(new_velocity_av[0]),
            vel_y=traj_t0.vel_y.at[av_idx].set(new_velocity_av[1]),
            valid=is_controlled[..., None] & traj_t0.valid,
            timestamp_micros=traj_t0.timestamp_micros + datatypes.TIMESTEP_MICROS_INTERVAL,
        )

        traj_combined = jax.tree_util.tree_map(
            lambda x, y: jnp.concatenate([x, y], axis=-1), traj_t0, traj_t1
        )
        
        actions = dynamics_model.inverse(traj_combined, state.object_metadata, timestep=0)

        return actor_core.WaymaxActorOutput(
            actor_state=new_actor_state,
            action=actions,
            is_controlled=is_controlled
        )

    return actor_core.actor_core_factory(
        init=actor_init,
        select_action=select_action,
        name="MPC_2D_actor"
    )

