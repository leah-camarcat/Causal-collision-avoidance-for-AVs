import jax
import jax.numpy as jnp
from waymax import datatypes, dynamics
from waymax.agents import actor_core
from functools import partial
from jaxopt import OSQP


def MPC_actor(
    dynamics_model: dynamics.DynamicsModel,
    is_controlled_func,
    av_idx: int,
    lead_idx: int,
    horizon: int = 10,
    accel_candidates=jnp.linspace(-3.0, 2.0, 7),  # candidate accelerations [m/s^2]
) -> actor_core.WaymaxActorCore:
    """MPC Actor (longitudinal only) using Waymax dynamic_index format."""
    
    def actor_init(rng, init_state):
        return {"reaction_timer": jnp.array(0, dtype=jnp.int32),
                "has_reacted": jnp.array(False)}

    def select_action(params, 
                      state: datatypes.SimulatorState, 
                      actor_state=None, 
                      rng=None
    ) -> actor_core.WaymaxActorOutput:
        # --- 1. Extract current AV + lead state ---
        is_controlled = is_controlled_func(state)
        
        jax.debug.print("timestep={}", state.timestep)

        traj_t0 = datatypes.dynamic_index(
            state.sim_trajectory, state.timestep, axis=-1, keepdims=True
        )
        def get_prev(_):
            return datatypes.dynamic_index(state.sim_trajectory, state.timestep - 1, axis=-1, keepdims=True)

        def get_same(_):
            return traj_t0
        
        traj_prev = jax.lax.cond(
            state.timestep > 0,
            get_prev,
            get_same,
            operand=None,
        )

        # AV variables
        av_x = traj_t0.x[av_idx, 0]
        av_y = traj_t0.y[av_idx, 0]
        av_vx = traj_t0.vel_x[av_idx, 0]
        av_vy = traj_t0.vel_y[av_idx, 0]

        # Lead vehicle variables
        lead_x = traj_t0.x[lead_idx, 0]
        lead_y = traj_t0.y[lead_idx, 0]
        lead_vx = traj_t0.vel_x[lead_idx, 0]
        lead_vy = traj_t0.vel_y[lead_idx, 0]

        pos_av_t0 = jnp.array([traj_t0.x[av_idx, 0], traj_t0.y[av_idx, 0]])
        vel_av_t0 = jnp.array([traj_t0.vel_x[av_idx, 0], traj_t0.vel_y[av_idx, 0]])
        vel_av_prev = jnp.array([traj_prev.vel_x[av_idx, 0], traj_prev.vel_y[av_idx, 0]])
        
        speed_av_t0 = jnp.linalg.norm(vel_av_t0)
        speed_av_prev = jnp.linalg.norm(vel_av_prev)
        acc_av = (speed_av_t0 - speed_av_prev) / 0.1

        pos_lead_t0 = jnp.array([traj_t0.x[lead_idx, 0], traj_t0.y[lead_idx, 0]])
        vel_lead_t0 = jnp.array([traj_t0.vel_x[lead_idx, 0], traj_t0.vel_y[lead_idx, 0]])
        vel_lead_prev = jnp.array([traj_prev.vel_x[lead_idx, 0], traj_prev.vel_y[lead_idx, 0]])
        
        speed_lead_t0 = jnp.linalg.norm(vel_lead_t0)
        speed_lead_prev = jnp.linalg.norm(vel_lead_prev)
        acc_lead = (speed_lead_t0 - speed_lead_prev) / 0.1
        # Current gap (Euclidean distance)
        gap = jnp.sqrt((lead_x - av_x)**2 + (lead_y - av_y)**2)
        rel_speed = jnp.sqrt(lead_vx**2 + lead_vy**2) - jnp.sqrt(av_vx**2 + av_vy**2)

        
        def make_difference_matrix(N: int):
            """
            Build J such that jerk = J @ a - b, where:
            jerk_0 = a0 - a_prev
            jerk_k = a_k - a_{k-1} for k>=1

            J is N x N. For N=3:
            [[1, 0, 0],
            [-1,1,0],
            [0,-1,1]]  but more convenient to make rows:
            row0: [1, 0, 0]
            row1: [-1,1,0]
            row2: [0,-1,1]
            We'll build J such that J @ a gives the vector of differences with sign consistent with above.
            """
            J = jnp.zeros((N, N))
            J = J.at[0, 0].set(1.0)
            for k in range(1, N):
                J = J.at[k, k].set(1.0)
                J = J.at[k, k-1].set(-1.0)
            return J


        def build_qp_matrices_old(
            N: int,
            dt: float,
            v0: float,
            obs_v: float,
            a_prev: float,
            a_min: float,
            a_max: float,
            alpha: float,
            beta: float,
            gamma: float,
        ):
            """
            Build matrices (P, q, A, l, u) for the QP:
            min_{z=[a; s]} 1/2 z^T P z + q^T z
            subject to:
                for each k: s_k + dt * sum_{i=0}^{k} a_i >= obs_v - v0
                s_k >= 0
                a_min <= a_i <= a_max
            We implement constraints by stacking linear rows and converting them to
            l <= A z <= u
            """

            # variable ordering: z = [a(0..N-1), s(0..N-1)] length 2N
            Z = 2 * N

            # ---- build P and q ----
            # cost: alpha * sum s_k^2  + beta * sum a_k^2  + gamma * ||J a - b||^2
            # => P has blocks: P_aa = 2*(beta*I + gamma*J^T J), P_ss = 2*alpha*I, cross terms zero
            I_N = jnp.eye(N)

            # J matrix for jerk
            J = make_difference_matrix(N)  # shape N x N
            b = jnp.zeros((N,))
            b = b.at[0].set(a_prev)  # because jerk_0 = a0 - a_prev -> (J a - b)

            # Contribution from beta * a^T a => (1/2) z^T P z yields beta*a^T a,
            # so P entries must be 2*beta on the diagonal for 'a' part.
            P_aa = 2.0 * beta * I_N

            # Jerk quadratic: gamma * (a^T J^T J a - 2 b^T J a + b^T b)
            P_aa = P_aa + 2.0 * gamma * (J.T @ J)  # multiply by 2 because QP uses 1/2 z^T P z

            # P for s (alpha * sum s^2)
            P_ss = 2.0 * alpha * I_N

            # combine P
            P = jnp.block([
                [P_aa, jnp.zeros((N, N))],
                [jnp.zeros((N, N)), P_ss]
            ])  # shape (2N, 2N)

            # q vector: comes from jerk linear term -2*gamma*b^T J a => q_a = -2 * gamma * J^T b
            q_a = -2.0 * gamma * (J.T @ b)  # length N
            q_s = jnp.zeros((N,))
            q = jnp.concatenate([q_a, q_s])  # length 2N

            # constant term gamma * b^T b can be ignored for optimization (doesn't affect argmin)

            # ---- constraints A z >= l (or l <= A z <= u) ----
            # We'll collect constraints in rows.
            rows = []

            # 1) For each k: s_k + dt * sum_{i=0}^k a_i >= obs_v - v0
            # That is: dt * cumulative_sum_row + s_k >= (obs_v - v0)
            for k in range(N):
                row = jnp.zeros((Z,))
                # coefficients for a_i, i=0..k : dt
                a_coeffs = jnp.concatenate([jnp.ones(k+1) * dt, jnp.zeros(N - (k+1))])
                row = row.at[:N].set(a_coeffs)
                # coefficient for s_k:
                row = row.at[N + k].set(1.0)
                rows.append(row)

            # 2) s_k >= 0  -> row picks s_k >= 0
            for k in range(N):
                row = jnp.zeros((Z,))
                row = row.at[N + k].set(1.0)
                rows.append(row)

            # 3) box bounds for a: a_min <= a_i <= a_max -> we can encode them as rows with identity
            # We'll append identity rows for each a_i and let l/u set to [a_min, a_max]
            for i in range(N):
                row = jnp.zeros((Z,))
                row = row.at[i].set(1.0)
                rows.append(row)

            # 4) Optionally, we could bound s from above with large number; not necessary.
            # Stack rows into A
            A = jnp.stack(rows, axis=0)  # shape (num_rows, Z)

            # Build l and u:
            num_rows = A.shape[0]
            # initialize with -inf / +inf
            NEG_INF = -1e20
            POS_INF = 1e20
            l = jnp.ones((num_rows,)) * NEG_INF
            u = jnp.ones((num_rows,)) * POS_INF

            row_idx = 0
            # fill for constraint type 1: s_k + dt*sum a_i >= obs_v - v0  -> l = obs_v - v0
            for k in range(N):
                l = l.at[row_idx].set(obs_v - v0)
                # u stays +inf
                row_idx += 1

            # constraint type 2: s_k >= 0
            for k in range(N):
                l = l.at[row_idx].set(0.0)
                row_idx += 1

            # constraint type 3: a_min <= a_i <= a_max
            for i in range(N):
                l = l.at[row_idx].set(a_min)
                u = u.at[row_idx].set(a_max)
                row_idx += 1

            assert row_idx == num_rows

            return P, q, A, l, u


        def build_qp_matrices(
            N: int,
            dt: float,
            x0: float,
            v0: float,
            obs_x0: float,
            obs_v: float,
            a_prev: float,
            a_min: float,
            a_max: float,
            alpha_vel: float,
            beta: float,
            gamma: float,
            d_safe: float = 5.0,
            use_pos_slack: bool = True,
            alpha_pos: float = 1e4,
        ):
            """
            Robust version that coerces shapes/dtypes so stacking won't fail.
            Returns: Q, c, G, h  (for jaxopt.OSQP as params_obj=(Q,c), params_ineq=(G,h))
            """

            # coerce scalar inputs to JAX scalars of a consistent dtype
            dt = jnp.asarray(dt, dtype=jnp.float32)
            x0 = jnp.asarray(x0, dtype=jnp.float32)
            v0 = jnp.asarray(v0, dtype=jnp.float32)
            obs_x0 = jnp.asarray(obs_x0, dtype=jnp.float32)
            obs_v = jnp.asarray(obs_v, dtype=jnp.float32)
            a_prev = jnp.asarray(a_prev, dtype=jnp.float32)
            a_min = jnp.asarray(a_min, dtype=jnp.float32)
            a_max = jnp.asarray(a_max, dtype=jnp.float32)
            d_safe = jnp.asarray(d_safe, dtype=jnp.float32)
            alpha_vel = jnp.asarray(alpha_vel, dtype=jnp.float32)
            beta = jnp.asarray(beta, dtype=jnp.float32)
            gamma = jnp.asarray(gamma, dtype=jnp.float32)
            alpha_pos = jnp.asarray(alpha_pos, dtype=jnp.float32)

            # variable counts
            n_a = N
            n_svel = N
            n_spos = N if use_pos_slack else 0
            Z = n_a + n_svel + n_spos

            # ---- objective Q (2N or 3N dims) and linear term c ----
            I_N = jnp.eye(N, dtype=jnp.float32)
            Jmat = make_difference_matrix(N)             # shape (N,N)
            b = jnp.zeros((N,), dtype=jnp.float32).at[0].set(a_prev)  # shape (N,)

            Q_aa = (2.0 * beta) * I_N + (2.0 * gamma) * (Jmat.T @ Jmat)  # (N,N)
            Q_svel = (2.0 * alpha_vel) * I_N                             # (N,N)
            Q_spos = (2.0 * alpha_pos) * I_N if use_pos_slack else jnp.zeros((0, 0), dtype=jnp.float32)

            # assemble Q block-wise with correct dtype
            top = jnp.concatenate([Q_aa, jnp.zeros((N, n_svel + n_spos), dtype=jnp.float32)], axis=1)
            mid = jnp.concatenate([jnp.zeros((n_svel, N), dtype=jnp.float32), Q_svel,
                                jnp.zeros((n_svel, n_spos), dtype=jnp.float32)], axis=1)
            if use_pos_slack:
                bot = jnp.concatenate([jnp.zeros((n_spos, N + n_svel), dtype=jnp.float32), Q_spos], axis=1)
                Q = jnp.concatenate([top, mid, bot], axis=0)
            else:
                Q = jnp.concatenate([top, mid], axis=0)

            # linear term c
            c_a = -2.0 * gamma * (Jmat.T @ b)
            c_svel = jnp.zeros((n_svel,), dtype=jnp.float32)
            c_spos = jnp.zeros((n_spos,), dtype=jnp.float32)
            c = jnp.concatenate([c_a, c_svel, c_spos])

            # ---- inequality rows G z <= h ----
            rows = []  # will hold 1D arrays of shape (Z,)
            hs = []    # will hold 0-D arrays (scalars)

            def append_row(row_vec, h_val):
                """
                Coerce row_vec -> 1D jnp.float32 vector (shape (Z,))
                Coerce h_val -> 0-D jnp.float32 scalar (shape ())
                Append to lists.
                """
                row_arr = jnp.asarray(row_vec, dtype=jnp.float32).ravel()
                # ensure row has exact length Z
                if row_arr.shape != (Z,):
                    # try to reshape / pad or raise with informative message
                    raise ValueError(f"Row has wrong shape {row_arr.shape}, expected {(Z,)}")
                h_arr = jnp.asarray(h_val, dtype=jnp.float32).reshape(())
                rows.append(row_arr)
                hs.append(h_arr)

            # 1) velocity slack constraints (for k=0..N-1)
            # -s_k + dt * sum_{i=0}^k a_i <= obs_v - v0  〈=> s_k >= v_k - obs_v〉
            for k in range(N):
                row = jnp.zeros((Z,), dtype=jnp.float32)
                a_coeffs = jnp.concatenate([jnp.ones(k + 1, dtype=jnp.float32) * dt,
                                            jnp.zeros(N - (k + 1), dtype=jnp.float32)])
                row = row.at[:N].set(a_coeffs)
                row = row.at[N + k].set(-1.0)  # -s_k
                append_row(row, (obs_v - v0))

            # 2) s_vel >= 0  -> -s_vel <= 0
            for k in range(N):
                row = jnp.zeros((Z,), dtype=jnp.float32)
                row = row.at[N + k].set(-1.0)
                append_row(row, 0.0)

            # 3) acceleration box: a_i <= a_max  and  -a_i <= -a_min
            for i in range(N):
                row_pos = jnp.zeros((Z,), dtype=jnp.float32).at[i].set(1.0)
                append_row(row_pos, a_max)
                row_neg = jnp.zeros((Z,), dtype=jnp.float32).at[i].set(-1.0)
                append_row(row_neg, -a_min)

            # 4) position / distance constraints
            # compute c_{k,i}: coefficient for a_i on x_{k+1}: dt^2 * (k - i + 0.5) for i <= k
            for k in range(N):
                cvec = jnp.zeros((N,), dtype=jnp.float32)
                # vectorized fill
                # indices 0..k get dt^2*(k-i+0.5)
                idxs = jnp.arange(k + 1)
                coeffs = dt * dt * ( (k - idxs) + 0.5 )
                cvec = cvec.at[:k+1].set(coeffs)
                RHS = (obs_x0 + (k + 1) * obs_v * dt) - (x0 + (k + 1) * v0 * dt) - d_safe
                if use_pos_slack:
                    # sum cvec * a_i - s_pos_k <= RHS  -> row has +cvec on a, -1 on s_pos_k
                    row = jnp.zeros((Z,), dtype=jnp.float32)
                    row = row.at[:N].set(cvec)
                    spos_idx = N + n_svel + k
                    row = row.at[spos_idx].set(-1.0)
                    append_row(row, RHS)
                    # also s_pos_k >= 0 -> -s_pos_k <= 0
                    row2 = jnp.zeros((Z,), dtype=jnp.float32)
                    row2 = row2.at[spos_idx].set(-1.0)
                    append_row(row2, 0.0)
                else:
                    # hard: sum cvec * a_i <= RHS
                    row = jnp.zeros((Z,), dtype=jnp.float32)
                    row = row.at[:N].set(cvec)
                    append_row(row, RHS)

            # finalize G and h with consistent shapes
            if len(rows) == 0:
                G = jnp.zeros((0, Z), dtype=jnp.float32)
                h = jnp.zeros((0,), dtype=jnp.float32)
            else:
                G = jnp.stack(rows, axis=0)    # shape (m, Z)
                h = jnp.stack(hs, axis=0)      # shape (m,)

            return Q, c, G, h


        @partial(jax.jit, static_argnums=(1,))  # static N
        def select_action_waymax_jax(
            state,  # a small dict-like containing v0, obs_v, a_prev, etc., may be a PyTree of floats
            N=3,
            dt=0.1,
            a_min=-5.0,
            a_max=2.0,
            alpha=1000.0,
            beta=1.0,
            gamma=10.0,
            alpha_vel=500.0,
            d_safe=5.0,
            use_pos_slack=True,
            alpha_pos=1e4,
            ):
            """
            JAX-native MPC QP controller.

            state should contain:
            - 'v0' : current ego velocity (scalar)
            - 'obs_v' : obstacle velocity (scalar) (we assume constant-velocity obstacle model)
            - 'a_prev' : previous acceleration (scalar)
            Optionally other keys could be added.

            Returns:
            a0 : the first acceleration command (scalar)
            info : dict with 'a_plan', 's_plan', 'status' (from solver)
            """
            x0 = jnp.asarray(state['x0'], dtype=jnp.float32)
            obs_x0 = jnp.asarray(state['obs_x0'], dtype=jnp.float32)
            v0 = jnp.asarray(state['v0'], dtype=jnp.float32)
            obs_v = jnp.asarray(state['obs_v'], dtype=jnp.float32)
            a_prev = jnp.asarray(state['a_prev'], dtype=jnp.float32)
            
            Q, c, G, h = build_qp_matrices(
                N=N, dt=dt, x0=x0, v0=v0, obs_x0=obs_x0, obs_v=obs_v,
                a_prev=a_prev, a_min=a_min, a_max=a_max,
                alpha_vel=alpha_vel, beta=beta, gamma=gamma,
                d_safe=d_safe, use_pos_slack=use_pos_slack, alpha_pos=alpha_pos
            )
            
            
            solver = OSQP()

            z, state  = solver.run(params_obj=(Q, c), params_ineq=(G, h))

            # however API might return 'params' or 'primals'; choose typical attribute 'params' above.
            # z is shape (2N,)
            a_plan = z[:N]
            s_plan = z[N:]

            a0 = a_plan[0]

            info = {
                "a_plan": a_plan,
                "s_plan": s_plan,
                "status": state.status,  # solver state info
            }

            return a0, info

        if actor_state is None:
            # initialize reaction timer
            actor_state = {"reaction_timer": 0,
                           "has_reacted": jnp.array(False)}
        SUDDEN_BRAKE_THRESHOLD = -2.0  # m/s^2
        REACTION_STEPS = int(0.25 / datatypes.TIME_INTERVAL)

        reaction_timer = actor_state["reaction_timer"]
        has_reacted = actor_state["has_reacted"]
        
        leader_sudden_brake = acc_lead < jnp.asarray(SUDDEN_BRAKE_THRESHOLD, dtype=acc_lead.dtype)
        start_reaction = (~has_reacted) & (reaction_timer == 0) & leader_sudden_brake
        
        reaction_timer = jnp.where(
            start_reaction,
            jnp.array(REACTION_STEPS, dtype=jnp.int32),
            jnp.where(reaction_timer > 0, reaction_timer - 1, jnp.array(0, dtype=jnp.int32)),
        )
        has_reacted = jnp.where(start_reaction, True, has_reacted)
        
        new_actor_state = {**actor_state, 
                           "reaction_timer": reaction_timer,
                           "has_reacted": has_reacted}

        def during_reaction(_):
            directionF = jnp.where(speed_av_t0 > 1e-3, vel_av_t0 / speed_av_t0, jnp.zeros_like(vel_av_t0))
            new_speedF = jnp.maximum(speed_av_t0 + acc_av * datatypes.TIME_INTERVAL, 0.0)
            new_velocity_av = directionF * new_speedF
            return new_velocity_av
        
        def after_reaction(_):
            state_qp = {"x0": jnp.linalg.norm(pos_av_t0), "obs_x0": jnp.linalg.norm(pos_lead_t0), "v0": speed_av_t0, "obs_v": jnp.linalg.norm(vel_lead_t0), "a_prev": acc_av}
            a0, info = select_action_waymax_jax(state_qp)
            best_accel = a0[0]

            direction_av = jnp.where(speed_av_t0 > 1e-3, vel_av_t0 / speed_av_t0, jnp.zeros_like(vel_av_t0))
            new_speed_av = jnp.maximum(speed_av_t0 + best_accel* datatypes.TIME_INTERVAL, 0.0)
            new_velocity_av = direction_av * new_speed_av
            return new_velocity_av
        
        new_velocity_av = jax.lax.cond(reaction_timer > 0, during_reaction, after_reaction, operand=None)
        
        traj_t1 = traj_t0.replace(
                x=traj_t0.x.at[av_idx].set(pos_av_t0[0] + new_velocity_av[0] * datatypes.TIME_INTERVAL),
                y=traj_t0.y.at[av_idx].set(pos_av_t0[1] + new_velocity_av[1] * datatypes.TIME_INTERVAL),
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
            actor_state=actor_state,
            action=actions,
            is_controlled=is_controlled
        )

    return actor_core.actor_core_factory(
        init=actor_init,
        select_action=select_action,
        name="MPC_actor"
    )