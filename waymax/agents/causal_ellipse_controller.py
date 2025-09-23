from typing import Callable
import jax
import jax.numpy as jnp
from waymax import datatypes, dynamics
from waymax.agents import actor_core


def causal_ellipse_actor(
    dynamics_model: dynamics.DynamicsModel,
    is_controlled_func: Callable[[datatypes.SimulatorState], jax.Array],
    av_idx: int,
    neigh_idx: int,
) -> actor_core.WaymaxActorCore:
    """Actor that uses an oriented safety ellipse to evaluate collision risk
    and select evasive longitudinal acceleration.
    """

    # Parameters for ellipse
    L = 4.5
    W = 2.0
    m_long = 2.0
    m_lat = 1.0
    kappa_a = 1.0   # time headway scaling (s)
    kappa_b = 0.05  # lateral margin scaling

    # Candidate accelerations to search
    NUM_CANDIDATES = 41
    MAX_BRAKE = 4.0
    MAX_ACCEL = 2.0

    def compute_axes(v_long: jax.Array):
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
            a, b = compute_axes(v_long_after)
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
