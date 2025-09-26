from typing import Callable
import jax
import jax.numpy as jnp

from waymax import datatypes, dynamics
from waymax.agents import actor_core


def create_lane_change_actor(
    dynamics_model: dynamics.DynamicsModel,
    is_controlled_func: Callable[[datatypes.SimulatorState], jax.Array],
    *,
    lane_width: float = 3.5,
    side: int = 1,          # +1 left, -1 right
    duration_s: float = 1.5,  # previously 3.0
):
    """Lane change actor that executes once and then maintains position in the new lane."""
    dt = datatypes.TIME_INTERVAL

    def init_fn(rng, init_state):
        return {
            "in_progress": jnp.array(False),
            "elapsed": jnp.array(0.0),
            "lateral_progress": jnp.array(0.0),
            "total_shift": jnp.array(0.0),
            "done": jnp.array(False),
        }

    def smooth_s(t, T):
        frac = jnp.clip(t / T, 0.0, 1.0)
        return 0.5 * (1.0 - jnp.cos(jnp.pi * frac))  # 0→1 平滑过渡

    def select_action(params, state: datatypes.SimulatorState, actor_state=None, rng=None):
        del params, rng
        if actor_state is None:
            actor_state = init_fn(None, None)

        is_controlled = jnp.asarray(is_controlled_func(state))

        traj_t0 = datatypes.dynamic_index(
            state.sim_trajectory, state.timestep, axis=-1, keepdims=True
        )
        pos = jnp.stack(
            [jnp.squeeze(traj_t0.x, -1), jnp.squeeze(traj_t0.y, -1)], axis=-1
        )
        vel = jnp.stack(
            [jnp.squeeze(traj_t0.vel_x, -1), jnp.squeeze(traj_t0.vel_y, -1)], axis=-1
        )

        speed = jnp.linalg.norm(vel, axis=-1, keepdims=True) + 1e-6
        heading = vel / speed
        lateral = jnp.stack([-heading[..., 1], heading[..., 0]], axis=-1)

        # check whether we need to start
        in_progress = actor_state["in_progress"]
        done = actor_state["done"]
        start = jnp.logical_and(is_controlled, jnp.logical_not(in_progress) & jnp.logical_not(done))

        total_shift = jnp.where(start, side * lane_width, actor_state["total_shift"])
        in_progress = jnp.logical_or(in_progress, start)
        elapsed = jnp.where(start, 0.0, actor_state["elapsed"] + dt)

        # compute lateral shift progress
        s_now = smooth_s(elapsed, duration_s)
        target_shift = total_shift * s_now

        delta_shift = target_shift - actor_state["lateral_progress"]
        lateral_progress = actor_state["lateral_progress"] + delta_shift

        # compute forward + lateral displacement
        forward_pos = pos + heading * speed[..., 0:1] * dt
        desired_pos = forward_pos + lateral * delta_shift[..., None]
        desired_vel = heading * speed  # 保持纵向速度不变

        # check if maneuver finished
        finished = elapsed >= duration_s
        in_progress = jnp.where(finished, False, in_progress)
        elapsed = jnp.where(finished, actor_state["elapsed"], elapsed)
        lateral_progress = jnp.where(finished, total_shift, lateral_progress)
        done = jnp.logical_or(done, finished)

        traj_t1 = traj_t0.replace(
            x=desired_pos[..., 0][..., None],
            y=desired_pos[..., 1][..., None],
            vel_x=desired_vel[..., 0][..., None],
            vel_y=desired_vel[..., 1][..., None],
            valid=is_controlled[..., None] & traj_t0.valid,
            timestamp_micros=(traj_t0.timestamp_micros + datatypes.TIMESTEP_MICROS_INTERVAL),
        )

        traj_combined = jax.tree_util.tree_map(
            lambda a, b: jnp.concatenate([a, b], axis=-1), traj_t0, traj_t1
        )
        actions = dynamics_model.inverse(traj_combined, state.object_metadata, timestep=0)

        new_actor_state = {
            "in_progress": in_progress,
            "elapsed": elapsed,
            "lateral_progress": lateral_progress,
            "total_shift": total_shift,
            "done": done,
        }

        return actor_core.WaymaxActorOutput(
            actor_state=new_actor_state, action=actions, is_controlled=is_controlled
        )

    return actor_core.actor_core_factory(
        init=init_fn, select_action=select_action, name="lane_change_once"
    )
