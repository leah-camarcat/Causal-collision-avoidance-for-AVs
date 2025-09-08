
from typing import Callable, Optional

import jax
import jax.numpy as jnp

from waymax import datatypes
from waymax import dynamics
from waymax.agents import actor_core
from waymax.agents import waypoint_following_agent

def create_constant_acceleration_actor(
    dynamics_model: dynamics.DynamicsModel,
    is_controlled_func: Callable[[datatypes.SimulatorState], jax.Array],
    acceleration: float = 0.5,  # m/s²
) -> actor_core.WaymaxActorCore:
  """Creates an actor that decelerates with a fixed rate each step."""

  def select_action(
      params: actor_core.Params,
      state: datatypes.SimulatorState,
      actor_state=None,
      rng: jax.Array = None,
  ) -> actor_core.WaymaxActorOutput:
    del params, actor_state, rng  # unused

        # Status of current state
    traj_t0 = datatypes.dynamic_index(
        state.sim_trajectory, state.timestep, axis=-1, keepdims=True
    )

    # cureent speed and direction
    vel_x = traj_t0.vel_x
    vel_y = traj_t0.vel_y
    velocity = jnp.stack([vel_x, vel_y], axis=-1)
    speed = jnp.linalg.norm(velocity, axis=-1, keepdims=True)
    direction = jnp.where(speed > 1e-3, velocity / speed, jnp.zeros_like(velocity))

    # new speed after deceleration, keeping the direction the same
    new_speed = jnp.maximum(speed + acceleration * datatypes.TIME_INTERVAL, 0.0)
    new_velocity = direction * new_speed
    new_vel_x = new_velocity[..., 0]
    new_vel_y = new_velocity[..., 1]

    is_controlled = is_controlled_func(state)

    traj_t1 = traj_t0.replace(
        x=traj_t0.x + new_vel_x * datatypes.TIME_INTERVAL,
        y=traj_t0.y + new_vel_y * datatypes.TIME_INTERVAL,
        vel_x=new_vel_x,
        vel_y=new_vel_y,
        valid=is_controlled[..., jnp.newaxis] & traj_t0.valid,
        timestamp_micros=traj_t0.timestamp_micros + datatypes.TIMESTEP_MICROS_INTERVAL,
    )

    # 拼接 traj_t0 和 traj_t1 传入 inverse()
    traj_combined = jax.tree_util.tree_map(
        lambda x, y: jnp.concatenate([x, y], axis=-1), traj_t0, traj_t1
    )
    actions = dynamics_model.inverse(
        traj_combined, state.object_metadata, timestep=0
    )

    return actor_core.WaymaxActorOutput(
        actor_state=None,
        action=actions,
        is_controlled=is_controlled,
    )

  return actor_core.actor_core_factory(
      init=lambda rng, init_state: None,
      select_action=select_action,
      name=f'constant_deceleration_{acceleration}',
  )

