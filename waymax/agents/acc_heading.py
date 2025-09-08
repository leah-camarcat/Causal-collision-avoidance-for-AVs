from typing import Callable

import jax
import jax.numpy as jnp

from waymax import datatypes
from waymax import dynamics
from waymax.agents import actor_core

def create_acceleration_heading_actor(
    dynamics_model: dynamics.DynamicsModel,
    is_controlled_func: Callable[[datatypes.SimulatorState], jax.Array],
    control_array: jax.Array,  # shape: [num_objects, 2, horizon], with [acc, heading]
) -> actor_core.WaymaxActorCore:
  """
  Create an actor that follows per-frame acceleration and heading commands.
  Args:
    dynamics_model: Waymax dynamics model.
    is_controlled_func: function that returns bool mask for controlled objects.
    control_array: shape [2, horizon], where [0, :] is acceleration,
                   and [1, :] is heading in radians.
  """

  def select_action(params, state, actor_state=None, rng=None):
      del params, actor_state, rng

      # index of current frame
      t = state.timestep

      # current acceleration and heading
      acc_t = control_array[0, t]
      heading_t = control_array[1, t]

      # current trajectory
      traj_t0 = datatypes.dynamic_index(state.sim_trajectory, t, axis=-1, keepdims=True)

      vel_x = traj_t0.vel_x
      vel_y = traj_t0.vel_y
      velocity = jnp.stack([vel_x, vel_y], axis=-1)
      speed = jnp.linalg.norm(velocity, axis=-1, keepdims=True)

      # Use current heading to update speed direction
      direction = jnp.stack([jnp.cos(heading_t), jnp.sin(heading_t)], axis=-1)

      # Update speed
      new_speed = jnp.maximum(speed + acc_t * datatypes.TIME_INTERVAL, 0.0)
      new_velocity = direction * new_speed
      new_vel_x, new_vel_y = new_velocity[..., 0], new_velocity[..., 1]
      is_controlled = is_controlled_func(state)
      heading = jnp.arctan2(new_velocity[..., 1], new_velocity[..., 0])

      # calculate the next frame's trajectory
      traj_t1 = traj_t0.replace(
          x=traj_t0.x + new_vel_x * datatypes.TIME_INTERVAL,
          y=traj_t0.y + new_vel_y * datatypes.TIME_INTERVAL,
          vel_x=new_vel_x,
          vel_y=new_vel_y,
          yaw=heading,
          valid=is_controlled[..., jnp.newaxis] & traj_t0.valid,
          timestamp_micros=traj_t0.timestamp_micros + datatypes.TIMESTEP_MICROS_INTERVAL,
      )

      # concatenate traj_t0 and traj_t1，put it in inverse()
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
      name="acceleration_heading_actor"
  )