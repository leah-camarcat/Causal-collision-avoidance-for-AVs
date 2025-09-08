from typing import Callable, Optional
import jax
import jax.numpy as jnp
from waymax import datatypes, dynamics
from waymax.agents import actor_core


def davis_actor(
        dynamics_model: dynamics.DynamicsModel,
        is_controlled_func: Callable[[datatypes.SimulatorState], jax.Array],
        av_idx: int,
        lead_idx: int,
) -> actor_core.WaymaxActorCore:
    """Actor that computes acceleration/deceleration based on leading vehicle and AV states."""

    def y_func(speedL, accL, speedF, headway, accF, r=0.7):
        cond = jnp.logical_and(accL < 0, accL > -0.1)

        y = jnp.where(
            cond,
            speedF * headway - r * speedF - (speedF ** 2) / (2 * accF + 1e-6),
            speedF * headway + (speedL ** 2) / (2 * accL + 1e-6)
            - r * speedF - (speedF ** 2) / (2 * accF + 1e-6),
        )
        return y

    def select_action(
            params: actor_core.Params,
            state: datatypes.SimulatorState,
            actor_state=None,
            rng: jax.Array = None,
    ) -> actor_core.WaymaxActorOutput:
        del params, actor_state, rng

        is_controlled = is_controlled_func(state)

        # Status of current state
        traj_t0 = datatypes.dynamic_index(
            state.sim_trajectory, state.timestep, axis=-1, keepdims=True
        )
        jax.debug.print("Timestep={}",state.timestep)

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

            # ========== 提取 AV 和 leading vehicle 的位置 / 速度 ==========
        posF_t0 = jnp.array([traj_t0.x[av_idx, 0], traj_t0.y[av_idx, 0]])
        velF_t0 = jnp.array([traj_t0.vel_x[av_idx, 0], traj_t0.vel_y[av_idx, 0]])
        velF_prev = jnp.array([traj_prev.vel_x[av_idx, 0], traj_prev.vel_y[av_idx, 0]])

        posL_t0 = jnp.array([traj_t0.x[lead_idx, 0], traj_t0.y[lead_idx, 0]])
        velL_t0 = jnp.array([traj_t0.vel_x[lead_idx, 0], traj_t0.vel_y[lead_idx, 0]])
        velL_prev = jnp.array([traj_prev.vel_x[lead_idx, 0], traj_prev.vel_y[lead_idx, 0]])

        # ========== 计算速度标量 ==========
        speedF_t0 = jnp.linalg.norm(velF_t0)
        speedF_prev = jnp.linalg.norm(velF_prev)
        speedL_t0 = jnp.linalg.norm(velL_t0)
        speedL_prev = jnp.linalg.norm(velL_prev)

        jax.debug.print("velF_t0={}", velF_t0)
        jax.debug.print("velL_t0={}", velL_t0)
        jax.debug.print("velF_prev={}", velF_prev)
        jax.debug.print("velL_prev={}", velL_prev)

        # ========== 计算加速度 ==========
        accF = (speedF_t0 - speedF_prev) / datatypes.TIME_INTERVAL
        accL = (speedL_t0 - speedL_prev) / datatypes.TIME_INTERVAL

        jax.debug.print("accF={}", accF)
        jax.debug.print("accL={}", accL)

        # ========== 计算 headway ==========
        headway = jnp.linalg.norm(posL_t0 - posF_t0)

        # ========== 调整 accF_try ==========
        y_val = y_func(speedL_t0, accL, speedF_t0, headway, accF)

        def adjust_acc_if_negative(_):
            """当 y_val < 0 时，调整 accF_try."""
            # 如果 accF > 0，就先设为 0
            # accF_init = jnp.where(accF > 0, 0.0, accF)
            y_init = y_func(speedL_t0, accL, speedF_t0, headway, accF)

            def cond_fun(val):
                y_new, acc_try, iter = val
                jax.debug.print("y_new:{}", y_new)
                jax.debug.print("acc_try:{}", acc_try)
                return (y_new < 0) & (iter < 1000)

            def body_fun(val):
                y_new, acc_try, iter = val

                acc_try = acc_try - 1
                jax.debug.print("y_new:{}",y_new)
                jax.debug.print("acc_try:{}",acc_try)

                y_new = y_func(speedL_t0, accL, speedF_t0, headway, acc_try)
                iter += 1
                return y_new, acc_try, iter

            y_new, accF_try, _ = jax.lax.while_loop(cond_fun, body_fun, (y_init, accF, 0))
            return accF_try

        def keep_acc_if_positive(_):
            """当 y_val >= 0 时，保持原 accF."""
            return accF

        # 如果 y_val < 0 就调用 adjust_acc_if_negative，否则保持 accF
        accF_try = jax.lax.cond(y_val < 0, adjust_acc_if_negative, keep_acc_if_positive, operand=None)

        jax.debug.print("Suggested accF_try={}", accF_try)

        # ========== 更新 AV 的速度 ==========
        directionF = jnp.where(speedF_t0 > 1e-3, velF_t0 / speedF_t0, jnp.zeros_like(velF_t0))
        new_speedF = jnp.maximum(speedF_t0 + accF_try * datatypes.TIME_INTERVAL, 0.0)
        new_velocityF = directionF * new_speedF

        # 构造新的 trajectory
        traj_t1 = traj_t0.replace(
            x=traj_t0.x.at[av_idx].set(posF_t0[0] + new_velocityF[0] * datatypes.TIME_INTERVAL),
            y=traj_t0.y.at[av_idx].set(posF_t0[1] + new_velocityF[1] * datatypes.TIME_INTERVAL),
            vel_x=traj_t0.vel_x.at[av_idx].set(new_velocityF[0]),
            vel_y=traj_t0.vel_y.at[av_idx].set(new_velocityF[1]),
            valid=is_controlled[..., None] & traj_t0.valid,
            timestamp_micros=traj_t0.timestamp_micros + datatypes.TIMESTEP_MICROS_INTERVAL,
        )

        # 拼接 t0 和 t1 来推导 action
        traj_combined = jax.tree_util.tree_map(
            lambda x, y: jnp.concatenate([x, y], axis=-1), traj_t0, traj_t1
        )
        actions = dynamics_model.inverse(traj_combined, state.object_metadata, timestep=0)

        return actor_core.WaymaxActorOutput(
            actor_state=None,
            action=actions,
            is_controlled=is_controlled,
        )

    return actor_core.actor_core_factory(
        init=lambda rng, init_state: None,
        select_action=select_action,
        name="davis_actor"
    )