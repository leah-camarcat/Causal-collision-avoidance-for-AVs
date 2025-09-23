from typing import Callable, Optional
import jax
import jax.numpy as jnp
from waymax import datatypes, dynamics
from waymax.agents import actor_core
import numpy as np

def davis_actor(
        dynamics_model: dynamics.DynamicsModel,
        is_controlled_func: Callable[[datatypes.SimulatorState], jax.Array],
        av_idx: int,
        lead_idx: int,
) -> actor_core.WaymaxActorCore:
    """Actor that computes acceleration/deceleration based on leading vehicle and AV states."""

    def y_func(speedL, speedF, accF, posL, posF, lengthF, lengthL):
        l = lengthF/2 + lengthL/2
        y = -(speedF**2) /((2 * jnp.abs(accF) + 1e-6)) + speedL*0.1 + (jnp.linalg.norm(posL - posF) - l)
        return y
    
    def actor_init(rng, init_state):
        # reaction_timer tracks remaining steps before AV reacts
        return {"reaction_timer": jnp.array(0, dtype=jnp.int32),
                "has_reacted": jnp.array(False)}

    def select_action(
            params: actor_core.Params,
            state: datatypes.SimulatorState,
            actor_state=None,
            rng: jax.Array = None,
    ) -> actor_core.WaymaxActorOutput:
        #del params, actor_state, rng

        is_controlled = is_controlled_func(state)

        # Status of current state
        traj_t0 = datatypes.dynamic_index(
            state.sim_trajectory, state.timestep, axis=-1, keepdims=True
        )
        #jax.debug.print("timestep={}, actor_state_in={}", state.timestep, actor_state)


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

        lengthF = traj_t0.length[av_idx, 0]
        lengthL = traj_t0.length[lead_idx, 0]
        # ========== 计算速度标量 ==========
        speedF_t0 = jnp.linalg.norm(velF_t0)
        speedF_prev = jnp.linalg.norm(velF_prev)
        speedL_t0 = jnp.linalg.norm(velL_t0)
        speedL_prev = jnp.linalg.norm(velL_prev)

        #jax.debug.print("velF_t0={}", velF_t0)
        #jax.debug.print("speedF_t0={}", speedF_t0)
        #jax.debug.print("velL_t0={}", velL_t0)
        #jax.debug.print("speedL_t0={}", speedL_t0)
        #jax.debug.print("velF_prev={}", velF_prev)
        #jax.debug.print("velL_prev={}", velL_prev)

        # ========== 计算加速度 ==========
        accF = (speedF_t0 - speedF_prev) / datatypes.TIME_INTERVAL
        accL = (speedL_t0 - speedL_prev) / datatypes.TIME_INTERVAL

        #jax.debug.print("accF={}", accF)
        #jax.debug.print("accL={}", accL)

        # ========== 计算 headway ==========
        headway = jnp.linalg.norm(posL_t0 - posF_t0)
        #jax.debug.print("h={}", headway)
        # ========== 调整 accF_try ==========

        if actor_state is None:
            # initialize reaction timer
            actor_state = {"reaction_timer": 0,
                           "has_reacted": jnp.array(False)}
        #jax.debug.print("actor_state={}", actor_state)

        # Detect sudden leader braking
        SUDDEN_BRAKE_THRESHOLD = -2.0  # m/s^2
        REACTION_STEPS = int(0.25 / datatypes.TIME_INTERVAL)

        # Step 1: read previous timer
        reaction_timer = actor_state["reaction_timer"]
        has_reacted = actor_state["has_reacted"]

        #jax.debug.print("reaction timer: {}", reaction_timer)
        # Step 2: detect trigger only when not already reacting
        leader_sudden_brake = accL < jnp.asarray(SUDDEN_BRAKE_THRESHOLD, dtype=accL.dtype)
        start_reaction = (~has_reacted) & (reaction_timer == 0) & leader_sudden_brake
        #start_reaction = (reaction_timer == 0) & (accL < SUDDEN_BRAKE_THRESHOLD)
        #reaction_timer = jnp.where(start_reaction, REACTION_STEPS, reaction_timer)
        reaction_timer = jnp.where(
            start_reaction,
            jnp.array(REACTION_STEPS, dtype=jnp.int32),
            jnp.where(reaction_timer > 0, reaction_timer - 1, jnp.array(0, dtype=jnp.int32)),
        )
        has_reacted = jnp.where(start_reaction, True, has_reacted)
        
        #jax.debug.print("accL: {}, start_reaction: {}, timer: {}", accL, start_reaction, reaction_timer)

        # Update actor_state
        new_actor_state = {**actor_state, 
                           "reaction_timer": reaction_timer,
                           "has_reacted": has_reacted}


        def during_reaction(_):
            return accF
        
        def after_reaction(_):
            # --- Parameters to tune ---
            NUM_CANDIDATES = 41          # resolution of search
            MAX_BRAKE = 3.50              # strongest braking (m/s^2)
            MAX_ACCEL = 2.0              # strongest comfortable acceleration (m/s^2)
            EPS = 1e-6

            # Build candidate accelerations
            candidates = jnp.linspace(MAX_ACCEL, -MAX_BRAKE, NUM_CANDIDATES)

            # Evaluate y_func for each candidate (counterfactual search)
            #y_vals = jax.vmap(lambda a: y_func(speedL_t0, accL, speedF_t0, headway, a))(candidates)
            y_vals = jax.vmap(lambda a: y_func(speedL_t0, speedF_t0, accF, posL_t0, posF_t0, lengthF, lengthL))(candidates)
        
            # Mask of safe accelerations
            safe_mask = y_vals >= 0.0
            any_safe = jnp.any(safe_mask)
            
            #jax.debug.print("y vals={} and any_safe={}", y_vals, any_safe)
            
            # Choose least intrusive safe acceleration
            def choose_safe():
                safe_indices = jnp.nonzero(safe_mask, size=NUM_CANDIDATES)[0]
                safe_candidates = candidates[safe_indices]
                diffs = jnp.abs(safe_candidates - accF)
                idx = jnp.argmin(diffs)
                return jnp.asarray(safe_candidates[idx])

            # Fallback: strongest brake if nothing is safe
            def choose_emergency():
                return jnp.asarray(-MAX_BRAKE)

            accF_try = jax.lax.cond(any_safe, choose_safe, choose_emergency)
        
            # Optional: limit jerk
            MAX_JERK = 6.0
            max_delta = MAX_JERK * datatypes.TIME_INTERVAL
            accF_try = jnp.clip(accF_try, accF_try - max_delta, accF_try + max_delta)

            return accF_try
        
        accF_try = jax.lax.cond(reaction_timer > 0, during_reaction, after_reaction, operand=None)

        #jax.debug.print("Suggested accF_try={}", accF_try)

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
            actor_state=new_actor_state,
            action=actions,
            is_controlled=is_controlled,
        )

    return actor_core.actor_core_factory(
        #init=lambda rng, init_state: {"reaction_timer": jnp.array(0, dtype=jnp.int32)},
        init=actor_init,
        select_action=select_action,
        name="davis_actor"
    )