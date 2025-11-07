#!/usr/bin/env python3
"""
Motion planning script for LightBulbInSocket task.
Pick up a lightbulb and insert it into a socket fixture.

This script uses a simple state machine to:
1. Move above the lightbulb
2. Descend and grasp it
3. Lift it up
4. Move above the socket
5. Align carefully
6. Insert into the socket
7. Release and retreat
"""
import time
import numpy as np
import gymnasium as gym

import mani_skill
import lightbulb_in_socket

from gymnasium.wrappers import ClipAction, RescaleAction

# --- Motion Planning Parameters ---
# Heights relative to table (z=0)
HOVER_HEIGHT = 0.15          # Safe height for moving around
GRASP_HEIGHT = 0.05          # Height to grasp bulb (center of 10cm bulb)
APPROACH_SOCKET_HEIGHT = 0.12  # Height when approaching socket
SOCKET_TOP_OFFSET = 0.06     # Height above socket base to start insertion

# Movement
STEP_SIZE = 0.008            # 8mm per step - slower, more controlled
TIGHT_XY_TOL = 0.005         # 5mm for precise alignment
LOOSE_XY_TOL = 0.015         # 15mm for general movement
Z_TOL = 0.008                # 8mm vertical tolerance

# Timing
FPS = 60.0
GRIP_HOLD_STEPS = 40         # Hold gripper longer for secure grasp
SETTLE_STEPS = 20            # Let physics settle after insertion

# Gripper commands (within RescaleAction bounds)
GRIPPER_OPEN = +0.05
GRIPPER_CLOSE = -0.05


def make_env():
    """Create the environment with appropriate wrappers."""
    env = gym.make(
        "LightBulbInSocket-v1",
        obs_mode="state_dict",
        render_mode="human",
        control_mode="pd_ee_delta_pos",
        num_envs=1,
        sim_backend="cpu",  # Use CPU for compatibility
    )
    env = ClipAction(env)
    env = RescaleAction(env, -0.05, 0.05)  # Limit to ±5cm/step
    return env


def clamp(value, min_val, max_val):
    """Clamp value between min and max."""
    return max(min_val, min(max_val, value))


def extract_xyz(pose7):
    """Extract xyz from 7D pose [x,y,z,qw,qx,qy,qz]."""
    if isinstance(pose7, np.ndarray):
        return pose7[:3].astype(np.float32)
    return np.array(pose7[:3], dtype=np.float32)


def step_towards(env, current_xyz, target_xyz, gripper_cmd, max_step=STEP_SIZE):
    """
    Take one small step towards target position.
    Returns: (obs, done, truncated, info)
    """
    delta = target_xyz - current_xyz
    # Clamp each axis
    delta_clamped = np.array([
        clamp(delta[0], -max_step, max_step),
        clamp(delta[1], -max_step, max_step),
        clamp(delta[2], -max_step, max_step)
    ], dtype=np.float32)

    action = np.concatenate([delta_clamped, [gripper_cmd]], dtype=np.float32)
    obs, _, terminated, truncated, info = env.step(action)
    return obs, terminated, truncated, info


def check_xy_reached(current_xyz, target_xyz, tolerance=LOOSE_XY_TOL):
    """Check if XY position is close enough to target."""
    xy_dist = np.linalg.norm(current_xyz[:2] - target_xyz[:2])
    return xy_dist < tolerance


def check_z_reached(current_z, target_z, tolerance=Z_TOL):
    """Check if Z position is close enough to target."""
    return abs(current_z - target_z) < tolerance


def hold_position(env, gripper_cmd, steps):
    """Hold current position for specified steps."""
    for _ in range(steps):
        action = np.array([0, 0, 0, gripper_cmd], dtype=np.float32)
        obs, _, _, _, info = env.step(action)
    return obs, info


def main():
    print("=" * 60)
    print("LightBulb-in-Socket Motion Planning Demo")
    print("=" * 60)

    env = make_env()
    obs, info = env.reset(seed=42)
    dt = 1.0 / FPS

    # Helper functions to extract positions from observations
    def get_tcp_pos():
        return extract_xyz(obs["extra"]["tcp_pose"][0])

    def get_bulb_pos():
        return extract_xyz(obs["extra"]["lightbulb_pose"][0])

    def get_socket_pos():
        return extract_xyz(obs["extra"]["socket_pose"][0])

    # Print initial positions
    print(f"\nInitial positions:")
    print(f"  TCP (end-effector): {get_tcp_pos()}")
    print(f"  Lightbulb:          {get_bulb_pos()}")
    print(f"  Socket:             {get_socket_pos()}")
    print()

    # State machine
    state = "move_above_bulb"
    hold_counter = 0
    step_count = 0
    state_start_step = 0

    try:
        while True:
            tcp = get_tcp_pos()
            bulb = get_bulb_pos()
            socket = get_socket_pos()
            step_count += 1

            # === STATE: Move above lightbulb ===
            if state == "move_above_bulb":
                target = np.array([bulb[0], bulb[1], HOVER_HEIGHT], dtype=np.float32)
                obs, _, _, info = step_towards(env, tcp, target, GRIPPER_OPEN)

                if check_xy_reached(tcp, target) and check_z_reached(tcp[2], target[2]):
                    print(f"[{step_count:04d}] ✓ Reached above lightbulb")
                    state = "descend_to_grasp"
                    state_start_step = step_count

            # === STATE: Descend to grasp lightbulb ===
            elif state == "descend_to_grasp":
                target = np.array([bulb[0], bulb[1], GRASP_HEIGHT], dtype=np.float32)
                obs, _, _, info = step_towards(env, tcp, target, GRIPPER_OPEN)

                if check_xy_reached(tcp, bulb, TIGHT_XY_TOL) and check_z_reached(tcp[2], GRASP_HEIGHT):
                    print(f"[{step_count:04d}] ✓ Positioned at grasp height")
                    state = "close_gripper"
                    hold_counter = GRIP_HOLD_STEPS

            # === STATE: Close gripper to grasp ===
            elif state == "close_gripper":
                obs, info = hold_position(env, GRIPPER_CLOSE, 1)
                hold_counter -= 1

                if hold_counter <= 0:
                    is_grasped = info.get("is_grasped", [False])[0]
                    if is_grasped:
                        print(f"[{step_count:04d}] ✓ Grasped lightbulb!")
                        state = "lift_bulb"
                    else:
                        print(f"[{step_count:04d}] ⚠ Grasp may have failed, continuing anyway...")
                        state = "lift_bulb"

            # === STATE: Lift lightbulb ===
            elif state == "lift_bulb":
                target = np.array([tcp[0], tcp[1], HOVER_HEIGHT], dtype=np.float32)
                obs, _, _, info = step_towards(env, tcp, target, GRIPPER_CLOSE)

                if check_z_reached(tcp[2], HOVER_HEIGHT):
                    print(f"[{step_count:04d}] ✓ Lifted lightbulb to safe height")
                    state = "move_above_socket"

            # === STATE: Move above socket ===
            elif state == "move_above_socket":
                target = np.array([socket[0], socket[1], APPROACH_SOCKET_HEIGHT], dtype=np.float32)
                obs, _, _, info = step_towards(env, tcp, target, GRIPPER_CLOSE)

                if check_xy_reached(tcp, target, LOOSE_XY_TOL) and check_z_reached(tcp[2], APPROACH_SOCKET_HEIGHT):
                    print(f"[{step_count:04d}] ✓ Reached above socket")
                    state = "align_with_socket"

            # === STATE: Fine alignment with socket ===
            elif state == "align_with_socket":
                target = np.array([socket[0], socket[1], APPROACH_SOCKET_HEIGHT], dtype=np.float32)
                obs, _, _, info = step_towards(env, tcp, target, GRIPPER_CLOSE, max_step=STEP_SIZE/2)

                if check_xy_reached(tcp, target, TIGHT_XY_TOL):
                    print(f"[{step_count:04d}] ✓ Aligned with socket opening")
                    state = "lower_to_socket"

            # === STATE: Lower bulb to socket top ===
            elif state == "lower_to_socket":
                # Socket base is at socket[2], add offset to get to socket opening
                socket_insertion_height = socket[2] + SOCKET_TOP_OFFSET
                target = np.array([socket[0], socket[1], socket_insertion_height], dtype=np.float32)
                obs, _, _, info = step_towards(env, tcp, target, GRIPPER_CLOSE, max_step=STEP_SIZE/2)

                if check_z_reached(tcp[2], socket_insertion_height, Z_TOL * 1.5):
                    print(f"[{step_count:04d}] ✓ Lowered to socket insertion height")
                    state = "insert_bulb"

            # === STATE: Insert bulb into socket ===
            elif state == "insert_bulb":
                # Go even lower to fully insert
                insert_depth = socket[2] + SOCKET_TOP_OFFSET - 0.02  # 2cm deeper
                target = np.array([socket[0], socket[1], insert_depth], dtype=np.float32)
                obs, _, _, info = step_towards(env, tcp, target, GRIPPER_CLOSE, max_step=STEP_SIZE/3)

                # Check if inserted
                if info.get("inserted", [False])[0]:
                    print(f"[{step_count:04d}] ✓ Lightbulb inserted into socket!")
                    state = "settle"
                    hold_counter = SETTLE_STEPS

                # Timeout if taking too long
                if step_count - state_start_step > 200:
                    print(f"[{step_count:04d}] ⚠ Insertion timeout, proceeding to release")
                    state = "settle"
                    hold_counter = SETTLE_STEPS

            # === STATE: Let physics settle ===
            elif state == "settle":
                obs, info = hold_position(env, GRIPPER_CLOSE, 1)
                hold_counter -= 1
                if hold_counter <= 0:
                    state = "open_gripper"
                    hold_counter = GRIP_HOLD_STEPS // 2

            # === STATE: Open gripper to release ===
            elif state == "open_gripper":
                obs, info = hold_position(env, GRIPPER_OPEN, 1)
                hold_counter -= 1
                if hold_counter <= 0:
                    print(f"[{step_count:04d}] ✓ Released lightbulb")
                    state = "retreat"

            # === STATE: Retreat from socket ===
            elif state == "retreat":
                target = np.array([tcp[0], tcp[1], HOVER_HEIGHT], dtype=np.float32)
                obs, _, _, info = step_towards(env, tcp, target, GRIPPER_OPEN)

                if check_z_reached(tcp[2], HOVER_HEIGHT):
                    print(f"[{step_count:04d}] ✓ Retreated to safe height")
                    state = "done"

            # === STATE: Done ===
            elif state == "done":
                obs, info = hold_position(env, GRIPPER_OPEN, 1)

                # Check final success
                if step_count % 60 == 0:  # Check every second
                    success = info.get("success", [False])[0]
                    inserted = info.get("inserted", [False])[0]
                    xy_aligned = info.get("xy_aligned", [False])[0]

                    print(f"\nStatus at step {step_count}:")
                    print(f"  XY aligned: {xy_aligned}")
                    print(f"  Inserted:   {inserted}")
                    print(f"  Success:    {success}")

                    if success:
                        print("\n" + "=" * 60)
                        print("🎉 SUCCESS! Lightbulb successfully inserted into socket!")
                        print("=" * 60)
                        break

            # Render
            env.render()
            time.sleep(dt)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        print(f"\nTotal steps: {step_count}")
        env.close()


if __name__ == "__main__":
    main()
