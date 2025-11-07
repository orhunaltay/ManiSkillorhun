#!/usr/bin/env python3
"""
Motion planning script for LightBulbInSocket task.
Pick up a lightbulb and insert it into a socket fixture.
"""
import time
import numpy as np
import gymnasium as gym

import mani_skill                    # ensure registry loads
import lightbulb_in_socket          # registers LightBulbInSocket-v1

from gymnasium.wrappers import ClipAction, RescaleAction

# --- Tunables ---
HOVER_Z   = 0.20   # hover height above table (z=0)
GRASP_Z   = 0.08   # grasp lightbulb height
APPROACH_Z = 0.25  # approach height above socket
INSERT_Z_OFFSET = 0.05  # how much to lower from socket top for insertion
STEP_XYZ  = 0.01   # 1 cm per step
FPS       = 60.0
HOLD_STEPS = 30
OPEN_CMD  = +0.05  # gripper commands
CLOSE_CMD = -0.05
XY_TOL    = 0.01   # 1 cm lateral tolerance
Z_TOL     = 0.005  # 5 mm vertical tolerance

def make_env():
    env = gym.make(
        "LightBulbInSocket-v1",
        obs_mode="state_dict",
        render_mode="human",
        control_mode="pd_ee_delta_pos",   # EE position deltas -> IK
        num_envs=1,
    )
    # Keep actions small and safe
    env = ClipAction(env)
    env = RescaleAction(env, -0.05, 0.05)
    return env

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def get_xyz(pose7):
    return np.asarray(pose7[..., :3], dtype=np.float32)

def step_towards(env, cur_xyz, target_xyz, grip):
    """Step with a tiny EE delta toward target."""
    d = target_xyz - cur_xyz
    delta = np.array([clamp(d[i], -STEP_XYZ, STEP_XYZ) for i in range(3)], dtype=np.float32)
    act = np.concatenate([delta, [grip]], dtype=np.float32)
    obs, _, terminated, truncated, info = env.step(act)
    return obs, terminated, truncated, info

def main():
    env = make_env()
    obs, info = env.reset(seed=0)
    dt = 1.0 / FPS

    def tcp():      return get_xyz(obs["extra"]["tcp_pose"][0])
    def bulb():     return get_xyz(obs["extra"]["lightbulb_pose"][0])
    def socket():   return get_xyz(obs["extra"]["socket_pose"][0])

    state, hold = "go_above_bulb", 0

    print("Starting LightBulb-in-Socket motion planning...")
    print(f"Initial positions:")
    print(f"  TCP: {tcp()}")
    print(f"  Bulb: {bulb()}")
    print(f"  Socket: {socket()}")

    try:
        step_count = 0
        while True:
            cur = tcp()
            step_count += 1

            if state == "go_above_bulb":
                target = np.array([bulb()[0], bulb()[1], HOVER_Z], np.float32)
                obs, _, _, _ = step_towards(env, cur, target, OPEN_CMD)
                if np.linalg.norm(cur[:2]-target[:2]) < XY_TOL and abs(cur[2]-target[2]) < Z_TOL:
                    print(f"[Step {step_count}] Reached above bulb")
                    state = "descend_to_grasp"

            elif state == "descend_to_grasp":
                target = np.array([bulb()[0], bulb()[1], GRASP_Z], np.float32)
                obs, _, _, _ = step_towards(env, cur, target, OPEN_CMD)
                if abs(cur[2]-GRASP_Z) < Z_TOL:
                    print(f"[Step {step_count}] Reached grasp height")
                    state, hold = "close_grip", HOLD_STEPS

            elif state == "close_grip":
                # Hold position, close gripper
                act = np.array([0, 0, 0, CLOSE_CMD], np.float32)
                obs, _, _, _, _ = env.step(act)
                hold -= 1
                if hold <= 0:
                    print(f"[Step {step_count}] Grasped bulb")
                    state = "lift_with_bulb"

            elif state == "lift_with_bulb":
                target = np.array([cur[0], cur[1], HOVER_Z], np.float32)
                obs, _, _, _ = step_towards(env, cur, target, CLOSE_CMD)
                if abs(cur[2]-HOVER_Z) < Z_TOL:
                    print(f"[Step {step_count}] Lifted bulb to hover height")
                    state = "go_above_socket"

            elif state == "go_above_socket":
                # Move to position above socket
                target = np.array([socket()[0], socket()[1], APPROACH_Z], np.float32)
                obs, _, _, _ = step_towards(env, cur, target, CLOSE_CMD)
                if np.linalg.norm(cur[:2]-target[:2]) < XY_TOL and abs(cur[2]-target[2]) < Z_TOL:
                    print(f"[Step {step_count}] Reached above socket")
                    state = "align_with_socket"

            elif state == "align_with_socket":
                # Fine alignment: center bulb over socket opening
                # Socket is at socket()[2], we want to be slightly above
                target = np.array([socket()[0], socket()[1], socket()[2] + 0.15], np.float32)
                obs, _, _, _ = step_towards(env, cur, target, CLOSE_CMD)
                if np.linalg.norm(cur[:2]-target[:2]) < XY_TOL/2:  # tighter tolerance
                    print(f"[Step {step_count}] Aligned with socket")
                    state = "descend_to_insert"

            elif state == "descend_to_insert":
                # Carefully insert bulb into socket
                # Target is slightly above socket base to insert the bulb base
                insert_target_z = socket()[2] + INSERT_Z_OFFSET
                target = np.array([socket()[0], socket()[1], insert_target_z], np.float32)
                obs, _, _, _ = step_towards(env, cur, target, CLOSE_CMD)

                # Check if inserted successfully
                if abs(cur[2] - insert_target_z) < Z_TOL:
                    print(f"[Step {step_count}] Inserted bulb into socket")
                    state, hold = "release_bulb", HOLD_STEPS

            elif state == "release_bulb":
                # Hold position briefly, then release
                act = np.array([0, 0, 0, OPEN_CMD], np.float32)
                obs, _, _, _, _ = env.step(act)
                hold -= 1
                if hold <= 0:
                    print(f"[Step {step_count}] Released bulb")
                    state = "retreat"

            elif state == "retreat":
                # Move up and away
                target = np.array([cur[0], cur[1], APPROACH_Z], np.float32)
                obs, _, _, _ = step_towards(env, cur, target, OPEN_CMD)
                if abs(cur[2] - APPROACH_Z) < Z_TOL:
                    print(f"[Step {step_count}] Retreated from socket")
                    # Check success
                    if "success" in info and info["success"]:
                        print("\n=== SUCCESS! Lightbulb inserted into socket ===")
                    else:
                        print("\n=== Task completed (may need adjustment) ===")
                    print(f"Total steps: {step_count}")
                    state = "done"

            elif state == "done":
                # Stay in place
                act = np.array([0, 0, 0, OPEN_CMD], np.float32)
                obs, _, _, _, _ = env.step(act)

            env.render()
            time.sleep(dt)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        env.close()

if __name__ == "__main__":
    main()
