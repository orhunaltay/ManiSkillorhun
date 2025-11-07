#!/usr/bin/env python3
"""
Scripted execution for PickBananaFromOpenDrawerSimple-v1 task.

This script demonstrates picking a banana from an open drawer and placing it
at a goal position using end-effector delta position control.

The state machine follows these steps:
1. Move above banana in drawer
2. Descend to grasp banana
3. Close gripper
4. Lift banana out of drawer
5. Move to goal position
6. Descend to place
7. Open gripper
8. Retreat
"""

import time
import numpy as np
import gymnasium as gym

import mani_skill  # ensure registry loads
import pick_banana_from_open_drawer_simple  # registers PickBananaFromOpenDrawerSimple-v1

from gymnasium.wrappers import ClipAction, RescaleAction

# --- Tunables ---
HOVER_Z = 0.25   # hover height above table (z=0)
GRASP_Z = 0.06   # touch banana height (matches spawn in drawer)
PLACE_Z = 0.10   # place height at goal (matches goal z)
STEP_XYZ = 0.01  # 1 cm per step
FPS = 60.0
HOLD_STEPS = 30
OPEN_CMD = +0.05   # keep within RescaleAction [-0.05, 0.05]
CLOSE_CMD = -0.05
XY_TOL = 0.02      # 2 cm lateral tolerance
Z_TOL = 0.015      # 1.5 cm vertical tolerance


def make_env():
    """Create the environment with appropriate wrappers."""
    env = gym.make(
        "PickBananaFromOpenDrawerSimple-v1",
        obs_mode="state_dict",
        render_mode="human",
        control_mode="pd_ee_delta_pos",   # EE position deltas -> IK via Pinocchio
        num_envs=1,
    )
    # Keep actions small and safe: EE deltas are meters
    env = ClipAction(env)
    env = RescaleAction(env, -0.05, 0.05)  # [-5 cm, 5 cm] and small gripper values
    return env


def clamp(v, lo, hi):
    """Clamp value between lo and hi."""
    return max(lo, min(hi, v))


def get_xyz(pose7):
    """Extract xyz position from pose [x,y,z,qw,qx,qy,qz]."""
    return np.asarray(pose7[..., :3], dtype=np.float32)


def step_towards(env, cur_xyz, target_xyz, grip):
    """
    Step with a tiny EE delta toward target and return (obs, terminated, truncated, info).
    """
    d = target_xyz - cur_xyz
    delta = np.array([clamp(d[i], -STEP_XYZ, STEP_XYZ) for i in range(3)], dtype=np.float32)
    act = np.concatenate([delta, [grip]], dtype=np.float32)  # [dx,dy,dz,grip]
    obs, _, terminated, truncated, info = env.step(act)
    return obs, terminated, truncated, info


def main():
    """Main execution loop."""
    env = make_env()
    obs, info = env.reset(seed=0)
    dt = 1.0 / FPS

    # Helper functions to extract poses from observations
    def tcp():
        return get_xyz(obs["extra"]["tcp_pose"][0])

    def banana():
        return get_xyz(obs["extra"]["banana_pose"][0])

    def goal():
        return get_xyz(obs["extra"]["goal_pose"][0])

    # State machine variables
    state = "go_above_banana"
    hold = 0

    print("=" * 80)
    print("Starting scripted banana picking task")
    print("=" * 80)
    print(f"Initial banana position: {banana()}")
    print(f"Goal position: {goal()}")
    print(f"Initial TCP position: {tcp()}")
    print("=" * 80)

    try:
        while True:
            cur = tcp()

            if state == "go_above_banana":
                # Move above banana in drawer
                target = np.array([banana()[0], banana()[1], HOVER_Z], np.float32)
                obs, _, _, _ = step_towards(env, cur, target, OPEN_CMD)
                if np.linalg.norm(cur[:2] - target[:2]) < XY_TOL and abs(cur[2] - target[2]) < Z_TOL:
                    print(f"✓ Reached above banana at {cur}")
                    state = "descend_to_grasp"

            elif state == "descend_to_grasp":
                # Descend to banana in drawer
                target = np.array([banana()[0], banana()[1], GRASP_Z], np.float32)
                obs, _, _, _ = step_towards(env, cur, target, OPEN_CMD)
                if abs(cur[2] - GRASP_Z) < Z_TOL:
                    print(f"✓ Reached banana at {cur}")
                    state, hold = "close_grip", HOLD_STEPS

            elif state == "close_grip":
                # Hold position, close fingers
                act = np.array([0, 0, 0, CLOSE_CMD], np.float32)
                obs, _, _, _, _ = env.step(act)
                hold -= 1
                if hold <= 0:
                    print(f"✓ Gripper closed")
                    state = "lift_with_banana"

            elif state == "lift_with_banana":
                # Lift banana out of drawer
                target = np.array([cur[0], cur[1], HOVER_Z], np.float32)
                obs, _, _, _ = step_towards(env, cur, target, CLOSE_CMD)
                if abs(cur[2] - HOVER_Z) < Z_TOL:
                    print(f"✓ Lifted banana to {cur}")
                    state = "go_to_goal_xy"

            elif state == "go_to_goal_xy":
                # Move to goal XY position while maintaining hover height
                target = np.array([goal()[0], goal()[1], HOVER_Z], np.float32)
                obs, _, _, _ = step_towards(env, cur, target, CLOSE_CMD)
                if np.linalg.norm(cur[:2] - target[:2]) < XY_TOL and abs(cur[2] - target[2]) < Z_TOL:
                    print(f"✓ Reached above goal at {cur}")
                    state = "descend_to_place"

            elif state == "descend_to_place":
                # Descend to place height at goal
                target = np.array([goal()[0], goal()[1], PLACE_Z], np.float32)
                obs, _, _, _ = step_towards(env, cur, target, CLOSE_CMD)
                if abs(cur[2] - PLACE_Z) < Z_TOL:
                    print(f"✓ Reached place position at {cur}")
                    state, hold = "open_grip", HOLD_STEPS

            elif state == "open_grip":
                # Hold position, open fingers
                act = np.array([0, 0, 0, OPEN_CMD], np.float32)
                obs, _, _, _, _ = env.step(act)
                hold -= 1
                if hold <= 0:
                    print(f"✓ Gripper opened, banana placed")
                    state = "retreat"

            elif state == "retreat":
                # Move back up to hover height
                target = np.array([cur[0], cur[1], HOVER_Z], np.float32)
                obs, _, _, _ = step_towards(env, cur, target, OPEN_CMD)
                if abs(cur[2] - HOVER_Z) < Z_TOL:
                    print(f"✓ Retreated to {cur}")
                    print("=" * 80)
                    print("Task completed!")
                    print(f"Final banana position: {banana()}")
                    print(f"Goal position: {goal()}")
                    print(f"Distance to goal: {np.linalg.norm(banana() - goal()):.4f}m")

                    # Check success
                    eval_info = info.get("eval_info", {})
                    if eval_info.get("success", [False])[0]:
                        print("✓✓✓ SUCCESS! ✓✓✓")
                    else:
                        print("Task completed but success criteria not met")
                    print("=" * 80)
                    state = "done"

            elif state == "done":
                # Hold final position
                act = np.array([0, 0, 0, 0], np.float32)
                obs, _, _, _, _ = env.step(act)

            env.render()
            time.sleep(dt)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        env.close()


if __name__ == "__main__":
    main()
