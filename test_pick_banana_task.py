#!/usr/bin/env python3
"""
Test script for PickBananaFromOpenDrawer task.
This script verifies that the task can be created and initialized properly.
"""

import numpy as np
import gymnasium as gym


def test_task_creation():
    """Test that the task can be created and reset"""
    print("=" * 80)
    print("Testing PickBananaFromOpenDrawer task creation...")
    print("=" * 80)

    try:
        # Create environment
        env = gym.make(
            "PickBananaFromOpenDrawer-v1",
            obs_mode="state",
            control_mode="pd_joint_pos",
            render_mode="rgb_array",
            camera_width=128,
            camera_height=128,
            distraction_set={},
        )
        print("✓ Environment created successfully")

        # Reset environment
        obs, info = env.reset(seed=0)
        print("✓ Environment reset successfully")

        # Print some basic info
        print(f"\nEnvironment info:")
        print(f"  - Observation space: {env.observation_space}")
        print(f"  - Action space: {env.action_space}")
        print(f"  - Max episode steps: {env.spec.max_episode_steps}")
        print(f"  - Observation keys: {list(obs.keys())}")

        # Check that key objects exist
        unwrapped_env = env.unwrapped
        assert hasattr(unwrapped_env, 'banana'), "Banana object not found"
        assert hasattr(unwrapped_env, 'cabinet'), "Cabinet object not found"
        assert hasattr(unwrapped_env, 'goal_site'), "Goal site not found"
        print("✓ All required objects exist in the environment")

        # Check banana position
        banana_pos = unwrapped_env.banana.pose.p.cpu().numpy()[0]
        print(f"\nBanana position: {banana_pos}")

        # Check drawer state
        drawer_qpos = unwrapped_env.handle_link.joint.qpos.cpu().numpy()[0]
        print(f"Drawer qpos: {drawer_qpos}")

        # Goal position
        goal_pos = unwrapped_env.goal_site.pose.p.cpu().numpy()[0]
        print(f"Goal position: {goal_pos}")

        # Run a few steps
        print("\nRunning 10 random steps...")
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print(f"  Episode ended at step {i+1}")
                break

        # Check evaluation
        eval_info = unwrapped_env.evaluate()
        print("\nEvaluation info:")
        print(f"  - Success: {eval_info['success'].cpu().numpy()}")
        print(f"  - Is grasped: {eval_info['is_grasped'].cpu().numpy()}")
        print(f"  - At goal: {eval_info['at_goal'].cpu().numpy()}")
        print(f"  - Distance to goal: {eval_info['banana_to_goal_dist'].cpu().numpy()}")

        env.close()
        print("\n" + "=" * 80)
        print("✓ All tests passed!")
        print("=" * 80)
        return True

    except Exception as e:
        print(f"\n✗ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_task_creation()
    exit(0 if success else 1)
