import numpy as np
import sapien

from mani_skill.envs.tasks import PickBananaFromOpenDrawerEnv
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)
from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb,
    get_actor_obb,
)


def solve(env: PickBananaFromOpenDrawerEnv, seed=None, debug=False, vis=False):
    """
    Motion planning solution for PickBananaFromOpenDrawer task.

    Steps:
    1. Reset environment (drawer is already open, banana is inside)
    2. Plan grasp for banana
    3. Reach toward banana in drawer
    4. Grasp banana
    5. Extract banana from drawer
    6. Move to goal position
    """
    env.reset(seed=seed)
    assert env.unwrapped.control_mode in [
        "pd_joint_pos",
        "pd_joint_pos_vel",
    ], env.unwrapped.control_mode

    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )

    FINGER_LENGTH = 0.025
    env = env.unwrapped

    # -------------------------------------------------------------------------- #
    # Compute grasp pose for banana
    # -------------------------------------------------------------------------- #
    # Get object oriented bounding box for banana
    obb = get_actor_obb(env.banana)

    # Define grasp approach direction (top-down grasp)
    approaching = np.array([0, 0, -1])

    # Get the closing direction of the gripper
    target_closing = (
        env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    )

    # Compute grasp pose using OBB
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]

    # Build grasp pose
    banana_pos = env.banana.pose.sp.p
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, banana_pos)

    # -------------------------------------------------------------------------- #
    # Pre-reach: Move above drawer opening
    # -------------------------------------------------------------------------- #
    # First move to a safe position above the drawer to avoid collisions
    pre_reach_pose = grasp_pose * sapien.Pose([0, 0, -0.15])

    # Dry run to check if path is valid
    res = planner.move_to_pose_with_screw(pre_reach_pose, dry_run=True)
    if res == -1:
        print("Failed to plan pre-reach motion")
        planner.close()
        return res

    # Execute pre-reach
    planner.open_gripper()
    planner.move_to_pose_with_screw(pre_reach_pose)

    # -------------------------------------------------------------------------- #
    # Reach: Move closer to banana
    # -------------------------------------------------------------------------- #
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])

    res = planner.move_to_pose_with_screw(reach_pose, dry_run=True)
    if res == -1:
        print("Failed to plan reach motion")
        planner.close()
        return res

    planner.move_to_pose_with_screw(reach_pose)

    # -------------------------------------------------------------------------- #
    # Grasp banana
    # -------------------------------------------------------------------------- #
    planner.move_to_pose_with_screw(grasp_pose)
    planner.close_gripper()

    # Small pause to ensure grasp is secure
    for _ in range(5):
        env.step(None)

    # -------------------------------------------------------------------------- #
    # Extract from drawer: Move up to clear drawer
    # -------------------------------------------------------------------------- #
    extraction_pose = grasp_pose * sapien.Pose([0, 0, -0.1])
    planner.move_to_pose_with_screw(extraction_pose)

    # -------------------------------------------------------------------------- #
    # Move to goal pose
    # -------------------------------------------------------------------------- #
    goal_position = env.goal_site.pose.sp.p
    goal_pose = sapien.Pose(goal_position, grasp_pose.q)

    res = planner.move_to_pose_with_screw(goal_pose, dry_run=True)
    if res == -1:
        print("Failed to plan move to goal")
        planner.close()
        return res

    res = planner.move_to_pose_with_screw(goal_pose)

    # -------------------------------------------------------------------------- #
    # Release banana at goal (optional)
    # -------------------------------------------------------------------------- #
    # Uncomment if you want to release the banana at the goal
    # planner.open_gripper()

    planner.close()
    return res
