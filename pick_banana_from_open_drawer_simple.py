import numpy as np
import torch
import sapien
import trimesh
from typing import Dict, Any, Union, List

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.building import actors, articulations
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.agents.robots import Panda, Fetch
from mani_skill.utils.structs import Articulation, Link
from mani_skill.utils import common

CABINET_COLLISION_BIT = 29


@register_env("PickBananaFromOpenDrawerSimple-v1", max_episode_steps=100000)
class PickBananaFromOpenDrawerSimpleEnv(BaseEnv):
    """
    Goal: grasp the YCB banana from an open drawer and place it at a goal position.

    This is a simplified version designed for scripted execution with pd_ee_delta_pos control.
    """

    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]
    handle_types = ["prismatic"]

    # Success thresholds
    goal_radius = 0.05   # 5cm tolerance for banana at goal
    height_tol = 0.03    # 3cm vertical tolerance

    # Drawer opening fraction (0.0 = closed, 1.0 = fully open)
    drawer_open_frac = 0.8

    @property
    def _default_human_render_camera_configs(self):
        # Good 3/4 view of the scene
        pose = sapien_utils.look_at(eye=[0.6, 0.7, 0.6], target=[-0.1, 0.0, 0.35])
        return [
            CameraConfig(
                "viewer_cam",
                pose=pose,
                width=960,
                height=720,
                fov=np.pi / 3,   # ~60°
                near=0.01,
                far=100.0,
            )
        ]

    def __init__(self, *args, robot_uids="panda",
                 num_envs=1, reconfiguration_freq=None,
                 **kwargs):
        # For single-env CPU sim, reconfigure every reset
        if reconfiguration_freq is None:
            reconfiguration_freq = 1 if num_envs == 1 else 0

        self._model_id = 1027  # Cabinet model with suitable drawer

        super().__init__(*args,
                         robot_uids=robot_uids,
                         reconfiguration_freq=reconfiguration_freq,
                         num_envs=num_envs,
                         **kwargs)

    # --- Load agent (spawn at reasonable height) ---
    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    # --- Load scene ---
    def _load_scene(self, options: dict):
        # Table scene
        self.table_scene = TableSceneBuilder(env=self)
        self.table_scene.build()

        # Load cabinet with drawer
        sapien.set_log_level("off")
        self._load_cabinets(self.handle_types)
        sapien.set_log_level("warn")

        # Load YCB banana
        banana_builder = actors.get_actor_builder(self.scene, id="ycb:011_banana")
        banana_builder.initial_pose = sapien.Pose(p=[0.0, 0.0, 0.60])
        self.banana = banana_builder.build(name="banana")

        # Create goal site (visual indicator)
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=0.04,
            color=[0, 1, 0, 0.75],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(p=[-0.3, 0.0, 0.3]),
        )

    def _load_cabinets(self, joint_types: List[str]):
        """Load cabinet from PartNet Mobility dataset"""
        link_ids = [0]  # Use first drawer

        self._cabinets = []
        handle_links: List[List[Link]] = []
        handle_links_meshes: List[List[trimesh.Trimesh]] = []

        cabinet_builder = articulations.get_articulation_builder(
            self.scene, f"partnet-mobility:{self._model_id}"
        )
        cabinet_builder.initial_pose = sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0])
        cabinet = cabinet_builder.build(name=f"cabinet-{self._model_id}")
        self.remove_from_state_dict_registry(cabinet)

        # Disable self collisions for cabinet
        for link in cabinet.links:
            link.set_collision_group_bit(
                group=2, bit_idx=CABINET_COLLISION_BIT, bit=1
            )
        self._cabinets.append(cabinet)
        handle_links.append([])
        handle_links_meshes.append([])

        # Find drawer links
        for link, joint in zip(cabinet.links, cabinet.joints):
            if joint.type[0] in joint_types:
                handle_links[-1].append(link)
                handle_links_meshes[-1].append(
                    link.generate_mesh(
                        filter=lambda _, render_shape: "handle" in render_shape.name,
                        mesh_name="handle",
                    )[0]
                )

        # Merge articulations
        self.cabinet = Articulation.merge(self._cabinets, name="cabinet")
        self.add_to_state_dict_registry(self.cabinet)
        self.handle_link = Link.merge(
            [links[link_ids[i] % len(links)] for i, links in enumerate(handle_links)],
            name="handle_link",
        )
        self.handle_link_pos = common.to_tensor(
            np.array(
                [
                    meshes[link_ids[i] % len(meshes)].bounding_box.center_mass
                    for i, meshes in enumerate(handle_links_meshes)
                ]
            ),
            device=self.device,
        )

    def _after_reconfigure(self, options):
        """Setup after GPU initialization"""
        self.cabinet_zs = []
        for cabinet in self._cabinets:
            collision_mesh = cabinet.get_first_collision_mesh()
            self.cabinet_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.cabinet_zs = common.to_tensor(self.cabinet_zs, device=self.device)

    # --- Episode initialization ---
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)

            # Initialize robot with a good starting pose closer to the banana
            # qpos for panda: 7 arm joints + 2 gripper joints
            qpos_0 = np.array([
                0.0,      # Joint 1
                -0.785,   # Joint 2 (slight down angle)
                0.0,      # Joint 3
                -2.356,   # Joint 4 (elbow bent)
                0.0,      # Joint 5
                1.571,    # Joint 6 (wrist up)
                0.785,    # Joint 7 (wrist rotate)
                0.04,     # Gripper finger 1 (open)
                0.04,     # Gripper finger 2 (open)
            ])
            self.table_scene.initialize(env_idx, qpos_0=qpos_0)

            # Position cabinet on table (fixed position for scripted execution)
            # Cabinet placed on the side of the table for easy access
            xy = torch.zeros((b, 3))
            xy[:, 0] = 0.10  # 10cm from robot
            xy[:, 1] = 0.25  # To the side
            xy[:, 2] = self.cabinet_zs[env_idx]
            self.cabinet.set_pose(Pose.create_from_pq(p=xy))

            # Lock the cabinet root so it can't be pushed around
            # Set very high mass for the root link to make it unmovable
            for cabinet in self._cabinets:
                # Make root link very heavy (10000kg) to prevent pushing
                root_link = cabinet.get_links()[0]  # Get root link
                root_link.set_mass(10000.0)

            # Open the drawer to the target position
            # Get joint limits [b, num_joints, 2] where [:,:,0] is min and [:,:,1] is max
            qlimits = self.cabinet.get_qlimits()
            qmin = qlimits[env_idx, :, 0]
            qmax = qlimits[env_idx, :, 1]

            # Set drawer to 80% open (interpolate between min and max)
            # This opens all prismatic/revolute joints by 80%
            qpos = qmin + (qmax - qmin) * self.drawer_open_frac

            self.cabinet.set_qpos(qpos)
            self.cabinet.set_qvel(self.cabinet.qpos[env_idx] * 0)

            # GPU sim stabilization
            if self.gpu_sim_enabled:
                self.scene._gpu_apply_all()
                self.scene.px.gpu_update_articulation_kinematics()
                self.scene.px.step()
                self.scene._gpu_fetch_all()

            # Place banana inside the drawer (fixed position for scripted execution)
            # Banana position relative to cabinet
            banana_pos = torch.zeros((b, 3))
            banana_pos[:, 0] = 0.05   # In front, inside drawer
            banana_pos[:, 1] = 0.25   # Same Y as cabinet
            banana_pos[:, 2] = 0.08   # On drawer bottom, elevated (will settle to ~0.06)
            q = [1, 0, 0, 0]
            self.banana.set_pose(Pose.create_from_pq(p=banana_pos, q=q))

            # Set goal position (fixed for scripted execution)
            # Goal position: on the table, away from drawer
            goal_pos = torch.zeros((b, 3))
            goal_pos[:, 0] = 0.10    # Same X as cabinet
            goal_pos[:, 1] = -0.20   # Opposite side of table
            goal_pos[:, 2] = 0.10    # On table surface
            self.goal_site.set_pose(Pose.create_from_pq(p=goal_pos, q=q))

    def _after_control_step(self):
        """Update kinematics after each control step"""
        if self.gpu_sim_enabled:
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene._gpu_fetch_all()
            self.scene._gpu_apply_all()

    # --- Success evaluation ---
    def evaluate(self):
        # Check if banana is at goal position
        banana_xy = self.banana.pose.p[..., :2]
        goal_xy = self.goal_site.pose.p[..., :2]
        xy_ok = torch.linalg.norm(banana_xy - goal_xy, dim=1) < self.goal_radius

        # Check if banana height is close to goal height
        banana_z = self.banana.pose.p[..., 2]
        goal_z = self.goal_site.pose.p[..., 2]
        z_ok = torch.abs(banana_z - goal_z) < self.height_tol

        # Check if banana has low velocity (stable)
        banana_vel = self.banana.linear_velocity
        banana_stable = torch.linalg.norm(banana_vel, dim=1) < 0.1

        success = xy_ok & z_ok & banana_stable

        return {
            "success": success,
            "xy_ok": xy_ok,
            "z_ok": z_ok,
            "banana_stable": banana_stable,
        }

    # --- Extra observations for scripted execution ---
    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,  # [x,y,z,qw,qx,qy,qz]
        )
        if self.obs_mode_struct.use_state:
            obs.update(
                banana_pose=self.banana.pose.raw_pose,
                goal_pose=self.goal_site.pose.raw_pose,
                drawer_qpos=self.handle_link.joint.qpos,
            )
        return obs

    # --- Dense reward (optional, for RL) ---
    def compute_normalized_dense_reward(self, obs: Any, action: np.ndarray, info: Dict):
        # Reward for moving banana toward goal
        banana_pos = self.banana.pose.p
        goal_pos = self.goal_site.pose.p
        dist = torch.linalg.norm(banana_pos - goal_pos, dim=1)

        # Reward for end-effector near banana (if not grasped yet)
        tcp_pos = self.agent.tcp.pose.p
        ee_to_banana = torch.linalg.norm(tcp_pos - banana_pos, dim=1)

        # Check if grasping
        is_grasped = self.agent.is_grasping(self.banana)

        # Combined reward
        reward = -2.0 * dist - 1.0 * ee_to_banana
        reward = reward + 2.0 * is_grasped  # Bonus for grasping
        reward = reward + 5.0 * (dist < 0.05)  # Big bonus for success

        return torch.clamp(reward / 5.0, -1.0, 1.0)
