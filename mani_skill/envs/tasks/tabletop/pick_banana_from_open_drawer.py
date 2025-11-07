from typing import Any, Dict, List, Optional, Union

import numpy as np
import sapien
import sapien.physx as physx
import torch
import trimesh

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors, articulations
from mani_skill.utils.building.actors.ycb import get_ycb_builder
from mani_skill.utils.geometry.geometry import transform_points
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Articulation, Link, Pose, Actor
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.envs.tasks.tabletop.get_camera_config import (
    get_human_render_camera_config,
    get_camera_configs,
    REALSENSE_DEPTH_FOV_VERTICAL_RAD,
    SHADER,
)
from mani_skill.envs.distraction_set import DistractionSet

CABINET_COLLISION_BIT = 29


@register_env(
    "PickBananaFromOpenDrawer-v1",
    asset_download_ids=["partnet_mobility_cabinet", "ycb"],
    max_episode_steps=100,
)
class PickBananaFromOpenDrawerEnv(BaseEnv):
    """
    **Task Description:**
    Pick a banana from an already-open drawer using the Panda robot.
    The drawer is pre-opened and the task focuses on grasping and extracting
    the banana object from the drawer using motion planning.

    **Key Features:**
    - Cabinet with pre-opened drawer from PartNet Mobility dataset
    - YCB banana object placed inside the drawer
    - Motion planning support with contact force observations
    - Goal position above and away from the drawer
    - Success criteria: banana grasped and moved to goal position

    **Appropriate pointcloud bounds:**
    x: [-0.4, 0.1]
    y: [-0.3, 0.3]
    z: [0.4, 0.8]
    """

    SUPPORTED_ROBOTS = ["panda"]
    agent: Union[Panda]
    handle_types = ["prismatic"]
    TRAIN_JSON = (
        PACKAGE_ASSET_DIR / "partnet_mobility/meta/info_cabinet_drawer_train.json"
    )
    CABINET_X_LIMS = [0.15, 0.25]
    CABINET_Y_LIMS = [-0.05, 0.05]

    # Drawer should be opened to this fraction (0.0 = closed, 1.0 = fully open)
    drawer_open_frac = 0.8

    def __init__(
        self,
        *args,
        robot_uids="panda",
        robot_init_qpos_noise=0.02,
        **kwargs,
    ):
        assert "camera_width" in kwargs and "camera_height" in kwargs, (
            "camera_width and camera_height must be provided"
        )
        assert "distraction_set" in kwargs, "distraction_set must be provided"
        self._camera_width = kwargs.pop("camera_width")
        self._camera_height = kwargs.pop("camera_height")
        self._distraction_set = kwargs.pop("distraction_set")
        if isinstance(self._distraction_set, dict):
            self._distraction_set = DistractionSet(**self._distraction_set)

        self.robot_init_qpos_noise = robot_init_qpos_noise
        self._model_id = 1027  # Cabinet model with suitable drawer

        # Goal configuration - position where banana should be moved to
        self._goal_radius = 0.04
        self._goal_thresh = 0.05  # Distance threshold for success

        super().__init__(
            *args,
            robot_uids=robot_uids,
            **kwargs,
        )
        self._human_render_shader = kwargs.pop("human_render_shader", None)

    @property
    def _default_human_render_camera_configs(self):
        return get_human_render_camera_config(
            eye=[-0.2, 0.5, 1.1], target=[-0.1, 0, 0.5], shader=self._human_render_shader
        )

    @property
    def _default_sensor_configs(self):
        target = [-0.1, 0, 0.0]
        eye_xy = 0.75
        eye_z = 0.75
        cfgs = get_camera_configs(
            eye_xy, eye_z, target, self._camera_width, self._camera_height
        )
        cfgs_adjusted = self._distraction_set.update_camera_configs(cfgs)
        return cfgs_adjusted

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # Load cabinet with drawer
        sapien.set_log_level("off")
        self._load_cabinets(self.handle_types)
        sapien.set_log_level("warn")

        # Load banana object from YCB dataset
        builder = get_ycb_builder(self.scene, id="011_banana")
        builder.initial_pose = sapien.Pose(p=[0.0, 0.0, 0.0])
        self.banana = builder.build(name="banana")

        # Create goal site (visual indicator for where to move banana)
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self._goal_radius,
            color=[0, 1, 0, 0.75],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(p=[-0.3, 0.0, 0.3]),
        )
        self._hidden_objects.append(self.goal_site)

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

        # Get drawer joint limits
        target_qlimits = self.handle_link.joint.limits  # [b, 1, 2]
        qmin, qmax = target_qlimits[..., 0], target_qlimits[..., 1]
        self.drawer_open_qpos = qmin + (qmax - qmin) * self.drawer_open_frac

    def handle_link_positions(self, env_idx: Optional[torch.Tensor] = None):
        """Get world position of drawer handle"""
        if env_idx is None:
            return transform_points(
                self.handle_link.pose.to_transformation_matrix().clone(),
                common.to_tensor(self.handle_link_pos, device=self.device),
            )
        return transform_points(
            self.handle_link.pose[env_idx].to_transformation_matrix().clone(),
            common.to_tensor(self.handle_link_pos[env_idx], device=self.device),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)

            # Position cabinet on table
            xy = torch.zeros((b, 3))
            cabinet_x_range = self.CABINET_X_LIMS[1] - self.CABINET_X_LIMS[0]
            cabinet_y_range = self.CABINET_Y_LIMS[1] - self.CABINET_Y_LIMS[0]
            xy[:, 0] = torch.rand(b) * cabinet_x_range + self.CABINET_X_LIMS[0]
            xy[:, 1] = torch.rand(b) * cabinet_y_range + self.CABINET_Y_LIMS[0]
            xy[:, 2] = self.cabinet_zs[env_idx]
            self.cabinet.set_pose(Pose.create_from_pq(p=xy))

            # Initialize robot
            qpos_0 = np.array(
                [
                    -0.13595445,
                    -1.2611351,
                    0.24094589,
                    -2.9000182,
                    2.5728698,
                    3.0259767,
                    0.029944034,
                    0.039999813,
                    0.03999985,
                ]
            )  # Gripper starts open
            self.table_scene.initialize(
                env_idx, table_z_rotation_angle=np.pi, qpos_0=qpos_0
            )

            # Open the drawer to the target position
            qlimits = self.cabinet.get_qlimits()
            qpos = qlimits[env_idx, :, 0].clone()
            # Set the drawer joint to open position
            qpos[:, self.handle_link.joint.active_joint_idx[0]] = self.drawer_open_qpos[env_idx].squeeze()
            self.cabinet.set_qpos(qpos)
            self.cabinet.set_qvel(self.cabinet.qpos[env_idx] * 0)

            # GPU sim stabilization
            if self.gpu_sim_enabled:
                self.scene._gpu_apply_all()
                self.scene.px.gpu_update_articulation_kinematics()
                self.scene.px.step()
                self.scene._gpu_fetch_all()

            # Place banana inside the drawer
            # Get drawer position and place banana inside
            handle_pos = self.handle_link_positions(env_idx)
            banana_pos = torch.zeros((b, 3))
            # Place banana slightly inside the drawer, centered
            banana_pos[:, 0] = handle_pos[:, 0] - 0.05  # Slightly back from handle
            banana_pos[:, 1] = handle_pos[:, 1]
            banana_pos[:, 2] = handle_pos[:, 2]

            banana_qs = randomization.random_quaternions(
                b, lock_x=True, lock_y=True, device=self.device
            )
            self.banana.set_pose(Pose.create_from_pq(banana_pos, banana_qs))

            # Set goal position (above and in front of drawer)
            goal_pos = torch.zeros((b, 3))
            goal_pos[:, 0] = -0.3
            goal_pos[:, 1] = 0.0
            goal_pos[:, 2] = 0.3
            self.goal_site.set_pose(Pose.create_from_pq(p=goal_pos))

    def _after_control_step(self):
        """Update kinematics after each control step"""
        if self.gpu_sim_enabled:
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene._gpu_fetch_all()
            self.scene._gpu_apply_all()

    def evaluate(self):
        """Evaluate success conditions"""
        # Check if banana is grasped
        is_grasped = self.agent.is_grasping(self.banana)

        # Check if banana is at goal position
        banana_pos = self.banana.pose.p
        goal_pos = self.goal_site.pose.p
        banana_to_goal_dist = torch.linalg.norm(banana_pos - goal_pos, axis=1)
        at_goal = banana_to_goal_dist <= self._goal_thresh

        # Check if banana has low velocity (stable)
        banana_vel = self.banana.linear_velocity
        banana_stable = torch.linalg.norm(banana_vel, axis=1) <= 0.1

        success = at_goal & banana_stable

        return {
            "success": success,
            "is_grasped": is_grasped,
            "at_goal": at_goal,
            "banana_to_goal_dist": banana_to_goal_dist,
            "banana_pos": banana_pos,
            "goal_pos": goal_pos,
        }

    def compute_dense_reward(self):
        """Compute dense reward for training"""
        # Get evaluation info
        info = self.evaluate()

        reward = torch.zeros(self.num_envs, device=self.device)

        # Reward for getting close to banana
        banana_pos = self.banana.pose.p
        ee_pos = self.agent.tcp.pose.p
        ee_to_banana_dist = torch.linalg.norm(ee_pos - banana_pos, axis=1)
        reaching_reward = 1.0 - torch.tanh(5.0 * ee_to_banana_dist)
        reward += reaching_reward

        # Reward for grasping
        if info["is_grasped"].any():
            reward[info["is_grasped"]] += 2.0

        # Reward for moving banana toward goal
        banana_to_goal_dist = info["banana_to_goal_dist"]
        goal_reward = 1.0 - torch.tanh(5.0 * banana_to_goal_dist)
        reward += goal_reward * 2.0

        # Large reward for success
        reward[info["success"]] += 5.0

        return reward

    def compute_normalized_dense_reward(self):
        """Normalized version of dense reward"""
        return self.compute_dense_reward() / 10.0

    def _get_obs_extra(self, info: dict):
        """Additional observations including contact forces for motion planning"""
        obs = {}

        # Contact forces between robot links and banana
        def entry_name(link_name, obj_name):
            return f"contact-force:{link_name}___{obj_name}"

        for link_name, link in self.agent.robot.links_map.items():
            # Contact with banana
            forces = self.scene.get_pairwise_contact_forces(link, self.banana)
            obs[entry_name(link_name, "banana")] = forces

            # Contact with goal site
            forces = self.scene.get_pairwise_contact_forces(link, self.goal_site)
            obs[entry_name(link_name, "goal_site")] = forces

            # Contact with cabinet
            for cabinet in self._cabinets:
                for cabinet_link in cabinet.links:
                    forces = self.scene.get_pairwise_contact_forces(link, cabinet_link)
                    obs[entry_name(link_name, f"cabinet_{cabinet_link.name}")] = forces

        # Add drawer state info
        obs["drawer_qpos"] = self.handle_link.joint.qpos
        obs["drawer_open_fraction"] = (
            self.handle_link.joint.qpos - self.handle_link.joint.limits[..., 0]
        ) / (
            self.handle_link.joint.limits[..., 1] - self.handle_link.joint.limits[..., 0]
        )

        return obs
