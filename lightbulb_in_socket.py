import numpy as np
import torch
import sapien
from typing import Dict, Any, Union
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.building import actors
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.agents.robots import Panda, Fetch
from transforms3d.euler import euler2quat


@register_env("LightBulbInSocket-v1", max_episode_steps=100000)
class LightBulbInSocketEnv(BaseEnv):
    """Goal: grasp the lightbulb and insert it into the socket fixture."""

    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]

    # success threshold (meters and radians)
    socket_radius = 0.025    # socket hole radius ~2.5cm
    insertion_depth = 0.04   # how deep the bulb base needs to be in socket
    position_tol = 0.015     # position tolerance for alignment
    rotation_tol = 0.2       # rotation tolerance (radians) ~11 degrees

    @property
    def _default_human_render_camera_configs(self):
        # A good 3/4 view of the table + arm
        pose = sapien_utils.look_at(eye=[0.6, 0.7, 0.6], target=[0.0, 0.0, 0.35])
        return [
            CameraConfig(
                "viewer_cam",
                pose=pose,
                width=960,
                height=720,
                fov=np.pi / 3,
                near=0.01,
                far=100.0,
            )
        ]

    def __init__(self, *args, robot_uids="panda",
                 num_envs=1, reconfiguration_freq=None,
                 **kwargs):
        if reconfiguration_freq is None:
            reconfiguration_freq = 1 if num_envs == 1 else 0

        super().__init__(*args,
                         robot_uids=robot_uids,
                         reconfiguration_freq=reconfiguration_freq,
                         num_envs=num_envs,
                         **kwargs)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[0, 0, 1.0]))

    def _load_scene(self, options: dict):
        # Create floor + table
        self.table_scene = TableSceneBuilder(env=self)
        self.table_scene.build()

        # Build lightbulb (sphere bulb + cylindrical base)
        # The lightbulb is composed of two parts for visual/collision
        bulb_builder = self.scene.create_actor_builder()

        # Glass bulb part (sphere)
        bulb_builder.add_sphere_collision(radius=0.03)
        bulb_builder.add_sphere_visual(
            radius=0.03,
            material=sapien.render.RenderMaterial(
                base_color=[1.0, 0.95, 0.8, 0.7],  # warm white, semi-transparent
                metallic=0.1,
                roughness=0.2,
            )
        )

        # Metal base/screw part (cylinder)
        bulb_builder.add_cylinder_collision(
            radius=0.015,
            half_length=0.025,
            pose=sapien.Pose(p=[0, 0, -0.055])  # below the bulb sphere
        )
        bulb_builder.add_cylinder_visual(
            radius=0.015,
            half_length=0.025,
            material=sapien.render.RenderMaterial(
                base_color=[0.8, 0.8, 0.8, 1.0],  # metallic silver
                metallic=0.9,
                roughness=0.3,
            ),
            pose=sapien.Pose(p=[0, 0, -0.055])
        )

        bulb_builder.initial_pose = sapien.Pose(p=[0.0, 0.0, 0.60])
        self.lightbulb = bulb_builder.build(name="lightbulb")

        # Build socket fixture (static, mounted on a stand)
        # Socket is a vertical cylinder with a hole to insert the bulb
        socket_builder = self.scene.create_actor_builder()

        # Socket body - hollow cylinder (we'll use box collision for simplicity)
        # Outer wall
        socket_height = 0.08
        socket_outer_radius = 0.04
        socket_inner_radius = 0.018  # slightly larger than bulb base

        # Create socket as a compound shape
        # Base platform
        socket_builder.add_cylinder_collision(
            radius=0.06,
            half_length=0.01,
            pose=sapien.Pose(p=[0, 0, -socket_height/2 - 0.01])
        )
        socket_builder.add_cylinder_visual(
            radius=0.06,
            half_length=0.01,
            material=sapien.render.RenderMaterial(
                base_color=[0.2, 0.2, 0.2, 1.0],  # dark gray
                metallic=0.5,
                roughness=0.7,
            ),
            pose=sapien.Pose(p=[0, 0, -socket_height/2 - 0.01])
        )

        # Socket cylinder (visual only, collision would block insertion)
        socket_builder.add_cylinder_visual(
            radius=socket_outer_radius,
            half_length=socket_height/2,
            material=sapien.render.RenderMaterial(
                base_color=[0.3, 0.3, 0.3, 1.0],
                metallic=0.6,
                roughness=0.6,
            ),
        )

        # Add collision walls around the socket (not in the center hole)
        # We'll add 4 box collisions around the perimeter to create a hollow center
        wall_thickness = 0.015
        for i in range(4):
            angle = i * np.pi / 2
            offset_x = (socket_outer_radius - wall_thickness/2) * np.cos(angle)
            offset_y = (socket_outer_radius - wall_thickness/2) * np.sin(angle)
            socket_builder.add_box_collision(
                half_size=[wall_thickness/2, socket_outer_radius, socket_height/2],
                pose=sapien.Pose(p=[offset_x, offset_y, 0], q=euler2quat(0, 0, angle))
            )

        socket_builder.initial_pose = sapien.Pose(p=[0.25, 0.0, 0.15])
        self.socket = socket_builder.build_static(name="socket")

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Randomize socket position slightly
            socket_xy = (torch.rand((b, 2)) - 0.5) * 0.10  # ±5 cm on table
            socket_z = torch.full((b, 1), 0.15)  # 15cm above table surface
            socket_p = torch.cat([socket_xy, socket_z], dim=-1)
            # Socket is vertical (no rotation)
            q = [1, 0, 0, 0]
            self.socket.set_pose(Pose.create_from_pq(p=socket_p, q=q))

            # Randomize lightbulb position on table
            bulb_xy = (torch.rand((b, 2)) - 0.5) * 0.20  # ±10 cm
            bulb_z = torch.full((b, 1), 0.08)  # on table surface
            bulb_p = torch.cat([bulb_xy, bulb_z], dim=-1)
            # Random rotation around Z axis (upright)
            angles = torch.rand((b,)) * 2 * np.pi
            bulb_q = torch.zeros((b, 4))
            bulb_q[:, 0] = torch.cos(angles / 2)  # w
            bulb_q[:, 3] = torch.sin(angles / 2)  # z
            self.lightbulb.set_pose(Pose.create_from_pq(p=bulb_p, q=bulb_q))

    def evaluate(self):
        # Check if bulb base is aligned with socket opening and inserted
        bulb_pos = self.lightbulb.pose.p
        socket_pos = self.socket.pose.p

        # XY alignment (bulb centered over socket)
        xy_dist = torch.linalg.norm(bulb_pos[..., :2] - socket_pos[..., :2], dim=1)
        xy_aligned = xy_dist < self.position_tol

        # Z alignment (bulb inserted into socket)
        # Bulb base should be at or below socket top
        bulb_base_z = bulb_pos[..., 2] - 0.055  # subtract bulb base offset
        socket_top_z = socket_pos[..., 2] + 0.04  # socket top
        z_inserted = (socket_top_z - bulb_base_z) > self.insertion_depth * 0.5

        # Orientation check (bulb should be roughly vertical/upright)
        # Check z-component of up vector (should be close to 1 for upright)
        bulb_rot_mat = self.lightbulb.pose.to_transformation_matrix()
        z_axis = bulb_rot_mat[..., :3, 2]  # z column of rotation matrix
        upright = torch.abs(z_axis[..., 2]) > torch.cos(torch.tensor(self.rotation_tol))

        success = xy_aligned & z_inserted & upright

        return {
            "success": success,
            "xy_aligned": xy_aligned,
            "z_inserted": z_inserted,
            "upright": upright,
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
        if self.obs_mode_struct.use_state:
            obs.update(
                lightbulb_pose=self.lightbulb.pose.raw_pose,
                socket_pose=self.socket.pose.raw_pose,
            )
        return obs

    def compute_normalized_dense_reward(self, obs: Any, action: np.ndarray, info: Dict):
        # Reward for approaching bulb, grasping, moving to socket, and inserting
        tcp_pos = self.agent.tcp.pose.p
        bulb_pos = self.lightbulb.pose.p
        socket_pos = self.socket.pose.p

        # Stage 1: Reach bulb
        tcp_to_bulb = torch.linalg.norm(tcp_pos - bulb_pos, dim=1)
        reach_reward = -2.0 * tcp_to_bulb

        # Stage 2: Move bulb to socket
        bulb_to_socket_xy = torch.linalg.norm(bulb_pos[..., :2] - socket_pos[..., :2], dim=1)
        approach_reward = -3.0 * bulb_to_socket_xy

        # Stage 3: Insert bulb (vertical alignment)
        bulb_base_z = bulb_pos[..., 2] - 0.055
        socket_top_z = socket_pos[..., 2] + 0.04
        insertion_depth_current = socket_top_z - bulb_base_z
        insertion_reward = 2.0 * torch.clamp(insertion_depth_current / self.insertion_depth, 0, 1)

        # Bonus for success
        success_bonus = 5.0 * info["success"].float()

        total_reward = reach_reward + approach_reward + insertion_reward + success_bonus

        # Normalize to roughly [-1, 1] range
        return torch.clamp(total_reward / 10.0, -1.0, 1.0)
