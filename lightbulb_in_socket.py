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
from mani_skill.utils.building.actor_builder import ActorBuilder
import sapien.render


@register_env("LightBulbInSocket-v1", max_episode_steps=100)
class LightBulbInSocketEnv(BaseEnv):
    """Goal: grasp the lightbulb and insert it into the socket fixture.

    The lightbulb has a bulbous glass top and a threaded metal base.
    The socket is a fixture that accepts the bulb base.
    """

    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]

    # Lightbulb dimensions (standard E26/E27 bulb proportions)
    bulb_radius = 0.03          # Glass bulb sphere radius
    base_radius = 0.013         # Metal base radius (E26 ~26mm)
    base_height = 0.027         # Metal base height
    total_height = 0.10         # Total lightbulb height

    # Socket dimensions
    socket_hole_radius = 0.015  # Slightly larger than base for insertion
    socket_depth = 0.04         # How deep the socket is

    # Success thresholds
    insertion_threshold = 0.025  # How deep bulb must be inserted (meters)
    xy_threshold = 0.02          # XY alignment tolerance (meters)

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.6, 0.7, 0.6], target=[0.0, 0.0, 0.35])
        return CameraConfig(
            "render_camera",
            pose=pose,
            width=960,
            height=720,
            fov=np.pi / 3,
            near=0.01,
            far=100.0,
        )

    def __init__(self, *args, robot_uids="panda",
                 num_envs=1, reconfiguration_freq=None,
                 robot_init_qpos_noise=0.02,
                 **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        if reconfiguration_freq is None:
            reconfiguration_freq = 1 if num_envs == 1 else 0

        super().__init__(*args,
                         robot_uids=robot_uids,
                         reconfiguration_freq=reconfiguration_freq,
                         num_envs=num_envs,
                         **kwargs)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        # Create table
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # Build lightbulb - SIMPLIFIED for better physics and grasping
        bulb_builder = self.scene.create_actor_builder()

        # Simple collision: capsule for the whole lightbulb (easier to grasp)
        # Capsule oriented vertically (along Z axis)
        bulb_builder.add_capsule_collision(
            radius=self.base_radius,
            half_length=self.total_height / 2,
            density=300,  # Light enough to pick up easily
        )

        # Visual: Glass bulb (sphere at top)
        bulb_builder.add_sphere_visual(
            radius=self.bulb_radius,
            pose=sapien.Pose(p=[0, 0, self.total_height/2 - self.bulb_radius/2]),
            material=sapien.render.RenderMaterial(
                base_color=[1.0, 0.98, 0.85, 0.85],  # Warm white glass
                metallic=0.0,
                roughness=0.1,
                transmission=0.7,  # Glass-like transparency
            )
        )

        # Visual: Metal base (cylinder at bottom)
        bulb_builder.add_cylinder_visual(
            radius=self.base_radius,
            half_length=self.base_height/2,
            pose=sapien.Pose(p=[0, 0, -self.total_height/2 + self.base_height/2]),
            material=sapien.render.RenderMaterial(
                base_color=[0.75, 0.75, 0.75, 1.0],  # Silver metal
                metallic=0.95,
                roughness=0.3,
            )
        )

        # Neck connecting bulb and base
        bulb_builder.add_cylinder_visual(
            radius=self.base_radius * 0.7,
            half_length=(self.total_height - self.bulb_radius - self.base_height)/2,
            pose=sapien.Pose(p=[0, 0, 0]),
            material=sapien.render.RenderMaterial(
                base_color=[0.85, 0.85, 0.85, 0.9],
                metallic=0.3,
                roughness=0.4,
            )
        )

        bulb_builder.initial_pose = sapien.Pose(p=[0.0, 0.0, 0.60])
        self.lightbulb = bulb_builder.build(name="lightbulb")

        # Build socket - SIMPLIFIED
        socket_builder = self.scene.create_actor_builder()

        # Base platform (cube)
        platform_size = 0.08
        socket_builder.add_box_collision(
            half_size=[platform_size/2, platform_size/2, 0.01],
            pose=sapien.Pose(p=[0, 0, -0.01])
        )
        socket_builder.add_box_visual(
            half_size=[platform_size/2, platform_size/2, 0.01],
            pose=sapien.Pose(p=[0, 0, -0.01]),
            material=sapien.render.RenderMaterial(
                base_color=[0.15, 0.15, 0.15, 1.0],
                metallic=0.7,
                roughness=0.6,
            )
        )

        # Socket body - cylinder with hole in middle
        socket_outer_radius = 0.035
        socket_height = self.socket_depth

        # Visual: outer cylinder
        socket_builder.add_cylinder_visual(
            radius=socket_outer_radius,
            half_length=socket_height/2,
            pose=sapien.Pose(p=[0, 0, socket_height/2]),
            material=sapien.render.RenderMaterial(
                base_color=[0.25, 0.25, 0.25, 1.0],
                metallic=0.8,
                roughness=0.5,
            )
        )

        # Collision: ring of boxes around the outside (leaving center open for insertion)
        # Create 8 thin walls in a circle
        num_walls = 8
        wall_thickness = 0.008
        for i in range(num_walls):
            angle = i * 2 * np.pi / num_walls
            wall_dist = (socket_outer_radius + self.socket_hole_radius) / 2
            x = wall_dist * np.cos(angle)
            y = wall_dist * np.sin(angle)

            # Rotation for radial wall
            from scipy.spatial.transform import Rotation
            rot = Rotation.from_euler('z', angle, degrees=False)
            quat = rot.as_quat()  # [x, y, z, w]
            quat_sapien = [quat[3], quat[0], quat[1], quat[2]]  # sapien uses [w, x, y, z]

            socket_builder.add_box_collision(
                half_size=[wall_thickness/2, (socket_outer_radius - self.socket_hole_radius)/2, socket_height/2],
                pose=sapien.Pose(p=[x, y, socket_height/2], q=quat_sapien)
            )

        socket_builder.initial_pose = sapien.Pose(p=[0.25, 0.0, 0.02])
        self.socket = socket_builder.build_static(name="socket")

        # Goal visualization (invisible marker at insertion target)
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=0.01,
            color=[0, 1, 0, 0.5],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_site)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Randomize socket position on table
            socket_xy = (torch.rand((b, 2)) - 0.5) * 0.10  # ±5 cm
            socket_z = torch.full((b, 1), 0.02)  # Just above table
            socket_p = torch.cat([socket_xy, socket_z], dim=-1)
            q = [1, 0, 0, 0]
            self.socket.set_pose(Pose.create_from_pq(p=socket_p, q=q))

            # Randomize lightbulb position on table
            # Make sure it's standing upright
            bulb_xy = (torch.rand((b, 2)) - 0.5) * 0.20  # ±10 cm
            # Height: half the total height above table to rest it properly
            bulb_z = torch.full((b, 1), self.total_height / 2)
            bulb_p = torch.cat([bulb_xy, bulb_z], dim=-1)
            # Upright orientation (no rotation)
            self.lightbulb.set_pose(Pose.create_from_pq(p=bulb_p, q=q))

            # Set goal visualization at socket top center
            goal_p = socket_p.clone()
            goal_p[:, 2] += self.socket_depth
            self.goal_site.set_pose(Pose.create_from_pq(goal_p))

    def evaluate(self):
        # Get bulb base position (bottom of lightbulb)
        bulb_pos = self.lightbulb.pose.p
        bulb_base_z = bulb_pos[..., 2] - self.total_height / 2

        # Get socket top position
        socket_pos = self.socket.pose.p
        socket_top_z = socket_pos[..., 2] + self.socket_depth

        # Check XY alignment
        xy_dist = torch.linalg.norm(bulb_pos[..., :2] - socket_pos[..., :2], dim=1)
        xy_aligned = xy_dist < self.xy_threshold

        # Check insertion depth (bulb base should be below socket top)
        insertion_depth = socket_top_z - bulb_base_z
        inserted = insertion_depth > self.insertion_threshold

        # Check if being grasped (for intermediate reward)
        is_grasped = self.agent.is_grasping(self.lightbulb)

        # Check if robot is static
        is_robot_static = self.agent.is_static(0.2)

        success = xy_aligned & inserted & is_robot_static

        return {
            "success": success,
            "xy_aligned": xy_aligned,
            "inserted": inserted,
            "is_grasped": is_grasped,
            "is_robot_static": is_robot_static,
            "insertion_depth": insertion_depth,
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            is_grasped=info["is_grasped"],
        )
        if "state" in self.obs_mode:
            obs.update(
                lightbulb_pose=self.lightbulb.pose.raw_pose,
                socket_pose=self.socket.pose.raw_pose,
                tcp_to_bulb_pos=self.lightbulb.pose.p - self.agent.tcp.pose.p,
                bulb_to_socket_pos=self.socket.pose.p - self.lightbulb.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # Stage 1: Reach the lightbulb
        tcp_to_bulb_dist = torch.linalg.norm(
            self.lightbulb.pose.p - self.agent.tcp.pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_bulb_dist)
        reward = reaching_reward

        # Stage 2: Grasp the bulb
        is_grasped = info["is_grasped"]
        reward = reward + is_grasped

        # Stage 3: Move bulb to socket
        bulb_to_socket_dist = torch.linalg.norm(
            self.socket.pose.p - self.lightbulb.pose.p, axis=1
        )
        transport_reward = 1 - torch.tanh(5 * bulb_to_socket_dist)
        reward = reward + transport_reward * is_grasped

        # Stage 4: Insert bulb
        reward = reward + info["inserted"].float() * is_grasped

        # Stage 5: Static (task complete)
        qvel = self.agent.robot.get_qvel()
        if self.robot_uids in ["panda"]:
            qvel = qvel[..., :-2]
        static_reward = 1 - torch.tanh(5 * torch.linalg.norm(qvel, axis=1))
        reward = reward + static_reward * info["inserted"].float()

        # Success bonus
        reward[info["success"]] = 6

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 6
