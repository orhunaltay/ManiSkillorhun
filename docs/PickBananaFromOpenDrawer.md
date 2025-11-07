# PickBananaFromOpenDrawer-v1

## Task Description

**PickBananaFromOpenDrawer-v1** is a tabletop manipulation task where the robot must grasp a banana object from an already-opened drawer and move it to a goal position. This task combines elements of confined-space manipulation with object picking using motion planning.

## Key Features

- **Pre-opened Drawer**: The drawer starts in an open state (80% open), eliminating the need for drawer manipulation
- **YCB Banana Object**: Uses a realistic banana model from the YCB dataset
- **Motion Planning Support**: Includes contact force observations for all robot links to support motion planning algorithms
- **Confined Space Manipulation**: Requires careful planning to grasp objects within the drawer without collisions
- **Goal-based Success**: Task is successful when the banana reaches the goal position with low velocity

## Environment Details

### Robot
- **Supported Robots**: Panda arm with parallel-jaw gripper
- **Control Modes**: `pd_joint_pos`, `pd_joint_pos_vel`

### Scene Setup
- **Cabinet**: PartNet Mobility cabinet (model ID: 1027)
- **Drawer**: Prismatic joint, initialized to 80% open
- **Banana**: YCB banana object (ID: 011_banana) placed inside drawer
- **Goal Site**: Green sphere indicating target position (-0.3, 0.0, 0.3)

### Success Criteria
1. Banana is at goal position (within 0.05m threshold)
2. Banana velocity is low (< 0.1 m/s)

### Episode Configuration
- **Max Steps**: 100
- **Cabinet Position**: Randomized within X: [0.15, 0.25], Y: [-0.05, 0.05]
- **Banana Orientation**: Random rotation around vertical axis
- **Required Assets**: `partnet_mobility_cabinet`, `ycb`

## Observation Space

The environment provides:
- Robot joint positions and velocities
- End-effector pose
- Banana pose and velocity
- Goal site position
- Drawer joint position and opening fraction
- Contact forces between robot links and objects (banana, cabinet, goal site)

## Point Cloud Bounds

For point cloud observations:
- **X**: [-0.4, 0.1]
- **Y**: [-0.3, 0.3]
- **Z**: [0.4, 0.8]

## Reward Function

The dense reward includes:
1. **Reaching reward** (0-1): Reward for end-effector approaching banana
2. **Grasping reward** (+2.0): Bonus when robot grasps banana
3. **Goal reward** (0-2): Reward for moving banana toward goal
4. **Success reward** (+5.0): Large bonus for achieving success

Maximum reward per step: ~10.0

## Usage

### Basic Example

```python
import gymnasium as gym

env = gym.make(
    "PickBananaFromOpenDrawer-v1",
    obs_mode="state",
    control_mode="pd_joint_pos",
    render_mode="human",
    camera_width=128,
    camera_height=128,
    distraction_set={},
)

obs, info = env.reset(seed=0)

for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

### Motion Planning Example

A motion planning solution is provided for this task:

```python
from mani_skill.examples.motionplanning.panda.solutions import (
    solvePickBananaFromOpenDrawer,
)

# Create environment
env = gym.make(
    "PickBananaFromOpenDrawer-v1",
    obs_mode="state",
    control_mode="pd_joint_pos",
    camera_width=128,
    camera_height=128,
    distraction_set={},
)

# Run motion planning solution
result = solvePickBananaFromOpenDrawer(env, seed=0, debug=False, vis=False)

if result != -1:
    print("Success!")
else:
    print("Planning failed")
```

## Motion Planning Solution

The provided motion planning solution follows these steps:

1. **Compute Grasp Pose**: Calculate grasp pose using object-oriented bounding box (OBB)
2. **Pre-reach**: Move to safe position above drawer opening
3. **Reach**: Approach banana inside drawer
4. **Grasp**: Close gripper to grasp banana
5. **Extract**: Lift banana out of drawer
6. **Move to Goal**: Transport banana to goal position

The solution uses:
- **Screw motion planning**: For smooth Cartesian space trajectories
- **Collision avoidance**: Built-in collision checking with cabinet and drawer
- **Grasp stability**: Pause after grasping to ensure secure grip

## Files

### Task Implementation
- **Task File**: `mani_skill/envs/tasks/tabletop/pick_banana_from_open_drawer.py`
- **Registration**: Added to `mani_skill/envs/tasks/tabletop/__init__.py`

### Motion Planning
- **Solution**: `mani_skill/examples/motionplanning/panda/solutions/pick_banana_from_open_drawer.py`
- **Registration**: Added to `mani_skill/examples/motionplanning/panda/solutions/__init__.py`

## Related Tasks

- **OpenDrawer-v1**: Opens a drawer (no picking)
- **PickCube-v1**: Picks a cube from table (no drawer/confined space)
- **PickCubeMP-v1**: Picks a cube with obstacles and motion planning support
- **PickSingleYCB-v1**: Picks various YCB objects from table

## Technical Notes

### Drawer Configuration
The drawer is set to 80% open (`drawer_open_frac = 0.8`) at initialization. This is achieved by:
1. Getting joint limits (qmin, qmax) for the prismatic drawer joint
2. Setting qpos = qmin + (qmax - qmin) * 0.8

### Banana Placement
The banana is placed inside the drawer at initialization:
- Position is computed relative to drawer handle position
- Placed slightly behind the handle (0.05m back)
- Centered horizontally with the drawer
- Random rotation around vertical axis for variation

### Contact Forces
The task provides detailed contact force observations between:
- All robot links and banana
- All robot links and cabinet/drawer parts
- All robot links and goal site

These observations are useful for:
- Motion planning collision avoidance
- Grasp stability analysis
- Contact-rich manipulation learning

## Camera Configuration

The task uses three synchronized cameras:
- **Camera Center**: Front view
- **Camera Left**: Left side view
- **Camera Right**: Right side view

All cameras use:
- RealSense depth camera FOV
- Configurable resolution (specified in constructor)
- Synchronized rendering

## Limitations

1. **Single Drawer**: Currently uses a fixed cabinet model (1027)
2. **Fixed Object**: Only banana is supported (could be extended to other YCB objects)
3. **Pre-opened Drawer**: Drawer opening skill is not required/tested
4. **Limited Randomization**: Cabinet position is randomized but drawer and banana placement are deterministic relative to cabinet

## Future Extensions

Possible extensions to this task:
1. **Variable drawer opening**: Require robot to open drawer further if needed
2. **Multiple objects**: Place multiple objects in drawer, pick specific target
3. **Different YCB objects**: Generalize to any YCB object
4. **Multiple drawers**: Random selection of which drawer contains target
5. **Partial observability**: Require visual search to locate object in drawer
