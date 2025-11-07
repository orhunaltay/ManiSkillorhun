# PickBananaFromOpenDrawerSimple-v1

## Overview

This is a simplified version of the banana picking task, designed for **scripted execution** using end-effector delta position control. Unlike the full motion planning version, this task uses fixed positions and a straightforward state machine for demonstration purposes.

## Task Description

**Goal**: Grasp a YCB banana from an open drawer and place it at a goal position using a Panda robot arm.

**Key Features**:
- Pre-opened drawer (80% open) from PartNet Mobility dataset
- YCB banana object placed inside drawer at fixed position
- Visual goal sphere indicating target placement location
- Designed for `pd_ee_delta_pos` control mode (EE Cartesian deltas)
- Simple state observations (poses only)

## Files

### Task Environment
- **`pick_banana_from_open_drawer_simple.py`** - Main task environment implementation
  - Registered as `PickBananaFromOpenDrawerSimple-v1`
  - Supports Panda and Fetch robots (default: Panda)
  - Fixed cabinet, banana, and goal positions for reproducibility

### Scripted Execution
- **`run_pick_banana_scripted.py`** - Scripted execution with state machine
  - Demonstrates complete pick-and-place sequence
  - Uses small EE position deltas (1 cm per step)
  - State machine with 8 states for smooth execution

## Usage

### Running the Scripted Demo

```bash
python3 run_pick_banana_scripted.py
```

This will:
1. Create the environment with `pd_ee_delta_pos` control
2. Execute a complete pick-and-place sequence
3. Display progress messages for each state
4. Report success/failure at the end

### Creating the Environment Manually

```python
import gymnasium as gym
import pick_banana_from_open_drawer_simple

env = gym.make(
    "PickBananaFromOpenDrawerSimple-v1",
    obs_mode="state_dict",
    render_mode="human",
    control_mode="pd_ee_delta_pos",
    num_envs=1,
)

obs, info = env.reset(seed=0)

# Your control loop here
```

## Control Mode

The environment is designed for **`pd_ee_delta_pos`** control:
- **Action space**: `[dx, dy, dz, gripper]` where:
  - `dx, dy, dz`: End-effector position deltas in meters (world frame)
  - `gripper`: Gripper position command (negative = close, positive = open)
- **Recommended action scale**: ±0.05 (5 cm max per step)

## Scene Layout

### Fixed Positions (for scripted execution)

- **Cabinet**: (0.20, 0.00, [table_height]) - 20cm in front, centered
- **Banana**: (0.15, 0.00, 0.04) - Inside drawer, on drawer bottom
- **Goal**: (-0.20, 0.20, 0.15) - Behind robot, to the side, elevated

### Drawer Configuration

- **Cabinet model**: PartNet Mobility ID 1027
- **Drawer opening**: 80% of full extension
- **Collision bit**: Disabled self-collisions for smooth operation

## State Machine (Scripted Execution)

The `run_pick_banana_scripted.py` follows this sequence:

1. **`go_above_banana`**
   - Move TCP to hover position above banana
   - Gripper: Open
   - Target: (banana_x, banana_y, 0.20)

2. **`descend_to_grasp`**
   - Lower TCP to banana grasp height
   - Gripper: Open
   - Target: (banana_x, banana_y, 0.04)

3. **`close_grip`**
   - Hold position for 30 steps
   - Gripper: Close
   - Ensures stable grasp

4. **`lift_with_banana`**
   - Lift banana out of drawer
   - Gripper: Closed
   - Target: (current_x, current_y, 0.20)

5. **`go_to_goal_xy`**
   - Move laterally to goal XY position
   - Gripper: Closed
   - Target: (goal_x, goal_y, 0.20)

6. **`descend_to_place`**
   - Lower to goal placement height
   - Gripper: Closed
   - Target: (goal_x, goal_y, goal_z)

7. **`open_grip`**
   - Hold position for 30 steps
   - Gripper: Open
   - Releases banana at goal

8. **`retreat`**
   - Lift TCP back to hover height
   - Gripper: Open
   - Target: (current_x, current_y, 0.20)

## Observations

The environment provides:

### Always Available
- `tcp_pose`: End-effector pose `[x, y, z, qw, qx, qy, qz]`

### In `state_dict` mode
- `banana_pose`: Banana pose `[x, y, z, qw, qx, qy, qz]`
- `goal_pose`: Goal site pose `[x, y, z, qw, qx, qy, qz]`
- `drawer_qpos`: Drawer joint position (scalar)

### Robot Proprioception (default)
- Joint positions, velocities
- Gripper state

## Success Criteria

Task is successful when ALL conditions are met:
1. **XY distance**: Banana within 5cm of goal center (XY plane)
2. **Z distance**: Banana within 3cm of goal height
3. **Stability**: Banana linear velocity < 0.1 m/s

## Tunables (in `run_pick_banana_scripted.py`)

```python
HOVER_Z   = 0.20   # Hover height above table
GRASP_Z   = 0.04   # Banana grasp height
PLACE_Z   = 0.15   # Goal placement height
STEP_XYZ  = 0.01   # Movement speed (1 cm/step)
FPS       = 60.0   # Rendering frame rate
HOLD_STEPS = 30    # Steps to hold during grip open/close
OPEN_CMD  = +0.05  # Gripper open command
CLOSE_CMD = -0.05  # Gripper close command
XY_TOL    = 0.015  # XY position tolerance (1.5cm)
Z_TOL     = 0.01   # Z position tolerance (1cm)
```

## Reward Function (for RL)

Dense reward includes:
- **Distance penalty**: -2.0 * distance_to_goal
- **Reaching reward**: -1.0 * ee_to_banana_distance
- **Grasp bonus**: +2.0 when grasping banana
- **Success bonus**: +5.0 when at goal

Normalized to [-1, 1] range.

## Comparison with Full Version

| Feature | Simple Version | Full Version (pick_banana_from_open_drawer.py) |
|---------|----------------|-----------------------------------------------|
| Control Mode | `pd_ee_delta_pos` | `pd_joint_pos`, `pd_joint_pos_vel` |
| Positions | Fixed | Randomized |
| Camera Config | Single viewer | 3 synchronized cameras |
| Observations | Poses only | Full contact forces |
| Motion Planning | Scripted | RRT/Screw motion planner |
| Purpose | Demo/scripting | RL training/MP research |
| Complexity | Simple | Complex |

## Example Output

```
================================================================================
Starting scripted banana picking task
================================================================================
Initial banana position: [0.15 0.   0.04]
Goal position: [-0.2   0.2   0.15]
Initial TCP position: [ 0.    0.    0.45]
================================================================================
✓ Reached above banana at [0.15  0.    0.20]
✓ Reached banana at [0.15  0.    0.04]
✓ Gripper closed
✓ Lifted banana to [0.15  0.    0.20]
✓ Reached above goal at [-0.2   0.2   0.20]
✓ Reached place position at [-0.2   0.2   0.15]
✓ Gripper opened, banana placed
✓ Retreated to [-0.2   0.2   0.20]
================================================================================
Task completed!
Final banana position: [-0.201  0.199  0.148]
Goal position: [-0.2   0.2   0.15]
Distance to goal: 0.0027m
✓✓✓ SUCCESS! ✓✓✓
================================================================================
```

## Troubleshooting

### "Module not found" error
Make sure you import the task module before creating the environment:
```python
import pick_banana_from_open_drawer_simple
```

### Actions too large
The environment expects small EE deltas. Use wrappers:
```python
from gymnasium.wrappers import ClipAction, RescaleAction
env = ClipAction(env)
env = RescaleAction(env, -0.05, 0.05)
```

### Gripper not closing/opening
Ensure gripper commands are within the action space bounds after rescaling:
- Open: +0.05 (after rescaling to [-0.05, 0.05])
- Close: -0.05

### Robot collides with drawer
The scripted execution approaches from above to avoid collisions. If you're implementing custom control:
1. Move to hover height first (z=0.20)
2. Move laterally to banana XY
3. Then descend to grasp height

## Extensions

Possible modifications:
1. **Randomize positions**: Modify `_initialize_episode()` to randomize banana/goal locations
2. **Add obstacles**: Place additional objects in the scene
3. **Vision-based control**: Use RGB-D observations instead of state
4. **Multiple objects**: Pick specific banana from multiple objects
5. **Partial drawer opening**: Start with partially closed drawer, require opening first

## Dependencies

- ManiSkill (with PartNet Mobility and YCB datasets)
- Gymnasium
- NumPy
- Sapien
- PyTorch

## See Also

- **Full version**: `pick_banana_from_open_drawer.py` - For motion planning and RL training
- **Apple task**: `place_apple_on_plate.py` - Similar structure for different objects
- **Motion planning examples**: `mani_skill/examples/motionplanning/panda/solutions/`
