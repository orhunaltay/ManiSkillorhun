# LightBulb-in-Socket Task

A ManiSkill manipulation task where a robotic arm must pick up a lightbulb and insert it into a socket fixture.

## Overview

This task demonstrates:
- **Grasping**: Picking up a lightbulb with a fragile glass bulb
- **Precise alignment**: Aligning the bulb base with the socket opening
- **Insertion**: Carefully inserting the bulb into the socket
- **Custom geometry**: Building custom objects (lightbulb and socket) using primitive shapes

## Files

1. **lightbulb_in_socket.py** - Environment definition
   - Defines the `LightBulbInSocket-v1` task
   - Creates lightbulb geometry (glass sphere + metal base)
   - Creates socket fixture (hollow cylinder on a stand)
   - Implements success criteria and reward function

2. **lightbulb_motion_planning.py** - Motion planning script
   - Demonstrates how to solve the task with scripted motion planning
   - Implements a state machine for: approach → grasp → lift → align → insert → retreat

## Task Description

### Objects

**Lightbulb**:
- Glass bulb: 3cm radius sphere (semi-transparent, warm white)
- Metal base: 1.5cm radius, 5cm tall cylinder
- Total height: ~8cm
- Dynamic object (can be picked up)

**Socket Fixture**:
- Hollow cylinder: 4cm outer radius, 1.8cm inner radius
- Height: 8cm
- Mounted on a base platform (6cm radius)
- Static object (fixed in place)

### Success Conditions

The task is successful when:
1. **XY Alignment**: Bulb center is within 1.5cm of socket center
2. **Insertion Depth**: Bulb base is inserted at least 2cm into socket
3. **Orientation**: Bulb is upright (within ~11° of vertical)

### Randomization

- **Lightbulb**: Spawns at random XY position on table (±10cm), random Z rotation
- **Socket**: Spawns at random XY position (±5cm), 15cm above table, always vertical

## Usage

### Running the Motion Planning Script

```bash
python3 lightbulb_motion_planning.py
```

This will:
1. Create the environment with human rendering enabled
2. Execute a scripted motion plan to complete the task
3. Display the robot's progress in real-time
4. Report success/failure at the end

### Using the Environment in Your Code

```python
import gymnasium as gym
import mani_skill
import lightbulb_in_socket

# Create environment
env = gym.make(
    "LightBulbInSocket-v1",
    obs_mode="state_dict",        # or "rgbd", "rgb", etc.
    render_mode="human",           # or "rgb_array", None
    control_mode="pd_ee_delta_pos",  # or other control modes
    num_envs=1,
)

# Reset and interact
obs, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()  # or your policy
    obs, reward, terminated, truncated, info = env.step(action)

    if info["success"]:
        print("Successfully inserted lightbulb!")
        break
```

### Observations

In `state_dict` mode, you get:
- `tcp_pose`: Robot TCP pose [x, y, z, qw, qx, qy, qz]
- `lightbulb_pose`: Lightbulb pose [x, y, z, qw, qx, qy, qz]
- `socket_pose`: Socket pose [x, y, z, qw, qx, qy, qz]

### Reward Function

The dense reward function includes:
1. **Reaching reward**: Negative distance from TCP to bulb
2. **Approach reward**: Negative XY distance from bulb to socket
3. **Insertion reward**: Proportional to insertion depth
4. **Success bonus**: Large bonus when all conditions are met

## Design Notes

### Why Custom Geometry?

The YCB dataset doesn't include lightbulbs, so this task demonstrates how to create custom objects using ManiSkill's primitive shape builders:
- `add_sphere_collision/visual` for the glass bulb
- `add_cylinder_collision/visual` for the metal base and socket

### Socket Design

The socket is designed as a hollow cylinder to allow insertion:
- Visual: Full cylinder for appearance
- Collision: Only outer walls (4 box collisions arranged in a ring)
- This allows the lightbulb base to physically enter the socket

### Challenges

This task is more challenging than simple pick-and-place because:
1. **Precision**: Requires tight XY alignment (~1.5cm tolerance)
2. **Orientation**: Must maintain bulb upright orientation
3. **Insertion**: Must lower carefully to insert without collision
4. **Two-stage grasping**: Must grasp without crushing the glass bulb

## Comparison to PlaceAppleOnPlate

| Aspect | PlaceAppleOnPlate | LightBulbInSocket |
|--------|-------------------|-------------------|
| Objects | YCB apple & plate | Custom lightbulb & socket |
| Task type | Placement on top | Insertion into |
| Precision | ~9cm radius | ~1.5cm radius |
| Orientation | Not critical | Must be upright |
| Complexity | Beginner | Intermediate |

## Future Extensions

Possible enhancements:
1. **Multi-stage task**: Unscrew old bulb, then insert new one
2. **Force control**: Detect when bulb is fully seated
3. **Visual servoing**: Use camera observations instead of state
4. **Dexterous manipulation**: Use multi-finger hands instead of parallel jaw gripper
5. **Real lightbulb models**: Load actual 3D models of standard bulb sizes (E26, E27, etc.)

## Troubleshooting

**Import Error**: Make sure both files are in the same directory or Python path:
```bash
export PYTHONPATH=$PYTHONPATH:/path/to/ManiSkillorhun
```

**Rendering Issues**: If human rendering doesn't work:
```python
env = gym.make("LightBulbInSocket-v1", render_mode=None)
```

**GPU/CPU**: The environment will automatically use GPU if available. For CPU-only:
```python
env = gym.make("LightBulbInSocket-v1", sim_backend="cpu")
```

## References

- ManiSkill Documentation: https://maniskill.readthedocs.io/
- Example tasks: `mani_skill/envs/tasks/tabletop/`
- Building custom actors: `mani_skill/utils/building/actors/common.py`
