# Simulated Gaming AI

A comprehensive AI simulation platform featuring:
1. **3D Soccer Simulation** - Reinforcement learning agents that learn to play soccer in a 3D environment with physics

## Projects

### 1. 3D Soccer Simulation (Reinforcement Learning)

A 1v1 soccer environment where two AI agents learn to play soccer through Q-learning reinforcement learning in a full 3D environment with realistic physics, collision detection, and 3D model visualization.

## Soccer Simulation Overview

This project implements a 3D soccer environment where AI agents learn to play soccer through reinforcement learning with full 3D physics, collision detection, and realistic 3D model visualization.

### 3D Soccer Simulation with Physics
The simulation features full 3D physics simulation with:
- **3D Physics Engine**: Complete collision detection and resolution
- **Realistic Ball Physics**: Gravity, bouncing, air resistance, and spin  
- **Player Colliders**: Sphere colliders for players and ball with collision resolution
- **3D Field**: Field boundaries, walls, goal posts, and ground plane
- **Extended Actions**: Jump, power kicks, headers, and 3D movement
- **Advanced AI**: Enhanced state space with 3D coordinates and height awareness
- **3D Model Visualization**: Real-time 3D rendering of field, players, and ball

## Features

### 3D Soccer Simulation with Physics
- **3D Soccer Field**: Full 3D field with boundaries, walls, goal posts, and colliders
- **Advanced Physics**: Complete physics engine with gravity, collision detection, and resolution
- **Realistic Ball Physics**: 3D ball movement with bouncing, air resistance, and realistic trajectories
- **Player Physics**: 3D agent movement with jumping, collision detection, and realistic movement
- **Enhanced AI**: Extended action space (12 actions) and 3D-aware state representation
- **Collision System**: Sphere and box colliders for realistic object interactions
- **3D Model Visualization**: Real-time 3D rendering using modern graphics with field, player, and ball models

## Quick Start

### 3D Soccer Simulation
```python
from soccer_simulation_3d import Soccer3DSimulation

# Create and run 3D simulation with physics
sim = Soccer3DSimulation()
result = sim.run_episode()
print(f"Episode completed in {result['steps']} steps")
print(f"Final score: {result['final_score']}")

# Show 3D visualization
print(sim.visualize_ascii_3d())
```

### Running Tests
```bash
# Test 3D simulation
python test_simulation_3d.py

# Quick 3D feature demo
python quick_3d_test.py

# Full interactive demo
python demo_3d_soccer.py
```

## Components

### Core Classes

**3D Simulation:**
- **Vector3D**: 3D vector mathematics with dot/cross products and advanced operations
- **Collider**: Base class for collision detection (SphereCollider, BoxCollider)
- **PhysicsEngine**: Complete physics simulation with collision detection and resolution
- **SoccerField3D**: 3D field with boundaries, walls, goal posts, and colliders
- **Ball3D**: 3D ball with gravity, bouncing, air resistance, and realistic physics
- **Agent3D**: 3D agent with jumping, collision detection, and enhanced AI
- **Soccer3DSimulation**: Main 3D simulation with physics integration

### Learning System

**3D Learning:**
- **Q-Learning**: Extended Q-learning with 3D state awareness
- **State Space**: 3D discretized positions, ball height, jumping states, and tactical awareness
- **Action Space**: 12 actions including:
  - 8-directional horizontal movement
  - Jump action for aerial play
  - Brake/stop action
  - Power kick for stronger shots
  - Header/precise kick for aerial balls
- **Enhanced Rewards**:
  - +100 for scoring goals
  - -100 for opponent scoring
  - Small rewards for ball possession and good positioning
  - Rewards for aerial ball control
  - Penalties for excessive jumping
  - Tactical positioning rewards based on ball height

## Soccer Simulation

### Output

The simulation generates:
- **Console Output**: Real-time training progress and statistics
- **ASCII Visualization**: Current game state with agent and ball positions
- **training_results.json**: Complete training results and statistics

## Example Output

```
Starting 1v1 Soccer Simulation Training
Training for 50 episodes...
--------------------------------------------------
Episode 1/50 | Score: {'left': 0, 'right': 0} | Steps: 1000 | Time: 0.02s
  Agent1 - Reward: 95.4, Q-table: 9 states, ε: 0.285
  Agent2 - Reward: -137.0, Q-table: 7 states, ε: 0.285

Current Game State:
1=Agent1(left), 2=Agent2(right), O=Ball, |=Goals
+----------------------------------------+
||..............1.......................||
||...........O..........................||
||.....................2................||
+----------------------------------------+
Score: Agent1=0, Agent2=2
```

## Extension Ideas

This simulation provides a foundation that can be extended in many ways:

### Enhanced Learning
- **Neural Networks**: Replace Q-tables with deep Q-networks (DQN)
- **Policy Gradient Methods**: Implement PPO or A3C for continuous actions
- **Multi-Agent Learning**: Advanced algorithms for competitive scenarios

### Improved Physics ✅ (IMPLEMENTED in 3D version)
- ✅ **3D Physics Engine**: Complete collision detection and resolution
- ✅ **Advanced Ball Physics**: Gravity, bouncing, air resistance, and realistic trajectories
- ✅ **Player Physics**: 3D movement, jumping, and collision between players
- ✅ **Continuous Actions**: Enhanced action space with jumping and special moves

### Rich Environment ✅ (IMPLEMENTED)
- ✅ **3D Physics**: Complete collision detection and physics simulation
- ✅ **3D Model Visualization**: Real-time 3D rendering with field, player, and ball models
- **Multiple Players**: Extend to 2v2 or larger teams
- **Field Complexity**: Obstacles, different field shapes, or multiple balls

### Advanced Features
- **Team Communication**: Allow agents to share information
- **Strategy Learning**: Higher-level tactical planning
- **Tournament Mode**: Multiple agents competing in brackets
- ✅ **Enhanced State Space**: 3D coordinates and tactical awareness

## Technical Details

### Dependencies
- Python 3.6+
- pygame: For 3D graphics and visualization
- PyOpenGL: For 3D rendering (OpenGL bindings)
- numpy: For mathematical operations (if not already available)

### Architecture
- Modular design with separated concerns
- Object-oriented structure for easy extension
- Comprehensive test suite for validation

### Performance
- Lightweight implementation suitable for experimentation
- Fast training loops for rapid iteration
- Scalable to longer training runs

## Contributing

This simulation is designed to be educational and extensible. Feel free to:
- Add new features or improvements
- Experiment with different learning algorithms
- Create visual interfaces or enhanced physics
- Optimize performance for larger scale training

The codebase is well-documented and modular to support easy modification and extension.