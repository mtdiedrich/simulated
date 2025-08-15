# Simulated Gaming AI

A comprehensive AI simulation platform featuring:
1. **1v1 Soccer Simulation** - Reinforcement learning agents that learn to play soccer
2. **Ecosystem Simulation** - Complex multi-species ecosystem with predator-prey dynamics
3. **Need for Speed: Carbon Bot** - Neural network that learns to play NFS Carbon from human gameplay

## Projects

### 1. Ecosystem Simulation (Multi-Species Environment)

A robust and in-depth ecosystem simulation that handles multiple species, prey-predator relationships, environmental niches, and population dynamics.

#### Features
- **Multiple Species Types**: Plants (producers), herbivores (primary consumers), carnivores (secondary consumers), and omnivores
- **Predator-Prey Relationships**: Complex food webs with realistic hunting and consumption mechanics
- **Environmental Niches**: Four distinct habitats (grassland, forest, water, rocky) with different resource characteristics
- **Population Dynamics**: Birth, death, reproduction, aging, and energy-based survival
- **Spatial Interactions**: Movement, territory, and resource competition in 2D space
- **Resource Management**: Energy systems, carrying capacity, and environmental constraints

#### Species and Interactions
- **Plants** (Grass, Trees): Primary producers that gain energy through photosynthesis
- **Herbivores** (Rabbits, Deer): Consume plants, serve as prey for carnivores
- **Carnivores** (Wolves, Hawks): Hunt herbivores, top predators in the food chain
- **Omnivores** (Bears): Eat both plants and herbivores, adaptive feeders

#### Quick Start
```bash
# Run ecosystem demonstration
python train_ecosystem.py

# Run ecosystem tests
python test_ecosystem.py

# Demo with visualization
python demo_ecosystem.py

# Training analysis
python train_ecosystem.py train
```

### 2. Soccer Simulation (Reinforcement Learning)

A 1v1 soccer environment where two AI agents learn to play soccer through Q-learning reinforcement learning.

### 3. Need for Speed: Carbon Bot (Imitation Learning)

A neural network-based bot that learns to play Need for Speed: Carbon by watching and imitating human gameplay.

## Ecosystem Simulation Overview

This project implements a comprehensive multi-species ecosystem simulation featuring complex predator-prey relationships, environmental niches, and realistic population dynamics. The simulation models interactions between producers (plants), primary consumers (herbivores), secondary consumers (carnivores), and omnivores in a spatially explicit environment.

### Ecosystem Components

#### Environmental System
- **Multiple Niches**: Grassland, forest, water, and rocky habitats with distinct characteristics
- **Resource Distribution**: Variable plant growth rates and water availability across niches
- **Carrying Capacity**: Each niche has limits on organism density
- **Spatial Dynamics**: 2D environment with movement and territorial behavior

#### Species Hierarchy
- **Plants** (Grass, Trees): Autotrophic organisms that convert environmental resources to energy
- **Herbivores** (Rabbits, Deer): Primary consumers that feed on plants
- **Carnivores** (Wolves, Hawks): Secondary consumers that hunt herbivores  
- **Omnivores** (Bears): Adaptive feeders that consume both plants and animals

#### Population Mechanics
- **Energy Systems**: All organisms have energy that affects survival and reproduction
- **Life Cycles**: Birth, growth, reproduction, aging, and death
- **Predation**: Hunting mechanics with vision range and success rates
- **Competition**: Resource competition affects population growth
- **Adaptation**: Species behavior adapts to local conditions

## Soccer Simulation Overview

This project implements both 2D and 3D soccer environments where AI agents learn to play soccer through reinforcement learning. 

### 2D Soccer Simulation
The original 2D simulation uses Q-learning with an epsilon-greedy policy to train agents that can move around a soccer field, chase the ball, and attempt to score goals.

### 3D Soccer Simulation (NEW!)
The enhanced 3D simulation adds full physics simulation with:
- **3D Physics Engine**: Complete collision detection and resolution
- **Realistic Ball Physics**: Gravity, bouncing, air resistance, and spin
- **Player Colliders**: Sphere colliders for players and ball with collision resolution
- **3D Field**: Field boundaries, walls, goal posts, and ground plane
- **Extended Actions**: Jump, power kicks, headers, and 3D movement
- **Advanced AI**: Enhanced state space with 3D coordinates and height awareness

## Features

### 2D Soccer Simulation
- **2D Soccer Field**: Simple rectangular field with goals on each side
- **Physics Simulation**: Basic ball physics with movement, friction, and collision
- **AI Agents**: Two learning agents using Q-learning for decision making
- **Reinforcement Learning**: Agents learn through rewards for ball possession, scoring, and good positioning
- **Training Loop**: Automated training with progress tracking and statistics
- **Visualization**: ASCII-based field visualization to monitor agent behavior

### 3D Soccer Simulation (NEW!)
- **3D Soccer Field**: Full 3D field with boundaries, walls, goal posts, and colliders
- **Advanced Physics**: Complete physics engine with gravity, collision detection, and resolution
- **Realistic Ball Physics**: 3D ball movement with bouncing, air resistance, and realistic trajectories
- **Player Physics**: 3D agent movement with jumping, collision detection, and realistic movement
- **Enhanced AI**: Extended action space (12 actions) and 3D-aware state representation
- **Collision System**: Sphere and box colliders for realistic object interactions
- **3D Visualization**: ASCII visualization with height information and jumping indicators

## Quick Start

### 2D Soccer Simulation
```python
from soccer_simulation import SoccerSimulation

# Create and run 2D simulation
sim = SoccerSimulation()
result = sim.run_episode()
print(f"Episode completed in {result['steps']} steps")
print(f"Final score: {result['final_score']}")
```

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
# Test 2D simulation
python test_simulation.py

# Test 3D simulation
python test_simulation_3d.py

# Quick 3D feature demo
python quick_3d_test.py

# Full interactive demo
python demo_3d_soccer.py
```

## Components

### Core Classes

**2D Simulation:**
- **Vector2D**: 2D vector mathematics for positions and velocities
- **SoccerField**: Field boundaries, goals, and collision detection
- **Ball**: Ball physics with movement, friction, and kicking mechanics
- **Agent**: AI agent with Q-learning, action selection, and reward processing
- **SoccerSimulation**: Main simulation orchestrating the entire game

**3D Simulation (NEW!):**
- **Vector3D**: 3D vector mathematics with dot/cross products and advanced operations
- **Collider**: Base class for collision detection (SphereCollider, BoxCollider)
- **PhysicsEngine**: Complete physics simulation with collision detection and resolution
- **SoccerField3D**: 3D field with boundaries, walls, goal posts, and colliders
- **Ball3D**: 3D ball with gravity, bouncing, air resistance, and realistic physics
- **Agent3D**: 3D agent with jumping, collision detection, and enhanced AI
- **Soccer3DSimulation**: Main 3D simulation with physics integration

### Learning System

**2D Learning:**
- **Q-Learning**: Tabular reinforcement learning for agent decision making
- **State Space**: Discretized positions and game conditions
- **Action Space**: 8-directional movement with automatic ball kicking
- **Reward System**: 
  - +100 for scoring goals
  - -100 for opponent scoring
  - Small rewards for ball possession and good positioning
  - Penalties for being too far from action

**3D Learning (Enhanced):**
- **Q-Learning**: Extended Q-learning with 3D state awareness
- **State Space**: 3D discretized positions, ball height, jumping states, and tactical awareness
- **Action Space**: 12 actions including:
  - 8-directional horizontal movement
  - Jump action for aerial play
  - Brake/stop action
  - Power kick for stronger shots
  - Header/precise kick for aerial balls
- **Enhanced Rewards**:
  - All 2D rewards plus:
  - Rewards for aerial ball control
  - Penalties for excessive jumping
  - Tactical positioning rewards based on ball height

## Need for Speed: Carbon Bot

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Collect training data (5 minutes of gameplay)
python train_nfs_bot.py collect --duration 5

# Train the neural network
python train_nfs_bot.py train --epochs 20

# Run the trained bot
python train_nfs_bot.py run path/to/model.pth

# Interactive example
python nfs_bot_example.py
```

### How It Works

1. **Data Collection**: Records your screen and inputs while playing NFS Carbon
2. **Neural Network Training**: Trains a CNN to map screen captures to actions
3. **Bot Execution**: Uses the trained model to play the game autonomously

### Features

- **Screen Capture**: Real-time capture of game footage
- **Input Monitoring**: Records keyboard and mouse inputs
- **CNN Model**: Deep learning model for action prediction
- **Bot Controller**: Autonomous game playing with input simulation
- **Training Pipeline**: Complete workflow from data to trained bot

### System Requirements

- Python 3.7+
- Need for Speed: Carbon game
- Adequate GPU recommended for training (CPU works but slower)

### File Structure

```
├── train_nfs_bot.py          # Main training script
├── nfs_bot_controller.py     # Bot controller for autonomous play
├── nfs_bot_model.py          # Neural network model
├── data_collector.py         # Data collection system
├── screen_capture.py         # Screen capture utilities
├── input_capture.py          # Input monitoring utilities
├── nfs_bot_example.py        # Interactive examples
├── nfs_training_data/        # Training data storage
└── nfs_models/               # Saved models
```

## Soccer Simulation (Original)

### Running the Simulation

```bash
# Run training and demonstration
python3 train_agents.py

# Run tests
python3 test_simulation.py
```

### Training Parameters

The training can be customized by modifying parameters in `train_agents.py`:

- `num_episodes`: Number of training episodes (default: 50)
- `episode_length`: Steps per episode (default: 1000)
- `learning_rate`: Q-learning rate (default: 0.1)
- `discount_factor`: Future reward discount (default: 0.9)
- `epsilon`: Exploration rate (default: 0.3, decays during training)

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

### Rich Environment ✅ (PARTIALLY IMPLEMENTED)
- ✅ **3D Visualization**: ASCII visualization with height information
- **Visual Interface**: Pygame or similar for real-time 3D visualization
- **Multiple Players**: Extend to 2v2 or larger teams
- **Field Complexity**: Obstacles, different field shapes, or multiple balls

### Advanced Features
- **Team Communication**: Allow agents to share information
- **Strategy Learning**: Higher-level tactical planning
- **Tournament Mode**: Multiple agents competing in brackets
- ✅ **Enhanced State Space**: 3D coordinates and tactical awareness

## Technical Details

### Dependencies
- Python 3.6+ (uses only standard library)
- No external dependencies required for basic functionality

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