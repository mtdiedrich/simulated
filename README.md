# 1v1 Soccer Simulation

A reinforcement learning simulation where two agents learn to play 1v1 soccer against each other.

## Overview

This project implements a simple soccer environment where two AI agents learn to play soccer through reinforcement learning. The simulation uses Q-learning with an epsilon-greedy policy to train agents that can move around a soccer field, chase the ball, and attempt to score goals.

## Features

- **2D Soccer Field**: Simple rectangular field with goals on each side
- **Physics Simulation**: Basic ball physics with movement, friction, and collision
- **AI Agents**: Two learning agents using Q-learning for decision making
- **Reinforcement Learning**: Agents learn through rewards for ball possession, scoring, and good positioning
- **Training Loop**: Automated training with progress tracking and statistics
- **Visualization**: ASCII-based field visualization to monitor agent behavior

## Components

### Core Classes

- **Vector2D**: 2D vector mathematics for positions and velocities
- **SoccerField**: Field boundaries, goals, and collision detection
- **Ball**: Ball physics with movement, friction, and kicking mechanics
- **Agent**: AI agent with Q-learning, action selection, and reward processing
- **SoccerSimulation**: Main simulation orchestrating the entire game

### Learning System

- **Q-Learning**: Tabular reinforcement learning for agent decision making
- **State Space**: Discretized positions and game conditions
- **Action Space**: 8-directional movement with automatic ball kicking
- **Reward System**: 
  - +100 for scoring goals
  - -100 for opponent scoring
  - Small rewards for ball possession and good positioning
  - Penalties for being too far from action

## Usage

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

### Improved Physics
- **Continuous Actions**: Replace discrete movement with continuous control
- **Advanced Ball Physics**: Spin, bounce, and more realistic movement
- **Player Physics**: Acceleration, momentum, and collision between players

### Rich Environment
- **Visual Interface**: Pygame or similar for real-time visualization
- **Multiple Players**: Extend to 2v2 or larger teams
- **Field Complexity**: Obstacles, different field shapes, or multiple balls

### Advanced Features
- **Team Communication**: Allow agents to share information
- **Strategy Learning**: Higher-level tactical planning
- **Tournament Mode**: Multiple agents competing in brackets

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