"""
1v1 Soccer Simulation Environment

A simple soccer simulation where two agents learn to play against each other.
This implementation uses only Python standard library for maximum compatibility.
"""

import math
import random
import json
from typing import Dict, List, Tuple, Optional, Any


class Vector2D:
    """Simple 2D vector class for positions and velocities."""
    
    def __init__(self, x: float = 0.0, y: float = 0.0):
        self.x = x
        self.y = y
    
    def __add__(self, other: 'Vector2D') -> 'Vector2D':
        return Vector2D(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other: 'Vector2D') -> 'Vector2D':
        return Vector2D(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar: float) -> 'Vector2D':
        return Vector2D(self.x * scalar, self.y * scalar)
    
    def magnitude(self) -> float:
        return math.sqrt(self.x ** 2 + self.y ** 2)
    
    def normalize(self) -> 'Vector2D':
        mag = self.magnitude()
        if mag == 0:
            return Vector2D(0, 0)
        return Vector2D(self.x / mag, self.y / mag)
    
    def distance_to(self, other: 'Vector2D') -> float:
        return (self - other).magnitude()
    
    def copy(self) -> 'Vector2D':
        return Vector2D(self.x, self.y)
    
    def __repr__(self) -> str:
        return f"Vector2D({self.x:.2f}, {self.y:.2f})"


class SoccerField:
    """Represents the soccer field boundaries and goals."""
    
    def __init__(self, width: float = 100.0, height: float = 60.0):
        self.width = width
        self.height = height
        self.goal_width = 20.0
        self.goal_height = 2.0
        
        # Goal positions
        self.left_goal = {
            'x': 0,
            'y_min': (height - self.goal_width) / 2,
            'y_max': (height + self.goal_width) / 2
        }
        self.right_goal = {
            'x': width,
            'y_min': (height - self.goal_width) / 2,
            'y_max': (height + self.goal_width) / 2
        }
    
    def is_goal_scored(self, ball_pos: Vector2D, previous_pos: Vector2D) -> Optional[str]:
        """Check if a goal was scored. Returns 'left' or 'right' or None."""
        # Left goal (player 2 scores)
        if (previous_pos.x > 0 and ball_pos.x <= 0 and 
            self.left_goal['y_min'] <= ball_pos.y <= self.left_goal['y_max']):
            return 'left'
        
        # Right goal (player 1 scores)
        if (previous_pos.x < self.width and ball_pos.x >= self.width and 
            self.right_goal['y_min'] <= ball_pos.y <= self.right_goal['y_max']):
            return 'right'
        
        return None
    
    def keep_in_bounds(self, pos: Vector2D) -> Vector2D:
        """Keep a position within field boundaries."""
        x = max(0, min(self.width, pos.x))
        y = max(0, min(self.height, pos.y))
        return Vector2D(x, y)


class Ball:
    """Soccer ball with physics."""
    
    def __init__(self, position: Vector2D):
        self.position = position.copy()
        self.velocity = Vector2D(0, 0)
        self.friction = 0.95
        self.max_speed = 15.0
    
    def update(self, dt: float = 0.1):
        """Update ball position and apply friction."""
        self.position = self.position + self.velocity * dt
        self.velocity = self.velocity * self.friction
        
        # Limit speed
        if self.velocity.magnitude() > self.max_speed:
            self.velocity = self.velocity.normalize() * self.max_speed
    
    def kick(self, force: Vector2D):
        """Apply a force to the ball."""
        self.velocity = self.velocity + force


class Agent:
    """Soccer agent (player)."""
    
    def __init__(self, agent_id: int, position: Vector2D, team: str):
        self.id = agent_id
        self.position = position.copy()
        self.velocity = Vector2D(0, 0)
        self.team = team  # 'left' or 'right'
        self.max_speed = 8.0
        self.kick_range = 3.0
        self.kick_power = 10.0
        
        # Simple Q-learning parameters
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.3  # exploration rate
        self.q_table = {}
    
    def get_state(self, ball: Ball, opponent: 'Agent', field: SoccerField) -> str:
        """Get current state for learning (simplified discrete state)."""
        # Discretize positions for Q-table
        my_x = int(self.position.x / 10) * 10
        my_y = int(self.position.y / 10) * 10
        ball_x = int(ball.position.x / 10) * 10
        ball_y = int(ball.position.y / 10) * 10
        
        ball_dist = self.position.distance_to(ball.position)
        ball_dist_discrete = "close" if ball_dist < 5 else "medium" if ball_dist < 15 else "far"
        
        # Determine if we have ball possession
        has_ball = ball_dist < self.kick_range
        
        return f"{my_x},{my_y},{ball_x},{ball_y},{ball_dist_discrete},{has_ball},{self.team}"
    
    def choose_action(self, state: str) -> int:
        """Choose action using epsilon-greedy policy."""
        actions = list(range(8))  # 8 possible actions
        
        if random.random() < self.epsilon:
            return random.choice(actions)
        
        if state not in self.q_table:
            self.q_table[state] = [0.0] * len(actions)
        
        return actions[self.q_table[state].index(max(self.q_table[state]))]
    
    def update_q_table(self, state: str, action: int, reward: float, next_state: str):
        """Update Q-table with new experience."""
        if state not in self.q_table:
            self.q_table[state] = [0.0] * 8
        if next_state not in self.q_table:
            self.q_table[next_state] = [0.0] * 8
        
        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state])
        
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state][action] = new_q
    
    def execute_action(self, action: int, ball: Ball, field: SoccerField):
        """Execute the chosen action."""
        # Actions: 0-7 for 8 directions, with special ball interaction
        directions = [
            Vector2D(0, -1),   # Up
            Vector2D(1, -1),   # Up-Right
            Vector2D(1, 0),    # Right
            Vector2D(1, 1),    # Down-Right
            Vector2D(0, 1),    # Down
            Vector2D(-1, 1),   # Down-Left
            Vector2D(-1, 0),   # Left
            Vector2D(-1, -1),  # Up-Left
        ]
        
        if action < len(directions):
            # Move in chosen direction
            direction = directions[action]
            self.velocity = direction * self.max_speed
            
            # Check if we can kick the ball
            ball_distance = self.position.distance_to(ball.position)
            if ball_distance < self.kick_range:
                # Kick ball towards opponent's goal
                if self.team == 'left':
                    goal_pos = Vector2D(field.width, field.height / 2)
                else:
                    goal_pos = Vector2D(0, field.height / 2)
                
                kick_direction = (goal_pos - ball.position).normalize()
                kick_force = kick_direction * self.kick_power
                ball.kick(kick_force)
    
    def update(self, dt: float, field: SoccerField):
        """Update agent position."""
        self.position = self.position + self.velocity * dt
        self.position = field.keep_in_bounds(self.position)
        
        # Apply some friction
        self.velocity = self.velocity * 0.8


class SoccerSimulation:
    """Main simulation class managing the soccer game."""
    
    def __init__(self):
        self.field = SoccerField()
        self.ball = Ball(Vector2D(self.field.width / 2, self.field.height / 2))
        
        # Create agents
        self.agent1 = Agent(1, Vector2D(20, self.field.height / 2), 'left')
        self.agent2 = Agent(2, Vector2D(80, self.field.height / 2), 'right')
        
        self.score = {'left': 0, 'right': 0}
        self.episode_length = 1000  # steps per episode
        self.current_step = 0
        self.episode = 0
        
        # For tracking learning progress
        self.episode_rewards = {'agent1': [], 'agent2': []}
    
    def reset_game(self):
        """Reset the game state for a new episode."""
        self.ball.position = Vector2D(self.field.width / 2, self.field.height / 2)
        self.ball.velocity = Vector2D(0, 0)
        
        self.agent1.position = Vector2D(20, self.field.height / 2)
        self.agent2.position = Vector2D(80, self.field.height / 2)
        self.agent1.velocity = Vector2D(0, 0)
        self.agent2.velocity = Vector2D(0, 0)
        
        self.current_step = 0
        self.episode += 1
    
    def calculate_reward(self, agent: Agent, goal_scored: Optional[str]) -> float:
        """Calculate reward for an agent."""
        reward = 0.0
        
        # Goal rewards
        if goal_scored:
            if (agent.team == 'left' and goal_scored == 'right') or \
               (agent.team == 'right' and goal_scored == 'left'):
                reward += 100.0  # Scored a goal
            else:
                reward -= 100.0  # Opponent scored
        
        # Distance to ball reward (encourage ball possession)
        ball_distance = agent.position.distance_to(self.ball.position)
        reward += max(0, 20 - ball_distance) * 0.1
        
        # Penalty for being too far from action
        if ball_distance > 30:
            reward -= 0.5
        
        # Small reward for being in good position
        if agent.team == 'left':
            # Left team should stay on left side but move towards ball
            if agent.position.x < self.field.width / 2:
                reward += 0.1
        else:
            # Right team should stay on right side but move towards ball
            if agent.position.x > self.field.width / 2:
                reward += 0.1
        
        return reward
    
    def step(self) -> Dict[str, Any]:
        """Execute one simulation step."""
        # Store previous ball position for goal detection
        prev_ball_pos = self.ball.position.copy()
        
        # Get current states
        state1 = self.agent1.get_state(self.ball, self.agent2, self.field)
        state2 = self.agent2.get_state(self.ball, self.agent1, self.field)
        
        # Choose actions
        action1 = self.agent1.choose_action(state1)
        action2 = self.agent2.choose_action(state2)
        
        # Execute actions
        self.agent1.execute_action(action1, self.ball, self.field)
        self.agent2.execute_action(action2, self.ball, self.field)
        
        # Update physics
        dt = 0.1
        self.agent1.update(dt, self.field)
        self.agent2.update(dt, self.field)
        self.ball.update(dt)
        
        # Keep ball in bounds
        self.ball.position = self.field.keep_in_bounds(self.ball.position)
        
        # Check for goals
        goal_scored = self.field.is_goal_scored(self.ball.position, prev_ball_pos)
        
        if goal_scored:
            if goal_scored == 'left':
                self.score['right'] += 1
            else:
                self.score['left'] += 1
        
        # Calculate rewards
        reward1 = self.calculate_reward(self.agent1, goal_scored)
        reward2 = self.calculate_reward(self.agent2, goal_scored)
        
        # Get new states
        next_state1 = self.agent1.get_state(self.ball, self.agent2, self.field)
        next_state2 = self.agent2.get_state(self.ball, self.agent1, self.field)
        
        # Update Q-tables
        self.agent1.update_q_table(state1, action1, reward1, next_state1)
        self.agent2.update_q_table(state2, action2, reward2, next_state2)
        
        # Track episode rewards
        while len(self.episode_rewards['agent1']) <= self.episode:
            self.episode_rewards['agent1'].append(0)
            self.episode_rewards['agent2'].append(0)
        
        self.episode_rewards['agent1'][self.episode] += reward1
        self.episode_rewards['agent2'][self.episode] += reward2
        
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= self.episode_length or goal_scored is not None
        
        if done and goal_scored is None:
            # Episode timeout - small penalty
            reward1 -= 10
            reward2 -= 10
        
        return {
            'step': self.current_step,
            'episode': self.episode,
            'score': self.score.copy(),
            'goal_scored': goal_scored,
            'agent1_reward': reward1,
            'agent2_reward': reward2,
            'done': done,
            'ball_pos': (self.ball.position.x, self.ball.position.y),
            'agent1_pos': (self.agent1.position.x, self.agent1.position.y),
            'agent2_pos': (self.agent2.position.x, self.agent2.position.y)
        }
    
    def run_episode(self) -> Dict[str, Any]:
        """Run a complete episode."""
        self.reset_game()
        episode_data = []
        
        while self.current_step < self.episode_length:
            step_result = self.step()
            episode_data.append(step_result)
            
            if step_result['done']:
                break
        
        return {
            'episode': self.episode,
            'steps': len(episode_data),
            'final_score': self.score.copy(),
            'total_reward_agent1': self.episode_rewards['agent1'][self.episode] if self.episode < len(self.episode_rewards['agent1']) else 0,
            'total_reward_agent2': self.episode_rewards['agent2'][self.episode] if self.episode < len(self.episode_rewards['agent2']) else 0,
            'episode_data': episode_data
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get learning statistics."""
        return {
            'episodes_completed': self.episode,
            'current_score': self.score.copy(),
            'agent1_q_table_size': len(self.agent1.q_table),
            'agent2_q_table_size': len(self.agent2.q_table),
            'agent1_epsilon': self.agent1.epsilon,
            'agent2_epsilon': self.agent2.epsilon,
            'recent_rewards_agent1': self.episode_rewards['agent1'][-10:] if len(self.episode_rewards['agent1']) >= 10 else self.episode_rewards['agent1'],
            'recent_rewards_agent2': self.episode_rewards['agent2'][-10:] if len(self.episode_rewards['agent2']) >= 10 else self.episode_rewards['agent2']
        }