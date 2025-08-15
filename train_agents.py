"""
Training script for 1v1 Soccer Simulation

This script runs the training loop for two agents learning to play soccer.
"""

import json
import time
from soccer_simulation import SoccerSimulation


def train_agents(num_episodes: int = 100, save_interval: int = 10):
    """Train the soccer agents."""
    simulation = SoccerSimulation()
    
    print("Starting 1v1 Soccer Simulation Training")
    print(f"Training for {num_episodes} episodes...")
    print("-" * 50)
    
    training_log = []
    
    for episode in range(num_episodes):
        start_time = time.time()
        
        # Run episode
        episode_result = simulation.run_episode()
        
        # Decay exploration rate
        if episode % 10 == 0:
            simulation.agent1.epsilon = max(0.05, simulation.agent1.epsilon * 0.95)
            simulation.agent2.epsilon = max(0.05, simulation.agent2.epsilon * 0.95)
        
        # Log episode
        training_log.append(episode_result)
        
        # Print progress
        if episode % save_interval == 0 or episode == num_episodes - 1:
            stats = simulation.get_statistics()
            episode_time = time.time() - start_time
            
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Score: {stats['current_score']} | "
                  f"Steps: {episode_result['steps']} | "
                  f"Time: {episode_time:.2f}s")
            
            print(f"  Agent1 - Reward: {episode_result['total_reward_agent1']:.1f}, "
                  f"Q-table: {stats['agent1_q_table_size']} states, "
                  f"ε: {stats['agent1_epsilon']:.3f}")
            
            print(f"  Agent2 - Reward: {episode_result['total_reward_agent2']:.1f}, "
                  f"Q-table: {stats['agent2_q_table_size']} states, "
                  f"ε: {stats['agent2_epsilon']:.3f}")
            
            # Show recent performance
            if len(stats['recent_rewards_agent1']) > 0:
                avg_reward1 = sum(stats['recent_rewards_agent1']) / len(stats['recent_rewards_agent1'])
                avg_reward2 = sum(stats['recent_rewards_agent2']) / len(stats['recent_rewards_agent2'])
                print(f"  Recent avg rewards: Agent1={avg_reward1:.1f}, Agent2={avg_reward2:.1f}")
            
            print("-" * 50)
    
    # Save training results
    final_stats = simulation.get_statistics()
    
    results = {
        'training_episodes': num_episodes,
        'final_statistics': final_stats,
        'training_log': training_log[-10:]  # Save last 10 episodes to keep file size manageable
    }
    
    with open('training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"Final score: {final_stats['current_score']}")
    print(f"Agent1 learned {final_stats['agent1_q_table_size']} states")
    print(f"Agent2 learned {final_stats['agent2_q_table_size']} states")
    print("Results saved to training_results.json")
    
    return simulation, results


def demonstrate_trained_agents(simulation: SoccerSimulation, num_demo_episodes: int = 3):
    """Demonstrate the trained agents playing."""
    print(f"\nDemonstrating trained agents for {num_demo_episodes} episodes...")
    
    # Disable exploration for demonstration
    original_epsilon1 = simulation.agent1.epsilon
    original_epsilon2 = simulation.agent2.epsilon
    simulation.agent1.epsilon = 0.0
    simulation.agent2.epsilon = 0.0
    
    for demo in range(num_demo_episodes):
        print(f"\nDemo Episode {demo + 1}:")
        
        episode_result = simulation.run_episode()
        
        print(f"  Duration: {episode_result['steps']} steps")
        print(f"  Final Score: {episode_result['final_score']}")
        
        # Show key moments
        episode_data = episode_result['episode_data']
        goals = [step for step in episode_data if step['goal_scored']]
        
        if goals:
            for i, goal in enumerate(goals):
                scorer = "Agent1" if goal['goal_scored'] == 'right' else "Agent2"
                print(f"  Goal {i+1}: {scorer} scored at step {goal['step']}")
        else:
            print("  No goals scored this episode")
    
    # Restore exploration rates
    simulation.agent1.epsilon = original_epsilon1
    simulation.agent2.epsilon = original_epsilon2


def visualize_game_state(simulation: SoccerSimulation):
    """Simple ASCII visualization of the current game state."""
    field = simulation.field
    ball = simulation.ball
    agent1 = simulation.agent1
    agent2 = simulation.agent2
    
    # Create ASCII field
    width_chars = 40
    height_chars = 20
    
    field_display = [['.' for _ in range(width_chars)] for _ in range(height_chars)]
    
    # Scale positions to display coordinates
    scale_x = width_chars / field.width
    scale_y = height_chars / field.height
    
    # Place ball
    ball_x = int(ball.position.x * scale_x)
    ball_y = int(ball.position.y * scale_y)
    if 0 <= ball_x < width_chars and 0 <= ball_y < height_chars:
        field_display[ball_y][ball_x] = 'O'
    
    # Place agents
    agent1_x = int(agent1.position.x * scale_x)
    agent1_y = int(agent1.position.y * scale_y)
    if 0 <= agent1_x < width_chars and 0 <= agent1_y < height_chars:
        field_display[agent1_y][agent1_x] = '1'
    
    agent2_x = int(agent2.position.x * scale_x)
    agent2_y = int(agent2.position.y * scale_y)
    if 0 <= agent2_x < width_chars and 0 <= agent2_y < height_chars:
        field_display[agent2_y][agent2_x] = '2'
    
    # Add goals
    goal_y_start = int(field.left_goal['y_min'] * scale_y)
    goal_y_end = int(field.left_goal['y_max'] * scale_y)
    for y in range(max(0, goal_y_start), min(height_chars, goal_y_end + 1)):
        field_display[y][0] = '|'
        field_display[y][width_chars - 1] = '|'
    
    # Print field
    print("\nCurrent Game State:")
    print("1=Agent1(left), 2=Agent2(right), O=Ball, |=Goals")
    print("+" + "-" * width_chars + "+")
    for row in field_display:
        print("|" + "".join(row) + "|")
    print("+" + "-" * width_chars + "+")
    print(f"Score: Agent1={simulation.score['left']}, Agent2={simulation.score['right']}")


if __name__ == "__main__":
    # Train the agents
    trained_simulation, results = train_agents(num_episodes=50)
    
    # Show current game state
    visualize_game_state(trained_simulation)
    
    # Demonstrate trained agents
    demonstrate_trained_agents(trained_simulation)
    
    print("\n" + "="*50)
    print("1v1 Soccer Simulation Complete!")
    print("The agents have learned basic soccer behavior through reinforcement learning.")
    print("You can extend this simulation by:")
    print("- Adding more sophisticated reward functions")
    print("- Implementing neural networks instead of Q-tables")
    print("- Adding more complex physics and game mechanics")
    print("- Creating a visual interface with pygame or similar")
    print("="*50)