"""
Quick demonstration of the 1v1 Soccer Simulation

This script provides a quick demo showing the key features of the simulation.
"""

from soccer_simulation import SoccerSimulation
import json


def quick_demo():
    """Run a quick demonstration of the soccer simulation."""
    print("🏈 1v1 Soccer Simulation - Quick Demo")
    print("=" * 50)
    
    # Create simulation
    sim = SoccerSimulation()
    
    print("Setting up soccer field and agents...")
    print(f"Field dimensions: {sim.field.width} x {sim.field.height}")
    print(f"Agent1 (left team) starting at: {sim.agent1.position}")
    print(f"Agent2 (right team) starting at: {sim.agent2.position}")
    print(f"Ball starting at: {sim.ball.position}")
    print()
    
    # Show initial state
    print("Initial Field State:")
    visualize_simple_field(sim)
    print()
    
    # Run a few training episodes
    print("🚀 Starting quick training (10 episodes)...")
    episode_results = []
    
    for episode in range(10):
        result = sim.run_episode()
        episode_results.append(result)
        
        if episode % 3 == 0:
            print(f"Episode {episode + 1}: {result['steps']} steps, "
                  f"Score: {result['final_score']}, "
                  f"Rewards: A1={result['total_reward_agent1']:.1f}, A2={result['total_reward_agent2']:.1f}")
    
    print()
    
    # Show learning progress
    stats = sim.get_statistics()
    print("📊 Learning Progress:")
    print(f"Agent1 learned {stats['agent1_q_table_size']} different game states")
    print(f"Agent2 learned {stats['agent2_q_table_size']} different game states")
    print(f"Current exploration rates: A1={stats['agent1_epsilon']:.3f}, A2={stats['agent2_epsilon']:.3f}")
    print()
    
    # Show final field state
    print("Final Field State:")
    visualize_simple_field(sim)
    print()
    
    # Analyze performance
    print("📈 Performance Analysis:")
    recent_rewards_1 = stats['recent_rewards_agent1']
    recent_rewards_2 = stats['recent_rewards_agent2']
    
    if recent_rewards_1:
        avg_reward_1 = sum(recent_rewards_1) / len(recent_rewards_1)
        avg_reward_2 = sum(recent_rewards_2) / len(recent_rewards_2)
        print(f"Average rewards over recent episodes:")
        print(f"  Agent1: {avg_reward_1:.1f}")
        print(f"  Agent2: {avg_reward_2:.1f}")
    
    # Show some example learned states
    print()
    print("🧠 Sample Learned States:")
    for i, (state, q_values) in enumerate(list(sim.agent1.q_table.items())[:3]):
        best_action = q_values.index(max(q_values))
        print(f"  State {i+1}: {state[:50]}...")
        print(f"    Best action: {best_action}, Q-value: {max(q_values):.2f}")
    
    print()
    print("✅ Demo complete! The agents have learned basic soccer behaviors.")
    
    return sim, episode_results


def visualize_simple_field(sim):
    """Simple field visualization for demo."""
    field = sim.field
    ball = sim.ball
    agent1 = sim.agent1
    agent2 = sim.agent2
    
    # Create a smaller field for demo
    width_chars = 30
    height_chars = 10
    
    field_display = [['.' for _ in range(width_chars)] for _ in range(height_chars)]
    
    # Scale positions
    scale_x = width_chars / field.width
    scale_y = height_chars / field.height
    
    # Place entities
    ball_x = int(ball.position.x * scale_x)
    ball_y = int(ball.position.y * scale_y)
    if 0 <= ball_x < width_chars and 0 <= ball_y < height_chars:
        field_display[ball_y][ball_x] = 'O'
    
    agent1_x = int(agent1.position.x * scale_x)
    agent1_y = int(agent1.position.y * scale_y)
    if 0 <= agent1_x < width_chars and 0 <= agent1_y < height_chars:
        field_display[agent1_y][agent1_x] = '1'
    
    agent2_x = int(agent2.position.x * scale_x)
    agent2_y = int(agent2.position.y * scale_y)
    if 0 <= agent2_x < width_chars and 0 <= agent2_y < height_chars:
        field_display[agent2_y][agent2_x] = '2'
    
    # Add goal markers
    goal_start = int((field.height - field.goal_width) / 2 * scale_y)
    goal_end = int((field.height + field.goal_width) / 2 * scale_y)
    for y in range(max(0, goal_start), min(height_chars, goal_end + 1)):
        if 0 <= y < height_chars:
            field_display[y][0] = '|'
            field_display[y][width_chars - 1] = '|'
    
    # Print field
    print("  " + "1=Agent1, 2=Agent2, O=Ball, |=Goals")
    print("  +" + "-" * width_chars + "+")
    for row in field_display:
        print("  |" + "".join(row) + "|")
    print("  +" + "-" * width_chars + "+")
    print(f"  Score: Left={sim.score['left']}, Right={sim.score['right']}")


def show_key_features():
    """Highlight the key features of the simulation."""
    print()
    print("🎯 Key Features Demonstrated:")
    print("  ✓ 2D soccer field with boundaries and goals")
    print("  ✓ Ball physics with movement and friction")
    print("  ✓ Two learning agents using Q-learning")
    print("  ✓ Reward system encouraging good soccer behavior")
    print("  ✓ Real-time learning and adaptation")
    print("  ✓ State space exploration and exploitation")
    print()
    print("🔧 Customization Options:")
    print("  • Adjust learning parameters (epsilon, learning rate)")
    print("  • Modify reward functions for different behaviors")
    print("  • Change field dimensions and physics")
    print("  • Extend to more complex neural network agents")
    print("  • Add visual interfaces and enhanced graphics")


if __name__ == "__main__":
    # Run the demo
    simulation, results = quick_demo()
    
    # Show key features
    show_key_features()
    
    print("\n" + "🏆" * 25)
    print("Demo completed successfully!")
    print("Run 'python3 train_agents.py' for full training session")
    print("Run 'python3 test_simulation.py' to validate components")
    print("🏆" * 25)