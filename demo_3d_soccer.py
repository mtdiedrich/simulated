"""
3D Soccer Simulation Demo

Demonstrates the 3D soccer simulation with physics and colliders.
Shows how agents learn to play soccer in a 3D environment with realistic ball physics.
"""

import time
import os
from soccer_simulation_3d import Soccer3DSimulation


def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def demo_basic_3d_simulation():
    """Demonstrate basic 3D simulation functionality."""
    print("=== 3D Soccer Simulation Demo ===\n")
    
    sim = Soccer3DSimulation()
    
    print("Initializing 3D Soccer Simulation...")
    print(f"Field dimensions: {sim.field.width} x {sim.field.height} x {sim.field.field_height}")
    print(f"Physics engine with {len(sim.physics.colliders)} colliders")
    print(f"Ball starts at: {sim.ball.position}")
    print(f"Agent 1 (Left team) at: {sim.agent1.position}")
    print(f"Agent 2 (Right team) at: {sim.agent2.position}")
    print()
    
    # Show initial field state
    print("Initial field state:")
    print(sim.visualize_ascii_3d())
    print()
    
    input("Press Enter to run simulation steps...")
    
    # Run a few steps and show the simulation in action
    for step in range(10):
        result = sim.step()
        
        if step % 3 == 0:  # Show every 3rd step
            clear_screen()
            print(f"=== Step {result['step']} ===")
            print(sim.visualize_ascii_3d())
            print(f"Rewards: Agent1={result['agent1_reward']:.2f}, Agent2={result['agent2_reward']:.2f}")
            print()
            time.sleep(1)
        
        if result['done']:
            print(f"Episode finished! Goal scored: {result['goal_scored']}")
            break
    
    print("\nFinal statistics:")
    stats = sim.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")


def demo_physics_features():
    """Demonstrate specific 3D physics features."""
    print("\n=== 3D Physics Features Demo ===\n")
    
    sim = Soccer3DSimulation()
    
    print("Testing 3D ball physics...")
    
    # Give ball initial velocity with upward component
    sim.ball.kick(Vector3D(10, 5, 8))
    print(f"Ball kicked with force: (10, 5, 8)")
    print(f"Ball velocity: {sim.ball.velocity}")
    
    # Run simulation to show ball trajectory
    for i in range(20):
        sim.ball.update(1/30)  # 30 FPS
        sim.physics.step(1/30)
        
        if i % 5 == 0:
            print(f"Frame {i:2d}: Ball at ({sim.ball.position.x:5.1f}, {sim.ball.position.y:5.1f}, {sim.ball.position.z:5.1f}) "
                  f"velocity ({sim.ball.velocity.x:5.1f}, {sim.ball.velocity.y:5.1f}, {sim.ball.velocity.z:5.1f})")
        
        # Stop if ball hits ground and stops bouncing
        if sim.ball.position.z <= sim.ball.radius + 0.1 and abs(sim.ball.velocity.z) < 0.1:
            print(f"Ball settled on ground at frame {i}")
            break
    
    print("\nTesting agent jumping...")
    
    # Reset simulation
    sim.reset_game()
    
    # Make agent jump
    print(f"Agent 1 initial position: {sim.agent1.position}")
    sim.agent1.execute_action(8, sim.ball, sim.field)  # Jump action
    print(f"Agent 1 after jump command: velocity.z = {sim.agent1.velocity.z}")
    
    # Show jump trajectory
    for i in range(15):
        sim.agent1.update(1/30, sim.field)
        sim.physics.step(1/30)
        
        if i % 3 == 0:
            print(f"Jump frame {i:2d}: Agent at z={sim.agent1.position.z:.2f}, velocity.z={sim.agent1.velocity.z:.2f} {'[JUMPING]' if sim.agent1.is_jumping else '[GROUNDED]'}")
        
        if not sim.agent1.is_jumping and sim.agent1.position.z <= sim.agent1.radius + 0.1:
            print(f"Agent landed at frame {i}")
            break


def demo_learning_progress():
    """Demonstrate learning progress over multiple episodes."""
    print("\n=== Learning Progress Demo ===\n")
    
    sim = Soccer3DSimulation()
    
    print("Running training episodes to show learning progress...")
    
    episode_scores = []
    episode_rewards = []
    
    for episode in range(5):
        result = sim.run_episode()
        
        # Calculate average rewards
        avg_reward_1 = sum(sim.episode_rewards['agent1'][-1:]) / max(1, len(sim.episode_rewards['agent1'][-1:]))
        avg_reward_2 = sum(sim.episode_rewards['agent2'][-1:]) / max(1, len(sim.episode_rewards['agent2'][-1:]))
        
        episode_scores.append(result['final_score'])
        episode_rewards.append((avg_reward_1, avg_reward_2))
        
        print(f"Episode {episode + 1}:")
        print(f"  Steps: {result['steps']}")
        print(f"  Final Score: Left {result['final_score']['left']} - {result['final_score']['right']} Right")
        print(f"  Agent 1 reward: {result['total_reward_agent1']:.2f}")
        print(f"  Agent 2 reward: {result['total_reward_agent2']:.2f}")
        print(f"  Q-table sizes: Agent1={len(sim.agent1.q_table)}, Agent2={len(sim.agent2.q_table)}")
        print()
    
    print("Learning Summary:")
    print(f"Total episodes: {len(episode_scores)}")
    print(f"Final Q-table sizes: Agent1={len(sim.agent1.q_table)}, Agent2={len(sim.agent2.q_table)}")
    
    # Show final simulation state
    print("\nFinal simulation state:")
    print(sim.visualize_ascii_3d())


def demo_3d_vs_2d_comparison():
    """Compare 3D simulation features with 2D."""
    print("\n=== 3D vs 2D Comparison ===\n")
    
    from soccer_simulation import SoccerSimulation  # Import 2D version
    
    # Create both simulations
    sim_2d = SoccerSimulation()
    sim_3d = Soccer3DSimulation()
    
    print("Feature Comparison:")
    print("                    2D Simulation    3D Simulation")
    print("=" * 50)
    print(f"Dimensions:         2D (x,y)        3D (x,y,z)")
    print(f"Ball Physics:       Basic           Gravity + Bounce")
    print(f"Agent Actions:      8 directions    12 actions (+ jump)")
    print(f"Colliders:          None            Sphere/Box colliders")
    print(f"Physics Engine:     None            Full 3D physics")
    print(f"Goals:              Line crossing   3D volume detection")
    print(f"Visualization:      2D ASCII        3D ASCII + height info")
    print()
    
    print("Action Space Comparison:")
    print("2D Actions: 8 directional movements")
    print("3D Actions: 8 directional movements + jump + brake + power kick + header")
    print()
    
    print("State Space Comparison:")
    # Get sample states
    ball_2d = sim_2d.ball
    agent_2d = sim_2d.agent1
    state_2d = agent_2d.get_state(ball_2d, sim_2d.agent2, sim_2d.field)
    
    ball_3d = sim_3d.ball
    agent_3d = sim_3d.agent1
    state_3d = agent_3d.get_state(ball_3d, sim_3d.agent2, sim_3d.field)
    
    print(f"2D State: {state_2d}")
    print(f"3D State: {state_3d}")
    print(f"2D State components: {len(state_2d.split(','))}")
    print(f"3D State components: {len(state_3d.split(','))}")


def main():
    """Run all demos."""
    try:
        demo_basic_3d_simulation()
        input("\nPress Enter to continue to physics demo...")
        
        demo_physics_features()
        input("\nPress Enter to continue to learning demo...")
        
        demo_learning_progress()
        input("\nPress Enter to continue to comparison demo...")
        
        demo_3d_vs_2d_comparison()
        
        print("\n=== Demo Complete ===")
        print("The 3D soccer simulation successfully demonstrates:")
        print("✓ 3D vector mathematics and physics")
        print("✓ Collision detection with sphere and box colliders")
        print("✓ Realistic ball physics with gravity and bouncing")
        print("✓ Agent physics with jumping and 3D movement")
        print("✓ 3D field with walls, goals, and boundaries")
        print("✓ Enhanced learning with extended state and action spaces")
        print("✓ Physics engine with collision resolution")
        print("\nBoth 2D and 3D simulations coexist without breaking existing functionality!")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nDemo error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Import the Vector3D class to avoid NameError in physics demo
    from soccer_simulation_3d import Vector3D
    main()