#!/usr/bin/env python3
"""
Enhanced 3D Soccer Simulation Demo

Demonstrates the 3D soccer simulation with physics and 3D model visualization.
Falls back gracefully to ASCII visualization if 3D rendering is not available.
"""

import sys
import time
from soccer_simulation_3d import Soccer3DSimulation


def demo_3d_features():
    """Demonstrate the key 3D features of the simulation."""
    print("=== 3D Soccer Simulation Features Demo ===")
    print()
    
    # Create simulation
    sim = Soccer3DSimulation()
    
    # Show initial setup
    print("Simulation Features:")
    print(f"✓ 3D Physics Engine with {len(sim.physics.colliders)} colliders")
    print(f"✓ Field dimensions: {sim.field.width} x {sim.field.height} x {sim.field.field_height}")
    print(f"✓ Ball physics: gravity, bouncing, air resistance")
    print(f"✓ Agent physics: jumping, collision detection, 3D movement")
    print(f"✓ Learning: Q-learning with 3D state space and 12 actions")
    print()
    
    return sim


def demo_ascii_visualization(sim):
    """Demonstrate ASCII visualization."""
    print("=== ASCII Visualization Demo ===")
    print("Running a few simulation steps with ASCII visualization...")
    print()
    
    sim.reset_game()
    
    for step in range(5):
        print(f"Step {step + 1}:")
        result = sim.step()
        print(sim.visualize_ascii_3d())
        print(f"Ball velocity: ({result['ball_velocity'][0]:.1f}, {result['ball_velocity'][1]:.1f}, {result['ball_velocity'][2]:.1f})")
        print(f"Agent 1 jumping: {result['agent1_jumping']}")
        print(f"Agent 2 jumping: {result['agent2_jumping']}")
        print()
        time.sleep(0.5)  # Small delay for readability


def demo_3d_visualization(sim):
    """Demonstrate 3D visualization."""
    print("=== 3D Model Visualization Demo ===")
    print("Attempting to launch 3D visualization...")
    print("Features:")
    print("✓ Real-time 3D rendering with OpenGL")
    print("✓ 3D field with grass, goals, and walls")
    print("✓ 3D agent models (cylinders) with team colors")
    print("✓ 3D ball physics visualization")
    print("✓ Camera positioned for optimal viewing")
    print("✓ Lighting and materials for realistic appearance")
    print()
    
    try:
        # Try to use 3D visualization
        result = sim.run_with_3d_visualization(episodes=1, use_3d=True)
        print(f"3D visualization completed successfully!")
        print(f"Final result: {result}")
        return True
        
    except Exception as e:
        print(f"3D visualization not available: {e}")
        return False


def demo_training_session(sim):
    """Demonstrate a training session."""
    print("=== Training Session Demo ===")
    print("Running 3 training episodes to show learning in action...")
    print()
    
    for episode in range(3):
        print(f"Episode {episode + 1}/3:")
        result = sim.run_episode()
        
        print(f"  Duration: {result['steps']} steps")
        print(f"  Final Score: Left {result['final_score']['left']} - {result['final_score']['right']} Right")
        print(f"  Agent 1 reward: {result['total_reward_agent1']:.1f}")
        print(f"  Agent 2 reward: {result['total_reward_agent2']:.1f}")
        print(f"  Agent 1 Q-table size: {len(sim.agent1.q_table)}")
        print(f"  Agent 2 Q-table size: {len(sim.agent2.q_table)}")
        print()


def interactive_menu():
    """Show interactive menu for demo."""
    print("Enhanced 3D Soccer Simulation Demo")
    print("===================================")
    print()
    print("Choose a demo mode:")
    print("1. Features overview")
    print("2. ASCII visualization demo") 
    print("3. 3D model visualization demo")
    print("4. Training session demo")
    print("5. Full interactive 3D session")
    print("6. Run all demos")
    print("0. Exit")
    print()
    
    while True:
        try:
            choice = input("Enter your choice (0-6): ").strip()
            
            if choice == "0":
                print("Goodbye!")
                sys.exit(0)
                
            elif choice == "1":
                sim = demo_3d_features()
                input("Press Enter to continue...")
                
            elif choice == "2":
                sim = demo_3d_features()
                demo_ascii_visualization(sim)
                input("Press Enter to continue...")
                
            elif choice == "3":
                sim = demo_3d_features()
                success = demo_3d_visualization(sim)
                if not success:
                    print("Falling back to ASCII visualization...")
                    demo_ascii_visualization(sim)
                input("Press Enter to continue...")
                
            elif choice == "4":
                sim = demo_3d_features()
                demo_training_session(sim)
                input("Press Enter to continue...")
                
            elif choice == "5":
                print("=== Full Interactive 3D Session ===")
                try:
                    from renderer_3d import Soccer3DVisualization
                    viz = Soccer3DVisualization()
                    print("Launching interactive 3D visualization...")
                    print("Instructions:")
                    print("- The simulation will run automatically")
                    print("- Press ESC to quit")
                    print("- Agents learn through Q-learning")
                    print()
                    input("Press Enter to start...")
                    viz.run_interactive(episodes=2, steps_per_episode=500)
                    
                except Exception as e:
                    print(f"Interactive 3D mode not available: {e}")
                    print("Running training session instead...")
                    sim = demo_3d_features()
                    demo_training_session(sim)
                
                input("Press Enter to continue...")
                
            elif choice == "6":
                print("=== Running All Demos ===")
                print()
                
                # Features demo
                sim = demo_3d_features()
                input("Press Enter for ASCII visualization demo...")
                
                # ASCII demo
                demo_ascii_visualization(sim)
                input("Press Enter for 3D visualization demo...")
                
                # 3D demo
                success = demo_3d_visualization(sim)
                if not success:
                    print("3D not available, showing training session...")
                    demo_training_session(sim)
                
                print("All demos completed!")
                input("Press Enter to continue...")
                
            else:
                print("Invalid choice. Please enter 0-6.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)
        except Exception as e:
            print(f"Error: {e}")
            input("Press Enter to continue...")


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        # Command line argument provided
        mode = sys.argv[1].lower()
        
        if mode == "features":
            demo_3d_features()
        elif mode == "ascii":
            sim = demo_3d_features()
            demo_ascii_visualization(sim)
        elif mode == "3d":
            sim = demo_3d_features()
            demo_3d_visualization(sim)
        elif mode == "training":
            sim = demo_3d_features()
            demo_training_session(sim)
        elif mode == "interactive":
            from renderer_3d import Soccer3DVisualization
            viz = Soccer3DVisualization()
            viz.run_interactive(episodes=1, steps_per_episode=1000)
        else:
            print(f"Unknown mode: {mode}")
            print("Available modes: features, ascii, 3d, training, interactive")
            sys.exit(1)
    else:
        # Interactive mode
        interactive_menu()


if __name__ == "__main__":
    main()