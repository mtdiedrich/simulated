"""
Tests for the 1v1 Soccer Simulation

Basic tests to verify the simulation components work correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from soccer_simulation import Vector2D, SoccerField, Ball, Agent, SoccerSimulation


def test_vector2d():
    """Test Vector2D operations."""
    print("Testing Vector2D...")
    
    v1 = Vector2D(3, 4)
    v2 = Vector2D(1, 2)
    
    # Test magnitude
    assert abs(v1.magnitude() - 5.0) < 0.001, f"Expected magnitude 5.0, got {v1.magnitude()}"
    
    # Test addition
    v3 = v1 + v2
    assert v3.x == 4 and v3.y == 6, f"Expected (4, 6), got ({v3.x}, {v3.y})"
    
    # Test normalization
    v4 = v1.normalize()
    assert abs(v4.magnitude() - 1.0) < 0.001, f"Expected normalized magnitude 1.0, got {v4.magnitude()}"
    
    print("✓ Vector2D tests passed")


def test_soccer_field():
    """Test SoccerField functionality."""
    print("Testing SoccerField...")
    
    field = SoccerField(100, 60)
    
    # Test goal detection
    ball_pos = Vector2D(-1, 30)  # Ball in left goal
    prev_pos = Vector2D(5, 30)
    goal = field.is_goal_scored(ball_pos, prev_pos)
    assert goal == 'left', f"Expected 'left' goal, got {goal}"
    
    # Test bounds keeping
    out_of_bounds = Vector2D(-10, 70)
    in_bounds = field.keep_in_bounds(out_of_bounds)
    assert in_bounds.x == 0 and in_bounds.y == 60, f"Expected (0, 60), got ({in_bounds.x}, {in_bounds.y})"
    
    print("✓ SoccerField tests passed")


def test_ball():
    """Test Ball physics."""
    print("Testing Ball...")
    
    ball = Ball(Vector2D(50, 30))
    original_pos = ball.position.copy()
    
    # Test kick
    ball.kick(Vector2D(10, 0))
    assert ball.velocity.x > 0, "Ball should have positive x velocity after kick"
    
    # Test update
    ball.update()
    assert ball.position.x > original_pos.x, "Ball should move after update"
    
    print("✓ Ball tests passed")


def test_agent():
    """Test Agent behavior."""
    print("Testing Agent...")
    
    agent = Agent(1, Vector2D(20, 30), 'left')
    ball = Ball(Vector2D(50, 30))
    field = SoccerField()
    
    # Test state generation
    state = agent.get_state(ball, Agent(2, Vector2D(80, 30), 'right'), field)
    assert isinstance(state, str), "State should be a string"
    assert len(state) > 0, "State should not be empty"
    
    # Test action choice
    action = agent.choose_action(state)
    assert 0 <= action < 8, f"Action should be between 0-7, got {action}"
    
    print("✓ Agent tests passed")


def test_simulation():
    """Test full simulation."""
    print("Testing SoccerSimulation...")
    
    sim = SoccerSimulation()
    
    # Test initial state
    assert sim.current_step == 0, "Simulation should start at step 0"
    assert sim.score['left'] == 0 and sim.score['right'] == 0, "Initial score should be 0-0"
    
    # Test single step
    result = sim.step()
    assert 'step' in result, "Step result should contain 'step'"
    assert 'agent1_reward' in result, "Step result should contain agent rewards"
    
    # Test episode
    episode_result = sim.run_episode()
    assert 'episode' in episode_result, "Episode result should contain episode number"
    assert episode_result['steps'] > 0, "Episode should have positive number of steps"
    
    print("✓ SoccerSimulation tests passed")


def test_learning():
    """Test that agents can learn (Q-table grows)."""
    print("Testing learning capability...")
    
    sim = SoccerSimulation()
    
    # Run a few episodes
    for _ in range(5):
        sim.run_episode()
    
    # Check that Q-tables have grown
    assert len(sim.agent1.q_table) > 0, "Agent1 should have learned some states"
    assert len(sim.agent2.q_table) > 0, "Agent2 should have learned some states"
    
    print(f"✓ Learning test passed - Agent1: {len(sim.agent1.q_table)} states, Agent2: {len(sim.agent2.q_table)} states")


def run_all_tests():
    """Run all tests."""
    print("Running 1v1 Soccer Simulation Tests")
    print("=" * 40)
    
    try:
        test_vector2d()
        test_soccer_field()
        test_ball()
        test_agent()
        test_simulation()
        test_learning()
        
        print("=" * 40)
        print("✓ All tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    if not success:
        sys.exit(1)