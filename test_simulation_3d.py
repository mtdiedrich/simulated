"""
Tests for the 3D Soccer Simulation

Tests to verify the 3D simulation components work correctly with physics and colliders.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from soccer_simulation_3d import (
    Vector3D, SphereCollider, BoxCollider, PhysicsEngine,
    SoccerField3D, Ball3D, Agent3D, Soccer3DSimulation
)


def test_vector3d():
    """Test Vector3D operations."""
    print("Testing Vector3D...")
    
    v1 = Vector3D(3, 4, 5)
    v2 = Vector3D(1, 2, 1)
    
    # Test magnitude
    expected_mag = math.sqrt(3**2 + 4**2 + 5**2)
    assert abs(v1.magnitude() - expected_mag) < 0.001, f"Expected magnitude {expected_mag}, got {v1.magnitude()}"
    
    # Test addition
    v3 = v1 + v2
    assert v3.x == 4 and v3.y == 6 and v3.z == 6, f"Expected (4, 6, 6), got ({v3.x}, {v3.y}, {v3.z})"
    
    # Test normalization
    v4 = v1.normalize()
    assert abs(v4.magnitude() - 1.0) < 0.001, f"Expected normalized magnitude 1.0, got {v4.magnitude()}"
    
    # Test dot product
    dot_product = v1.dot(v2)
    expected_dot = 3*1 + 4*2 + 5*1
    assert dot_product == expected_dot, f"Expected dot product {expected_dot}, got {dot_product}"
    
    # Test cross product
    cross = Vector3D(1, 0, 0).cross(Vector3D(0, 1, 0))
    expected_cross = Vector3D(0, 0, 1)
    assert (abs(cross.x - expected_cross.x) < 0.001 and 
            abs(cross.y - expected_cross.y) < 0.001 and 
            abs(cross.z - expected_cross.z) < 0.001), f"Expected cross product {expected_cross}, got {cross}"
    
    print("✓ Vector3D tests passed")


def test_colliders():
    """Test collider classes."""
    print("Testing Colliders...")
    
    # Test sphere collider
    sphere = SphereCollider(Vector3D(0, 0, 0), 2.0)
    assert sphere.contains_point(Vector3D(1, 1, 0)), "Sphere should contain point (1,1,0)"
    assert not sphere.contains_point(Vector3D(3, 0, 0)), "Sphere should not contain point (3,0,0)"
    
    # Test box collider
    box = BoxCollider(Vector3D(0, 0, 0), Vector3D(4, 4, 4))
    assert box.contains_point(Vector3D(1, 1, 1)), "Box should contain point (1,1,1)"
    assert not box.contains_point(Vector3D(3, 3, 3)), "Box should not contain point (3,3,3)"
    
    # Test sphere-sphere intersection
    sphere2 = SphereCollider(Vector3D(3, 0, 0), 2.0)
    assert sphere.intersects(sphere2), "Overlapping spheres should intersect"
    
    sphere3 = SphereCollider(Vector3D(5, 0, 0), 1.0)
    assert not sphere.intersects(sphere3), "Non-overlapping spheres should not intersect"
    
    # Test sphere-box intersection
    box2 = BoxCollider(Vector3D(3, 0, 0), Vector3D(2, 2, 2))
    assert sphere.intersects(box2), "Overlapping sphere and box should intersect"
    
    print("✓ Collider tests passed")


def test_physics_engine():
    """Test physics engine functionality."""
    print("Testing PhysicsEngine...")
    
    physics = PhysicsEngine()
    
    # Create test objects
    sphere1 = SphereCollider(Vector3D(0, 0, 0), 1.0, 1.0)
    sphere2 = SphereCollider(Vector3D(1.5, 0, 0), 1.0, 1.0)  # Overlapping
    
    # Mock objects with velocities
    class MockObject:
        def __init__(self, collider):
            self.collider = collider
            self.velocity = Vector3D(1, 0, 0)
    
    obj1 = MockObject(sphere1)
    obj2 = MockObject(sphere2)
    
    physics.add_dynamic_object(obj1)
    physics.add_dynamic_object(obj2)
    
    # Test collision detection
    collisions = physics.detect_collisions()
    assert len(collisions) > 0, "Overlapping objects should generate collisions"
    
    print("✓ PhysicsEngine tests passed")


def test_soccer_field_3d():
    """Test SoccerField3D functionality."""
    print("Testing SoccerField3D...")
    
    field = SoccerField3D(100, 60, 20)
    
    # Test goal detection
    ball_pos = Vector3D(-1, 30, 3)  # Ball in left goal area
    prev_pos = Vector3D(5, 30, 3)
    goal = field.is_goal_scored(ball_pos, prev_pos)
    assert goal == 'right', f"Expected 'right' goal, got {goal}"
    
    # Test ball too high for goal
    ball_pos_high = Vector3D(-1, 30, 15)  # Ball too high
    goal_high = field.is_goal_scored(ball_pos_high, prev_pos)
    assert goal_high is None, "Ball too high should not count as goal"
    
    # Test bounds keeping
    out_of_bounds = Vector3D(-10, 70, 25)
    in_bounds = field.keep_in_bounds(out_of_bounds)
    assert (in_bounds.x >= -5 and in_bounds.y <= 60 and in_bounds.z <= 20), \
           f"Expected bounds correction, got ({in_bounds.x}, {in_bounds.y}, {in_bounds.z})"
    
    # Test collider generation
    colliders = field.get_all_colliders()
    assert len(colliders) > 0, "Field should have colliders"
    
    print("✓ SoccerField3D tests passed")


def test_ball_3d():
    """Test Ball3D physics."""
    print("Testing Ball3D...")
    
    ball = Ball3D(Vector3D(50, 30, 5))
    original_pos = ball.position.copy()
    
    # Test kick
    ball.kick(Vector3D(10, 0, 5))
    assert ball.velocity.x > 0, "Ball should have positive x velocity after kick"
    assert ball.velocity.z > 0, "Ball should have positive z velocity after kick"
    
    # Test update with gravity
    ball.use_gravity = True
    initial_z_velocity = ball.velocity.z
    ball.update(0.1)
    
    # Ball should have moved
    assert ball.position.x > original_pos.x, "Ball should move horizontally after update"
    
    # Test ground collision
    ball.position = Vector3D(50, 30, 0.1)
    ball.velocity = Vector3D(0, 0, -5)
    ball.update(0.1)
    assert ball.position.z >= ball.radius, "Ball should not go below ground"
    assert ball.velocity.z >= 0, "Ball should bounce off ground"
    
    print("✓ Ball3D tests passed")


def test_agent_3d():
    """Test Agent3D behavior."""
    print("Testing Agent3D...")
    
    agent = Agent3D(1, Vector3D(20, 30, 1), 'left')
    ball = Ball3D(Vector3D(50, 30, 1))
    field = SoccerField3D()
    
    # Test state generation
    state = agent.get_state(ball, Agent3D(2, Vector3D(80, 30, 1), 'right'), field)
    assert isinstance(state, str), "State should be a string"
    assert len(state) > 0, "State should not be empty"
    assert ',' in state, "State should contain comma separators"
    
    # Test action choice (extended action space)
    action = agent.choose_action(state)
    assert 0 <= action < 12, f"Action should be between 0-11, got {action}"
    
    # Test jumping action
    original_z = agent.position.z
    agent.execute_action(8, ball, field)  # Jump action
    agent.update(0.1, field)
    assert agent.is_jumping or agent.velocity.z > 0, "Agent should jump when action 8 is executed"
    
    print("✓ Agent3D tests passed")


def test_simulation_3d():
    """Test full 3D simulation."""
    print("Testing Soccer3DSimulation...")
    
    sim = Soccer3DSimulation()
    
    # Test initial state
    assert sim.current_step == 0, "Simulation should start at step 0"
    assert sim.score['left'] == 0 and sim.score['right'] == 0, "Initial score should be 0-0"
    assert sim.ball.position.z > 0, "Ball should start above ground"
    
    # Test physics engine setup
    assert len(sim.physics.dynamic_objects) == 3, "Should have 3 dynamic objects (ball + 2 agents)"
    assert len(sim.physics.colliders) > 0, "Should have field colliders"
    
    # Test single step
    result = sim.step()
    assert 'step' in result, "Step result should contain 'step'"
    assert 'agent1_reward' in result, "Step result should contain agent rewards"
    assert 'ball_pos' in result and len(result['ball_pos']) == 3, "Should have 3D ball position"
    assert 'agent1_jumping' in result, "Should track jumping state"
    
    # Test episode
    episode_result = sim.run_episode()
    assert 'episode' in episode_result, "Episode result should contain episode number"
    assert episode_result['steps'] > 0, "Episode should have positive number of steps"
    
    # Test visualization
    viz = sim.visualize_ascii_3d()
    assert isinstance(viz, str), "Visualization should return string"
    assert len(viz) > 0, "Visualization should not be empty"
    assert 'Ball:' in viz, "Visualization should contain ball position info"
    
    print("✓ Soccer3DSimulation tests passed")


def test_learning_3d():
    """Test that 3D agents can learn (Q-table grows with 3D states)."""
    print("Testing 3D learning capability...")
    
    sim = Soccer3DSimulation()
    
    # Run a few episodes
    for _ in range(3):
        sim.run_episode()
    
    # Check that Q-tables have grown
    assert len(sim.agent1.q_table) > 0, "Agent1 should have learned some states"
    assert len(sim.agent2.q_table) > 0, "Agent2 should have learned some states"
    
    # Check that 3D states are being generated (should contain z coordinates)
    sample_state = list(sim.agent1.q_table.keys())[0] if sim.agent1.q_table else ""
    state_parts = sample_state.split(',')
    assert len(state_parts) >= 10, f"3D states should have more components, got {len(state_parts)} in '{sample_state}'"
    
    print(f"✓ 3D Learning test passed - Agent1: {len(sim.agent1.q_table)} states, Agent2: {len(sim.agent2.q_table)} states")


def test_collision_scenarios():
    """Test specific collision scenarios."""
    print("Testing collision scenarios...")
    
    sim = Soccer3DSimulation()
    
    # Test collision detection works
    sim.ball.position = Vector3D(20, 30, 1)
    sim.agent1.position = Vector3D(21, 30, 1)  # Close but not overlapping initially
    
    # Move ball towards agent to test collision
    sim.ball.velocity = Vector3D(5, 0, 0)
    sim.agent1.velocity = Vector3D(-2, 0, 0)
    
    # Store initial positions
    initial_ball_x = sim.ball.position.x
    initial_agent_x = sim.agent1.position.x
    
    # Run several physics steps to allow collision to occur and resolve
    for _ in range(10):
        sim.physics.step(1/60)
        sim.ball.update(1/60)
        sim.agent1.update(1/60, sim.field)
    
    # Check that objects have moved (physics is working)
    assert sim.ball.position.x != initial_ball_x or sim.agent1.position.x != initial_agent_x, \
           "Objects should have moved during simulation"
    
    # Test that colliders are properly positioned
    assert sim.ball.collider.position.distance_to(sim.ball.position) < 0.1, \
           "Ball collider should be positioned with ball"
    assert sim.agent1.collider.position.distance_to(sim.agent1.position) < 0.1, \
           "Agent collider should be positioned with agent"
    
    print("✓ Collision scenario tests passed")


def run_all_tests():
    """Run all 3D simulation tests."""
    print("Running 3D Soccer Simulation Tests")
    print("=" * 40)
    
    # Import math here since it's needed for tests
    import math
    globals()['math'] = math
    
    try:
        test_vector3d()
        test_colliders()
        test_physics_engine()
        test_soccer_field_3d()
        test_ball_3d()
        test_agent_3d()
        test_simulation_3d()
        test_learning_3d()
        test_collision_scenarios()
        
        print("=" * 40)
        print("✓ All 3D tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ 3D Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    if not success:
        sys.exit(1)