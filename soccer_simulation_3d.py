"""
3D Soccer Simulation Environment

A 3D soccer simulation with physics and colliders where two agents learn to play against each other.
This extends the 2D simulation with full 3D physics, collision detection, and realistic ball/player interactions.
"""

import math
import random
import json
from typing import Dict, List, Tuple, Optional, Any
from soccer_simulation import Vector2D, SoccerField, Ball, Agent, SoccerSimulation


class Vector3D:
    """3D vector class for positions, velocities, and forces."""
    
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z
    
    def __add__(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar: float) -> 'Vector3D':
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def __truediv__(self, scalar: float) -> 'Vector3D':
        if scalar == 0:
            return Vector3D(0, 0, 0)
        return Vector3D(self.x / scalar, self.y / scalar, self.z / scalar)
    
    def magnitude(self) -> float:
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)
    
    def magnitude_squared(self) -> float:
        """More efficient than magnitude() when only comparing distances."""
        return self.x ** 2 + self.y ** 2 + self.z ** 2
    
    def normalize(self) -> 'Vector3D':
        mag = self.magnitude()
        if mag == 0:
            return Vector3D(0, 0, 0)
        return Vector3D(self.x / mag, self.y / mag, self.z / mag)
    
    def distance_to(self, other: 'Vector3D') -> float:
        return (self - other).magnitude()
    
    def distance_squared_to(self, other: 'Vector3D') -> float:
        """More efficient than distance_to() when only comparing distances."""
        return (self - other).magnitude_squared()
    
    def dot(self, other: 'Vector3D') -> float:
        """Dot product."""
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other: 'Vector3D') -> 'Vector3D':
        """Cross product."""
        return Vector3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    
    def copy(self) -> 'Vector3D':
        return Vector3D(self.x, self.y, self.z)
    
    def to_2d(self) -> Vector2D:
        """Convert to 2D by ignoring z-coordinate."""
        return Vector2D(self.x, self.y)
    
    @classmethod
    def from_2d(cls, vec2d: Vector2D, z: float = 0.0) -> 'Vector3D':
        """Create 3D vector from 2D vector."""
        return cls(vec2d.x, vec2d.y, z)
    
    def __repr__(self) -> str:
        return f"Vector3D({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"


class Collider:
    """Base class for collision detection."""
    
    def __init__(self, position: Vector3D, mass: float = 1.0):
        self.position = position.copy()
        self.mass = mass
        self.is_trigger = False  # If True, detects collisions but doesn't resolve them
    
    def contains_point(self, point: Vector3D) -> bool:
        """Check if a point is inside this collider."""
        raise NotImplementedError
    
    def intersects(self, other: 'Collider') -> bool:
        """Check if this collider intersects with another."""
        raise NotImplementedError
    
    def get_closest_point(self, point: Vector3D) -> Vector3D:
        """Get the closest point on this collider to the given point."""
        raise NotImplementedError


class SphereCollider(Collider):
    """Spherical collider for ball and player collision detection."""
    
    def __init__(self, position: Vector3D, radius: float, mass: float = 1.0):
        super().__init__(position, mass)
        self.radius = radius
    
    def contains_point(self, point: Vector3D) -> bool:
        return self.position.distance_squared_to(point) <= self.radius ** 2
    
    def intersects(self, other: 'Collider') -> bool:
        if isinstance(other, SphereCollider):
            distance = self.position.distance_to(other.position)
            return distance <= (self.radius + other.radius)
        elif isinstance(other, BoxCollider):
            return other.intersects(self)  # Use box's sphere intersection logic
        return False
    
    def get_closest_point(self, point: Vector3D) -> Vector3D:
        direction = (point - self.position).normalize()
        return self.position + direction * self.radius


class BoxCollider(Collider):
    """Box collider for field boundaries and goals."""
    
    def __init__(self, position: Vector3D, size: Vector3D, mass: float = float('inf')):
        super().__init__(position, mass)
        self.size = size  # width, height, depth
        self.half_size = size / 2
    
    def contains_point(self, point: Vector3D) -> bool:
        rel_pos = point - self.position
        return (abs(rel_pos.x) <= self.half_size.x and
                abs(rel_pos.y) <= self.half_size.y and
                abs(rel_pos.z) <= self.half_size.z)
    
    def intersects(self, other: 'Collider') -> bool:
        if isinstance(other, SphereCollider):
            # Find closest point on box to sphere center
            closest = self.get_closest_point(other.position)
            distance = closest.distance_to(other.position)
            return distance <= other.radius
        elif isinstance(other, BoxCollider):
            # AABB vs AABB collision
            min1 = self.position - self.half_size
            max1 = self.position + self.half_size
            min2 = other.position - other.half_size
            max2 = other.position + other.half_size
            
            return (min1.x <= max2.x and max1.x >= min2.x and
                    min1.y <= max2.y and max1.y >= min2.y and
                    min1.z <= max2.z and max1.z >= min2.z)
        return False
    
    def get_closest_point(self, point: Vector3D) -> Vector3D:
        rel_pos = point - self.position
        closest_rel = Vector3D(
            max(-self.half_size.x, min(self.half_size.x, rel_pos.x)),
            max(-self.half_size.y, min(self.half_size.y, rel_pos.y)),
            max(-self.half_size.z, min(self.half_size.z, rel_pos.z))
        )
        return self.position + closest_rel


class PhysicsEngine:
    """Simple physics engine for collision detection and resolution."""
    
    def __init__(self, gravity: Vector3D = Vector3D(0, 0, -9.81)):
        self.gravity = gravity
        self.colliders = []
        self.dynamic_objects = []  # Objects that can move and be affected by physics
    
    def add_collider(self, collider: Collider):
        """Add a static collider to the physics world."""
        self.colliders.append(collider)
    
    def add_dynamic_object(self, obj):
        """Add a dynamic object that participates in physics."""
        self.dynamic_objects.append(obj)
    
    def detect_collisions(self) -> List[Tuple[object, object]]:
        """Detect all collisions between dynamic objects and colliders."""
        collisions = []
        
        # Check dynamic objects against static colliders
        for obj in self.dynamic_objects:
            if hasattr(obj, 'collider'):
                for collider in self.colliders:
                    if obj.collider.intersects(collider):
                        collisions.append((obj, collider))
        
        # Check dynamic objects against each other
        for i, obj1 in enumerate(self.dynamic_objects):
            for j, obj2 in enumerate(self.dynamic_objects[i+1:], i+1):
                if (hasattr(obj1, 'collider') and hasattr(obj2, 'collider') and 
                    obj1.collider.intersects(obj2.collider)):
                    collisions.append((obj1, obj2))
        
        return collisions
    
    def resolve_collision(self, obj1, obj2):
        """Resolve collision between two objects."""
        # Simple elastic collision resolution
        if hasattr(obj1, 'velocity') and hasattr(obj2, 'velocity'):
            # Both objects are dynamic
            self._resolve_dynamic_collision(obj1, obj2)
        elif hasattr(obj1, 'velocity'):
            # obj1 is dynamic, obj2 is static
            self._resolve_static_collision(obj1, obj2)
        elif hasattr(obj2, 'velocity'):
            # obj2 is dynamic, obj1 is static
            self._resolve_static_collision(obj2, obj1)
    
    def _resolve_dynamic_collision(self, obj1, obj2):
        """Resolve collision between two dynamic objects."""
        if not (hasattr(obj1, 'collider') and hasattr(obj2, 'collider')):
            return
        
        # Calculate collision normal
        collision_vector = obj2.collider.position - obj1.collider.position
        if collision_vector.magnitude() == 0:
            collision_vector = Vector3D(1, 0, 0)  # Default direction
        normal = collision_vector.normalize()
        
        # Separate objects
        overlap_distance = (obj1.collider.radius + obj2.collider.radius) - collision_vector.magnitude()
        if overlap_distance > 0:
            separation = normal * (overlap_distance / 2)
            obj1.collider.position = obj1.collider.position - separation
            obj2.collider.position = obj2.collider.position + separation
            
            # Update object positions
            if hasattr(obj1, 'position'):
                obj1.position = obj1.collider.position.copy()
            if hasattr(obj2, 'position'):
                obj2.position = obj2.collider.position.copy()
        
        # Calculate relative velocity
        relative_velocity = obj1.velocity - obj2.velocity
        velocity_along_normal = relative_velocity.dot(normal)
        
        # Don't resolve if velocities are separating
        if velocity_along_normal > 0:
            return
        
        # Calculate restitution (bounciness)
        restitution = 0.8
        
        # Calculate impulse scalar
        impulse_scalar = -(1 + restitution) * velocity_along_normal
        impulse_scalar /= (1 / obj1.collider.mass + 1 / obj2.collider.mass)
        
        # Apply impulse
        impulse = normal * impulse_scalar
        obj1.velocity = obj1.velocity + impulse / obj1.collider.mass
        obj2.velocity = obj2.velocity - impulse / obj2.collider.mass
    
    def _resolve_static_collision(self, dynamic_obj, static_obj):
        """Resolve collision between dynamic and static object."""
        if not hasattr(dynamic_obj, 'collider'):
            return
        
        if isinstance(static_obj, BoxCollider):
            # Find closest point on box surface
            closest_point = static_obj.get_closest_point(dynamic_obj.collider.position)
            collision_normal = (dynamic_obj.collider.position - closest_point).normalize()
            
            # Move object out of collision
            penetration_depth = dynamic_obj.collider.radius - dynamic_obj.collider.position.distance_to(closest_point)
            if penetration_depth > 0:
                dynamic_obj.collider.position = dynamic_obj.collider.position + collision_normal * penetration_depth
                if hasattr(dynamic_obj, 'position'):
                    dynamic_obj.position = dynamic_obj.collider.position.copy()
            
            # Reflect velocity
            if hasattr(dynamic_obj, 'velocity'):
                restitution = 0.6
                velocity_along_normal = dynamic_obj.velocity.dot(collision_normal)
                if velocity_along_normal < 0:
                    reflection = collision_normal * velocity_along_normal * (1 + restitution)
                    dynamic_obj.velocity = dynamic_obj.velocity - reflection
    
    def step(self, dt: float):
        """Update physics simulation."""
        # Detect and resolve collisions
        collisions = self.detect_collisions()
        for obj1, obj2 in collisions:
            self.resolve_collision(obj1, obj2)
        
        # Apply gravity to dynamic objects
        for obj in self.dynamic_objects:
            if hasattr(obj, 'velocity') and hasattr(obj, 'use_gravity') and obj.use_gravity:
                obj.velocity = obj.velocity + self.gravity * dt


class SoccerField3D:
    """3D soccer field with boundaries, goals, and colliders."""
    
    def __init__(self, width: float = 100.0, height: float = 60.0, field_height: float = 20.0):
        self.width = width
        self.height = height
        self.field_height = field_height
        self.goal_width = 20.0
        self.goal_height = 8.0
        self.goal_depth = 5.0
        
        # Field boundaries (walls)
        self.boundaries = []
        wall_thickness = 1.0
        
        # Bottom wall
        self.boundaries.append(BoxCollider(
            Vector3D(width/2, -wall_thickness/2, field_height/2),
            Vector3D(width + wall_thickness, wall_thickness, field_height)
        ))
        
        # Top wall  
        self.boundaries.append(BoxCollider(
            Vector3D(width/2, height + wall_thickness/2, field_height/2),
            Vector3D(width + wall_thickness, wall_thickness, field_height)
        ))
        
        # Left wall (with goal opening)
        goal_y_min = (height - self.goal_width) / 2
        goal_y_max = (height + self.goal_width) / 2
        
        # Left wall bottom section
        if goal_y_min > 0:
            self.boundaries.append(BoxCollider(
                Vector3D(-wall_thickness/2, goal_y_min/2, field_height/2),
                Vector3D(wall_thickness, goal_y_min, field_height)
            ))
        
        # Left wall top section
        if goal_y_max < height:
            remaining_height = height - goal_y_max
            self.boundaries.append(BoxCollider(
                Vector3D(-wall_thickness/2, goal_y_max + remaining_height/2, field_height/2),
                Vector3D(wall_thickness, remaining_height, field_height)
            ))
        
        # Right wall (with goal opening)
        # Right wall bottom section
        if goal_y_min > 0:
            self.boundaries.append(BoxCollider(
                Vector3D(width + wall_thickness/2, goal_y_min/2, field_height/2),
                Vector3D(wall_thickness, goal_y_min, field_height)
            ))
        
        # Right wall top section  
        if goal_y_max < height:
            remaining_height = height - goal_y_max
            self.boundaries.append(BoxCollider(
                Vector3D(width + wall_thickness/2, goal_y_max + remaining_height/2, field_height/2),
                Vector3D(wall_thickness, remaining_height, field_height)
            ))
        
        # Goal posts and crossbars
        post_radius = 0.5
        
        # Left goal
        self.left_goal = {
            'position': Vector3D(-self.goal_depth/2, height/2, 0),
            'size': Vector3D(self.goal_depth, self.goal_width, self.goal_height),
            'posts': [
                # Left post bottom
                SphereCollider(Vector3D(0, goal_y_min, 0), post_radius, float('inf')),
                # Left post top
                SphereCollider(Vector3D(0, goal_y_max, 0), post_radius, float('inf')),
                # Left post crossbar left
                SphereCollider(Vector3D(0, goal_y_min, self.goal_height), post_radius, float('inf')),
                # Left post crossbar right
                SphereCollider(Vector3D(0, goal_y_max, self.goal_height), post_radius, float('inf'))
            ]
        }
        
        # Right goal
        self.right_goal = {
            'position': Vector3D(width + self.goal_depth/2, height/2, 0),
            'size': Vector3D(self.goal_depth, self.goal_width, self.goal_height), 
            'posts': [
                # Right post bottom
                SphereCollider(Vector3D(width, goal_y_min, 0), post_radius, float('inf')),
                # Right post top
                SphereCollider(Vector3D(width, goal_y_max, 0), post_radius, float('inf')),
                # Right post crossbar left
                SphereCollider(Vector3D(width, goal_y_min, self.goal_height), post_radius, float('inf')),
                # Right post crossbar right
                SphereCollider(Vector3D(width, goal_y_max, self.goal_height), post_radius, float('inf'))
            ]
        }
        
        # Ground plane
        self.ground = BoxCollider(
            Vector3D(width/2, height/2, -wall_thickness/2),
            Vector3D(width + wall_thickness*2, height + wall_thickness*2, wall_thickness)
        )
    
    def get_all_colliders(self) -> List[Collider]:
        """Get all field colliders for physics engine."""
        colliders = self.boundaries.copy()
        colliders.append(self.ground)
        colliders.extend(self.left_goal['posts'])
        colliders.extend(self.right_goal['posts'])
        return colliders
    
    def is_goal_scored(self, ball_pos: Vector3D, previous_pos: Vector3D) -> Optional[str]:
        """Check if a goal was scored in 3D space."""
        goal_y_min = (self.height - self.goal_width) / 2
        goal_y_max = (self.height + self.goal_width) / 2
        
        # Check if ball is within goal height
        if ball_pos.z < 0 or ball_pos.z > self.goal_height:
            return None
        
        # Check if ball is within goal width
        if ball_pos.y < goal_y_min or ball_pos.y > goal_y_max:
            return None
        
        # Left goal (right team scores)
        if previous_pos.x > 0 and ball_pos.x <= 0:
            return 'right'
        
        # Right goal (left team scores)
        if previous_pos.x < self.width and ball_pos.x >= self.width:
            return 'left'
        
        return None
    
    def keep_in_bounds(self, pos: Vector3D) -> Vector3D:
        """Keep a position within field boundaries (soft bounds)."""
        x = max(-5, min(self.width + 5, pos.x))  # Allow slight overshoot for goals
        y = max(0, min(self.height, pos.y))
        z = max(0, min(self.field_height, pos.z))
        return Vector3D(x, y, z)


class Ball3D:
    """3D soccer ball with physics and collider."""
    
    def __init__(self, position: Vector3D):
        self.position = position.copy()
        self.velocity = Vector3D(0, 0, 0)
        self.friction = 0.98  # Air resistance
        self.ground_friction = 0.95  # Rolling friction when on ground
        self.max_speed = 25.0
        self.radius = 0.5
        self.mass = 0.45  # FIFA regulation ball mass in kg
        self.use_gravity = True
        
        # Create collider
        self.collider = SphereCollider(self.position, self.radius, self.mass)
        
        # Bounce properties
        self.restitution = 0.7  # How bouncy the ball is
        self.min_bounce_velocity = 0.5  # Minimum velocity for bounce
    
    def update(self, dt: float = 1/60):
        """Update ball position and apply physics."""
        # Update collider position
        self.collider.position = self.position.copy()
        
        # Apply physics
        self.position = self.position + self.velocity * dt
        
        # Apply air resistance
        self.velocity = self.velocity * self.friction
        
        # Extra friction when ball is on or near ground
        if self.position.z <= self.radius + 0.1:
            self.velocity = Vector3D(
                self.velocity.x * self.ground_friction,
                self.velocity.y * self.ground_friction,
                self.velocity.z
            )
        
        # Limit speed
        if self.velocity.magnitude() > self.max_speed:
            self.velocity = self.velocity.normalize() * self.max_speed
        
        # Simple ground collision
        if self.position.z < self.radius:
            self.position.z = self.radius
            if self.velocity.z < 0:
                self.velocity.z = -self.velocity.z * self.restitution
                if abs(self.velocity.z) < self.min_bounce_velocity:
                    self.velocity.z = 0
    
    def kick(self, force: Vector3D):
        """Apply a force to the ball."""
        self.velocity = self.velocity + force
    
    def apply_impulse(self, impulse: Vector3D):
        """Apply an impulse to the ball (instantaneous force)."""
        self.velocity = self.velocity + impulse / self.mass


class Agent3D:
    """3D soccer agent with collider and physics."""
    
    def __init__(self, agent_id: int, position: Vector3D, team: str):
        self.id = agent_id
        self.position = position.copy()
        self.velocity = Vector3D(0, 0, 0)
        self.team = team  # 'left' or 'right'
        
        # Physical properties
        self.max_speed = 10.0
        self.radius = 1.0
        self.mass = 75.0  # kg
        self.height = 1.8
        self.use_gravity = True
        
        # Create collider (cylinder approximated as sphere for simplicity)
        self.collider = SphereCollider(self.position, self.radius, self.mass)
        
        # Soccer abilities
        self.kick_range = 2.0
        self.kick_power = 15.0
        self.jump_power = 8.0
        self.is_jumping = False
        
        # Simple Q-learning parameters (extended for 3D)
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.3
        self.q_table = {}
    
    def get_state(self, ball: Ball3D, opponent: 'Agent3D', field: SoccerField3D) -> str:
        """Get current state for learning (simplified discrete state in 3D)."""
        # Discretize positions for Q-table
        my_x = int(self.position.x / 10) * 10
        my_y = int(self.position.y / 10) * 10
        my_z = int(self.position.z / 5) * 5  # Less granular for z
        
        ball_x = int(ball.position.x / 10) * 10
        ball_y = int(ball.position.y / 10) * 10
        ball_z = int(ball.position.z / 5) * 5
        
        ball_dist = self.position.distance_to(ball.position)
        ball_dist_discrete = "close" if ball_dist < 5 else "medium" if ball_dist < 15 else "far"
        
        # Determine if we have ball possession
        has_ball = ball_dist < self.kick_range
        
        # Check if ball is in air
        ball_in_air = ball.position.z > ball.radius + 1.0
        
        # Check if we can jump to reach ball
        can_reach_ball = (ball_dist < self.kick_range * 1.5 and 
                         abs(ball.position.z - self.position.z) < 3.0)
        
        return f"{my_x},{my_y},{my_z},{ball_x},{ball_y},{ball_z},{ball_dist_discrete},{has_ball},{ball_in_air},{can_reach_ball},{self.team}"
    
    def choose_action(self, state: str) -> int:
        """Choose action using epsilon-greedy policy. Extended action space for 3D."""
        actions = list(range(12))  # 12 possible actions in 3D
        
        if random.random() < self.epsilon:
            return random.choice(actions)
        
        if state not in self.q_table:
            self.q_table[state] = [0.0] * len(actions)
        
        return actions[self.q_table[state].index(max(self.q_table[state]))]
    
    def update_q_table(self, state: str, action: int, reward: float, next_state: str):
        """Update Q-table with new experience."""
        if state not in self.q_table:
            self.q_table[state] = [0.0] * 12
        if next_state not in self.q_table:
            self.q_table[next_state] = [0.0] * 12
        
        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state])
        
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state][action] = new_q
    
    def execute_action(self, action: int, ball: Ball3D, field: SoccerField3D):
        """Execute the chosen action in 3D space."""
        # Actions: 0-7 for horizontal movement, 8-9 for vertical, 10-11 for special actions
        if action < 8:
            # Horizontal movement (same as 2D but in 3D)
            directions = [
                Vector3D(0, -1, 0),   # Forward
                Vector3D(1, -1, 0),   # Forward-Right
                Vector3D(1, 0, 0),    # Right
                Vector3D(1, 1, 0),    # Back-Right
                Vector3D(0, 1, 0),    # Back
                Vector3D(-1, 1, 0),   # Back-Left
                Vector3D(-1, 0, 0),   # Left
                Vector3D(-1, -1, 0),  # Forward-Left
            ]
            
            if action < len(directions):
                direction = directions[action]
                self.velocity = Vector3D(
                    direction.x * self.max_speed,
                    direction.y * self.max_speed,
                    self.velocity.z  # Preserve vertical velocity
                )
        
        elif action == 8:
            # Jump
            if self.position.z <= self.radius + 0.1 and not self.is_jumping:
                self.velocity.z = self.jump_power
                self.is_jumping = True
        
        elif action == 9:
            # Stop/brake
            self.velocity = Vector3D(
                self.velocity.x * 0.5,
                self.velocity.y * 0.5,
                self.velocity.z
            )
        
        elif action == 10:
            # Power kick (stronger kick)
            self._attempt_kick(ball, field, power_multiplier=2.0)
        
        elif action == 11:
            # Header/precise kick (when ball is in air)
            if ball.position.z > ball.radius + 0.5:
                self._attempt_kick(ball, field, power_multiplier=1.5, precise=True)
            else:
                self._attempt_kick(ball, field)
        
        # Always try to kick if close to ball (in addition to specific kick actions)
        if action < 8:  # Only during movement actions
            self._attempt_kick(ball, field)
    
    def _attempt_kick(self, ball: Ball3D, field: SoccerField3D, power_multiplier: float = 1.0, precise: bool = False):
        """Attempt to kick the ball."""
        ball_distance = self.position.distance_to(ball.position)
        
        if ball_distance < self.kick_range:
            # Determine kick direction
            if precise:
                # Precise kick towards goal
                if self.team == 'left':
                    target = Vector3D(field.width, field.height / 2, field.goal_height / 3)
                else:
                    target = Vector3D(0, field.height / 2, field.goal_height / 3)
            else:
                # Regular kick towards opponent's goal
                if self.team == 'left':
                    target = Vector3D(field.width, field.height / 2, ball.radius)
                else:
                    target = Vector3D(0, field.height / 2, ball.radius)
            
            kick_direction = (target - ball.position).normalize()
            kick_force = kick_direction * self.kick_power * power_multiplier
            
            # Add some upward component if ball is on ground
            if ball.position.z <= ball.radius + 0.1:
                kick_force.z += self.kick_power * 0.3 * power_multiplier
            
            ball.kick(kick_force)
    
    def update(self, dt: float, field: SoccerField3D):
        """Update agent position and physics."""
        # Update collider position
        self.collider.position = self.position.copy()
        
        # Apply movement
        self.position = self.position + self.velocity * dt
        
        # Keep in bounds
        self.position = field.keep_in_bounds(self.position)
        
        # Ground collision
        if self.position.z < self.radius:
            self.position.z = self.radius
            if self.velocity.z < 0:
                self.velocity.z = 0
                self.is_jumping = False
        
        # Apply friction
        ground_friction = 0.85 if self.position.z <= self.radius + 0.1 else 0.95
        self.velocity = Vector3D(
            self.velocity.x * ground_friction,
            self.velocity.y * ground_friction,
            self.velocity.z * 0.98  # Air resistance
        )


class Soccer3DSimulation:
    """Main 3D simulation class managing the soccer game with physics."""
    
    def __init__(self):
        self.field = SoccerField3D()
        self.ball = Ball3D(Vector3D(self.field.width / 2, self.field.height / 2, 1.0))
        
        # Create agents
        self.agent1 = Agent3D(1, Vector3D(20, self.field.height / 2, 1.0), 'left')
        self.agent2 = Agent3D(2, Vector3D(80, self.field.height / 2, 1.0), 'right')
        
        # Initialize physics engine
        self.physics = PhysicsEngine()
        
        # Add field colliders to physics
        for collider in self.field.get_all_colliders():
            self.physics.add_collider(collider)
        
        # Add dynamic objects to physics
        self.physics.add_dynamic_object(self.ball)
        self.physics.add_dynamic_object(self.agent1)
        self.physics.add_dynamic_object(self.agent2)
        
        # Game state
        self.score = {'left': 0, 'right': 0}
        self.episode_length = 1000  # steps per episode
        self.current_step = 0
        self.episode = 0
        
        # For tracking learning progress
        self.episode_rewards = {'agent1': [], 'agent2': []}
        
        # Physics timestep
        self.physics_dt = 1/60  # 60 FPS physics
    
    def reset_game(self):
        """Reset the game state for a new episode."""
        self.ball.position = Vector3D(self.field.width / 2, self.field.height / 2, 1.0)
        self.ball.velocity = Vector3D(0, 0, 0)
        
        self.agent1.position = Vector3D(20, self.field.height / 2, 1.0)
        self.agent2.position = Vector3D(80, self.field.height / 2, 1.0)
        self.agent1.velocity = Vector3D(0, 0, 0)
        self.agent2.velocity = Vector3D(0, 0, 0)
        self.agent1.is_jumping = False
        self.agent2.is_jumping = False
        
        self.current_step = 0
        self.episode += 1
    
    def calculate_reward(self, agent: Agent3D, goal_scored: Optional[str]) -> float:
        """Calculate reward for an agent."""
        reward = 0.0
        
        # Goal rewards
        if goal_scored:
            if (agent.team == 'left' and goal_scored == 'left') or \
               (agent.team == 'right' and goal_scored == 'right'):
                reward += 100.0  # Scored a goal
            else:
                reward -= 100.0  # Opponent scored
        
        # Distance to ball reward (encourage ball possession)
        ball_distance = agent.position.distance_to(self.ball.position)
        reward += max(0, 20 - ball_distance) * 0.1
        
        # Penalty for being too far from action
        if ball_distance > 30:
            reward -= 0.5
        
        # Reward for good positioning
        if agent.team == 'left':
            # Left team should stay on left side but move towards ball
            if agent.position.x < self.field.width / 2:
                reward += 0.1
        else:
            # Right team should stay on right side but move towards ball
            if agent.position.x > self.field.width / 2:
                reward += 0.1
        
        # Reward for keeping ball on ground vs air control
        ball_height = self.ball.position.z
        if ball_height > self.ball.radius + 2.0:
            # Ball is in air - reward for being close to intercept
            if ball_distance < 5.0:
                reward += 0.2
        
        # Small penalty for excessive jumping
        if agent.is_jumping and ball_distance > 5.0:
            reward -= 0.1
        
        return reward
    
    def step(self) -> Dict[str, Any]:
        """Execute one simulation step with 3D physics."""
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
        
        # Update physics (multiple substeps for stability)
        substeps = 3
        for _ in range(substeps):
            # Update individual object physics
            self.agent1.update(self.physics_dt / substeps, self.field)
            self.agent2.update(self.physics_dt / substeps, self.field)
            self.ball.update(self.physics_dt / substeps)
            
            # Run physics engine
            self.physics.step(self.physics_dt / substeps)
        
        # Keep ball in bounds (soft constraint)
        self.ball.position = self.field.keep_in_bounds(self.ball.position)
        
        # Check for goals
        goal_scored = self.field.is_goal_scored(self.ball.position, prev_ball_pos)
        
        if goal_scored:
            if goal_scored == 'left':
                self.score['left'] += 1
            else:
                self.score['right'] += 1
        
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
            'ball_pos': (self.ball.position.x, self.ball.position.y, self.ball.position.z),
            'ball_velocity': (self.ball.velocity.x, self.ball.velocity.y, self.ball.velocity.z),
            'agent1_pos': (self.agent1.position.x, self.agent1.position.y, self.agent1.position.z),
            'agent2_pos': (self.agent2.position.x, self.agent2.position.y, self.agent2.position.z),
            'agent1_jumping': self.agent1.is_jumping,
            'agent2_jumping': self.agent2.is_jumping
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
            'recent_rewards_agent2': self.episode_rewards['agent2'][-10:] if len(self.episode_rewards['agent2']) >= 10 else self.episode_rewards['agent2'],
            'ball_height': self.ball.position.z,
            'physics_colliders': len(self.physics.colliders),
            'physics_dynamic_objects': len(self.physics.dynamic_objects)
        }
    
    def visualize_ascii_3d(self) -> str:
        """Simple ASCII visualization of the 3D field (top-down view with height info)."""
        width_chars = 50
        height_chars = 30
        
        # Create field representation
        field = [['.' for _ in range(width_chars)] for _ in range(height_chars)]
        
        # Add field boundaries
        for i in range(height_chars):
            field[i][0] = '|'
            field[i][width_chars-1] = '|'
        for j in range(width_chars):
            field[0][j] = '-'
            field[height_chars-1][j] = '-'
        
        # Add goals
        goal_start = height_chars // 2 - 3
        goal_end = height_chars // 2 + 3
        for i in range(goal_start, goal_end):
            field[i][0] = 'G'
            field[i][width_chars-1] = 'G'
        
        # Add ball (with height indicator)
        ball_x = int((self.ball.position.x / self.field.width) * (width_chars - 2)) + 1
        ball_y = int((self.ball.position.y / self.field.height) * (height_chars - 2)) + 1
        ball_x = max(1, min(width_chars-2, ball_x))
        ball_y = max(1, min(height_chars-2, ball_y))
        
        ball_height = int(self.ball.position.z)
        if ball_height > 5:
            field[ball_y][ball_x] = 'O'  # High ball
        elif ball_height > 2:
            field[ball_y][ball_x] = 'o'  # Medium height ball
        else:
            field[ball_y][ball_x] = '*'  # Ground ball
        
        # Add agents
        agent1_x = int((self.agent1.position.x / self.field.width) * (width_chars - 2)) + 1
        agent1_y = int((self.agent1.position.y / self.field.height) * (height_chars - 2)) + 1
        agent1_x = max(1, min(width_chars-2, agent1_x))
        agent1_y = max(1, min(height_chars-2, agent1_y))
        
        agent2_x = int((self.agent2.position.x / self.field.width) * (width_chars - 2)) + 1
        agent2_y = int((self.agent2.position.y / self.field.height) * (height_chars - 2)) + 1
        agent2_x = max(1, min(width_chars-2, agent2_x))
        agent2_y = max(1, min(height_chars-2, agent2_y))
        
        # Agent symbols (with jumping indicator)
        agent1_char = '1' if not self.agent1.is_jumping else '^'
        agent2_char = '2' if not self.agent2.is_jumping else '^'
        
        if ball_x != agent1_x or ball_y != agent1_y:
            field[agent1_y][agent1_x] = agent1_char
        if ball_x != agent2_x or ball_y != agent2_y:
            field[agent2_y][agent2_x] = agent2_char
        
        # Convert to string
        result = []
        for row in field:
            result.append(''.join(row))
        
        # Add info
        result.append('')
        result.append(f"Score: Left {self.score['left']} - {self.score['right']} Right")
        result.append(f"Ball: ({self.ball.position.x:.1f}, {self.ball.position.y:.1f}, {self.ball.position.z:.1f})")
        result.append(f"Ball velocity: ({self.ball.velocity.x:.1f}, {self.ball.velocity.y:.1f}, {self.ball.velocity.z:.1f})")
        result.append(f"Agent 1: ({self.agent1.position.x:.1f}, {self.agent1.position.y:.1f}, {self.agent1.position.z:.1f}) {'[JUMPING]' if self.agent1.is_jumping else ''}")
        result.append(f"Agent 2: ({self.agent2.position.x:.1f}, {self.agent2.position.y:.1f}, {self.agent2.position.z:.1f}) {'[JUMPING]' if self.agent2.is_jumping else ''}")
        result.append("Legend: * = ball on ground, o = ball medium height, O = ball high, ^ = jumping agent")
        
        return '\n'.join(result)
    
    def run_with_3d_visualization(self, episodes: int = 1, use_3d: bool = True) -> Dict[str, Any]:
        """Run simulation with optional 3D visualization."""
        results = []
        
        if use_3d:
            try:
                from renderer_3d import Soccer3DVisualization
                viz = Soccer3DVisualization()
                print("Starting simulation with 3D visualization...")
                viz.run_training_with_visualization(episodes=episodes, steps_per_episode=self.episode_length)
                
                # Return final statistics
                return {
                    'episodes': episodes,
                    'final_score': self.score.copy(),
                    'agent1_q_table_size': len(self.agent1.q_table),
                    'agent2_q_table_size': len(self.agent2.q_table),
                    'visualization': '3D'
                }
                
            except Exception as e:
                print(f"3D visualization failed: {e}")
                print("Falling back to ASCII visualization...")
                use_3d = False
        
        if not use_3d:
            # Fallback to regular ASCII simulation
            for episode in range(episodes):
                result = self.run_episode()
                results.append(result)
                print(f"Episode {episode + 1}/{episodes}: Score {result['final_score']} in {result['steps']} steps")
                if episode < episodes - 1:  # Don't print on last episode
                    print(self.visualize_ascii_3d())
                    print()
            
            return {
                'episodes': episodes,
                'results': results,
                'final_score': self.score.copy(),
                'agent1_q_table_size': len(self.agent1.q_table),
                'agent2_q_table_size': len(self.agent2.q_table),
                'visualization': 'ASCII'
            }