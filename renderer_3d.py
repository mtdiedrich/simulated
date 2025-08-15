"""
3D Renderer for Soccer Simulation

Provides OpenGL-based 3D visualization for the soccer simulation with physics.
Renders the field, agents, and ball as 3D models with proper lighting and perspective.
"""

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import math
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from soccer_simulation_3d import Soccer3DSimulation, Vector3D


class Renderer3D:
    """3D OpenGL renderer for soccer simulation."""
    
    def __init__(self, width: int = 1024, height: int = 768, title: str = "3D Soccer Simulation"):
        self.width = width
        self.height = height
        self.title = title
        
        # Camera parameters
        self.camera_pos = Vector3D(50, -80, 40)  # Position camera above and behind center field
        self.camera_target = Vector3D(50, 30, 0)  # Look at center of field
        self.camera_up = Vector3D(0, 0, 1)  # Z-axis is up
        
        # Lighting parameters
        self.light_pos = [50, 30, 50, 1.0]  # Light position above center field
        self.ambient_light = [0.3, 0.3, 0.3, 1.0]
        self.diffuse_light = [0.8, 0.8, 0.8, 1.0]
        self.specular_light = [1.0, 1.0, 1.0, 1.0]
        
        # Colors
        self.colors = {
            'field': (0.2, 0.8, 0.2),      # Green field
            'field_lines': (1.0, 1.0, 1.0), # White lines
            'agent1': (0.0, 0.0, 1.0),     # Blue agent 1
            'agent2': (1.0, 0.0, 0.0),     # Red agent 2
            'ball': (1.0, 1.0, 0.0),       # Yellow ball
            'goal': (0.8, 0.8, 0.8),       # Gray goals
            'walls': (0.6, 0.4, 0.2),      # Brown walls
        }
        
        self.running = False
        
    def initialize(self) -> bool:
        """Initialize pygame and OpenGL."""
        try:
            # Try to detect if we're in a headless environment
            import os
            if 'DISPLAY' not in os.environ and os.name == 'posix':
                print("No display available - running in headless mode")
                print("3D visualization disabled, falling back to ASCII")
                return False
                
            pygame.init()
            pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL)
            pygame.display.set_caption(self.title)
            
            # Set up OpenGL
            self._setup_opengl()
            
            self.running = True
            return True
            
        except Exception as e:
            print(f"Failed to initialize 3D renderer: {e}")
            print("Falling back to ASCII visualization")
            return False
    
    def _setup_opengl(self):
        """Configure OpenGL settings."""
        # Enable depth testing
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        
        # Enable lighting
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        
        # Set up lighting
        glLightfv(GL_LIGHT0, GL_POSITION, self.light_pos)
        glLightfv(GL_LIGHT0, GL_AMBIENT, self.ambient_light)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, self.diffuse_light)
        glLightfv(GL_LIGHT0, GL_SPECULAR, self.specular_light)
        
        # Enable smooth shading
        glShadeModel(GL_SMOOTH)
        
        # Set up perspective
        glMatrixMode(GL_PROJECTION)
        gluPerspective(45, (self.width / self.height), 0.1, 500.0)
        
        # Set up camera
        glMatrixMode(GL_MODELVIEW)
        gluLookAt(
            self.camera_pos.x, self.camera_pos.y, self.camera_pos.z,
            self.camera_target.x, self.camera_target.y, self.camera_target.z,
            self.camera_up.x, self.camera_up.y, self.camera_up.z
        )
        
        # Set background color (sky blue)
        glClearColor(0.5, 0.8, 1.0, 1.0)
    
    def _set_material_color(self, color: Tuple[float, float, float], shininess: float = 50.0):
        """Set material properties for current color."""
        ambient = [c * 0.3 for c in color] + [1.0]
        diffuse = [c * 0.8 for c in color] + [1.0]
        specular = [0.5, 0.5, 0.5, 1.0]
        
        glMaterialfv(GL_FRONT, GL_AMBIENT, ambient)
        glMaterialfv(GL_FRONT, GL_DIFFUSE, diffuse)
        glMaterialfv(GL_FRONT, GL_SPECULAR, specular)
        glMaterialf(GL_FRONT, GL_SHININESS, shininess)
    
    def _draw_cube(self, width: float, height: float, depth: float):
        """Draw a cube with given dimensions."""
        w, h, d = width/2, height/2, depth/2
        
        glBegin(GL_QUADS)
        
        # Front face
        glNormal3f(0, 0, 1)
        glVertex3f(-w, -h, d)
        glVertex3f(w, -h, d)
        glVertex3f(w, h, d)
        glVertex3f(-w, h, d)
        
        # Back face
        glNormal3f(0, 0, -1)
        glVertex3f(-w, -h, -d)
        glVertex3f(-w, h, -d)
        glVertex3f(w, h, -d)
        glVertex3f(w, -h, -d)
        
        # Top face
        glNormal3f(0, 1, 0)
        glVertex3f(-w, h, -d)
        glVertex3f(-w, h, d)
        glVertex3f(w, h, d)
        glVertex3f(w, h, -d)
        
        # Bottom face
        glNormal3f(0, -1, 0)
        glVertex3f(-w, -h, -d)
        glVertex3f(w, -h, -d)
        glVertex3f(w, -h, d)
        glVertex3f(-w, -h, d)
        
        # Right face
        glNormal3f(1, 0, 0)
        glVertex3f(w, -h, -d)
        glVertex3f(w, h, -d)
        glVertex3f(w, h, d)
        glVertex3f(w, -h, d)
        
        # Left face
        glNormal3f(-1, 0, 0)
        glVertex3f(-w, -h, -d)
        glVertex3f(-w, -h, d)
        glVertex3f(-w, h, d)
        glVertex3f(-w, h, -d)
        
        glEnd()
    
    def _draw_sphere(self, radius: float, slices: int = 20, stacks: int = 20):
        """Draw a sphere with given radius."""
        for i in range(stacks):
            lat0 = math.pi * (-0.5 + (i / stacks))
            z0 = radius * math.sin(lat0)
            zr0 = radius * math.cos(lat0)
            
            lat1 = math.pi * (-0.5 + ((i + 1) / stacks))
            z1 = radius * math.sin(lat1)
            zr1 = radius * math.cos(lat1)
            
            glBegin(GL_QUAD_STRIP)
            for j in range(slices + 1):
                lng = 2 * math.pi * (j / slices)
                x = math.cos(lng)
                y = math.sin(lng)
                
                glNormal3f(x * zr0, y * zr0, z0)
                glVertex3f(x * zr0, y * zr0, z0)
                glNormal3f(x * zr1, y * zr1, z1)
                glVertex3f(x * zr1, y * zr1, z1)
            glEnd()
    
    def _draw_cylinder(self, radius: float, height: float, slices: int = 20):
        """Draw a cylinder with given radius and height."""
        # Draw sides
        glBegin(GL_QUAD_STRIP)
        for i in range(slices + 1):
            angle = 2 * math.pi * i / slices
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            
            glNormal3f(x/radius, y/radius, 0)
            glVertex3f(x, y, 0)
            glVertex3f(x, y, height)
        glEnd()
        
        # Draw top and bottom circles
        glBegin(GL_TRIANGLE_FAN)
        glNormal3f(0, 0, 1)
        glVertex3f(0, 0, height)
        for i in range(slices + 1):
            angle = 2 * math.pi * i / slices
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            glVertex3f(x, y, height)
        glEnd()
        
        glBegin(GL_TRIANGLE_FAN)
        glNormal3f(0, 0, -1)
        glVertex3f(0, 0, 0)
        for i in range(slices, -1, -1):
            angle = 2 * math.pi * i / slices
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            glVertex3f(x, y, 0)
        glEnd()
    
    def _draw_field(self, simulation: Soccer3DSimulation):
        """Draw the soccer field."""
        field = simulation.field
        
        # Draw field surface
        self._set_material_color(self.colors['field'])
        glPushMatrix()
        glTranslatef(field.width/2, field.height/2, 0)
        glScalef(field.width, field.height, 0.1)
        self._draw_cube(1, 1, 1)
        glPopMatrix()
        
        # Draw field lines
        self._set_material_color(self.colors['field_lines'])
        line_width = 0.5
        line_height = 0.05
        
        # Center line
        glPushMatrix()
        glTranslatef(field.width/2, field.height/2, line_height)
        glScalef(line_width, field.height, line_height)
        self._draw_cube(1, 1, 1)
        glPopMatrix()
        
        # Center circle
        glPushMatrix()
        glTranslatef(field.width/2, field.height/2, line_height)
        self._draw_cylinder(field.center_circle_radius, line_height, 32)
        glPopMatrix()
        
        # Goal areas
        goal_area_width = 16
        goal_area_depth = 5
        
        # Left goal area
        glPushMatrix()
        glTranslatef(goal_area_depth/2, field.height/2, line_height)
        glScalef(goal_area_depth, goal_area_width, line_height)
        self._draw_cube(1, 1, 1)
        glPopMatrix()
        
        # Right goal area
        glPushMatrix()
        glTranslatef(field.width - goal_area_depth/2, field.height/2, line_height)
        glScalef(goal_area_depth, goal_area_width, line_height)
        self._draw_cube(1, 1, 1)
        glPopMatrix()
        
        # Draw goals
        self._set_material_color(self.colors['goal'])
        goal_width = field.goal_width
        goal_height = field.goal_height
        goal_depth = 2
        
        # Left goal
        glPushMatrix()
        glTranslatef(-goal_depth/2, field.height/2, goal_height/2)
        glScalef(goal_depth, goal_width, goal_height)
        self._draw_cube(1, 1, 1)
        glPopMatrix()
        
        # Right goal
        glPushMatrix()
        glTranslatef(field.width + goal_depth/2, field.height/2, goal_height/2)
        glScalef(goal_depth, goal_width, goal_height)
        self._draw_cube(1, 1, 1)
        glPopMatrix()
        
        # Draw walls
        self._set_material_color(self.colors['walls'])
        wall_height = 10
        wall_thickness = 1
        
        # Side walls
        glPushMatrix()
        glTranslatef(field.width/2, -wall_thickness/2, wall_height/2)
        glScalef(field.width, wall_thickness, wall_height)
        self._draw_cube(1, 1, 1)
        glPopMatrix()
        
        glPushMatrix()
        glTranslatef(field.width/2, field.height + wall_thickness/2, wall_height/2)
        glScalef(field.width, wall_thickness, wall_height)
        self._draw_cube(1, 1, 1)
        glPopMatrix()
    
    def _draw_agent(self, agent, team_color: Tuple[float, float, float]):
        """Draw a 3D agent/player."""
        self._set_material_color(team_color)
        
        # Agent is represented as a cylinder (like a humanoid figure)
        glPushMatrix()
        glTranslatef(agent.position.x, agent.position.y, agent.position.z)
        
        # Scale based on whether jumping (make taller when jumping)
        scale_z = 1.3 if agent.is_jumping else 1.0
        glScalef(1, 1, scale_z)
        
        self._draw_cylinder(agent.radius, agent.height, 16)
        glPopMatrix()
    
    def _draw_ball(self, ball):
        """Draw the 3D ball."""
        self._set_material_color(self.colors['ball'])
        
        glPushMatrix()
        glTranslatef(ball.position.x, ball.position.y, ball.position.z)
        self._draw_sphere(ball.radius, 16, 16)
        glPopMatrix()
    
    def render_frame(self, simulation: Soccer3DSimulation):
        """Render a single frame of the simulation."""
        if not self.running:
            return False
            
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                    return False
        
        # Clear the screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Reset modelview matrix
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(
            self.camera_pos.x, self.camera_pos.y, self.camera_pos.z,
            self.camera_target.x, self.camera_target.y, self.camera_target.z,
            self.camera_up.x, self.camera_up.y, self.camera_up.z
        )
        
        # Draw field
        self._draw_field(simulation)
        
        # Draw agents
        self._draw_agent(simulation.agent1, self.colors['agent1'])
        self._draw_agent(simulation.agent2, self.colors['agent2'])
        
        # Draw ball
        self._draw_ball(simulation.ball)
        
        # Swap buffers
        pygame.display.flip()
        
        return True
    
    def cleanup(self):
        """Clean up resources."""
        self.running = False
        pygame.quit()
    
    def is_running(self) -> bool:
        """Check if renderer is still running."""
        return self.running


class Soccer3DVisualization:
    """Main class for running 3D soccer simulation with visualization."""
    
    def __init__(self, renderer_width: int = 1024, renderer_height: int = 768):
        self.simulation = Soccer3DSimulation()
        self.renderer = Renderer3D(renderer_width, renderer_height)
        self.clock = pygame.time.Clock()
        self.fps = 60
        
    def run_interactive(self, episodes: int = 1, steps_per_episode: int = 1000):
        """Run the simulation with interactive 3D visualization."""
        if not self.renderer.initialize():
            print("Failed to initialize 3D renderer")
            return
        
        print(f"Starting 3D Soccer Simulation Visualization")
        print(f"Episodes: {episodes}")
        print(f"Controls: ESC to quit, simulation runs automatically")
        print()
        
        try:
            for episode in range(episodes):
                print(f"Episode {episode + 1}/{episodes}")
                self.simulation.reset_game()
                
                for step in range(steps_per_episode):
                    # Run simulation step
                    result = self.simulation.step()
                    
                    # Render frame
                    if not self.renderer.render_frame(self.simulation):
                        break
                    
                    # Control frame rate
                    self.clock.tick(self.fps)
                    
                    # Check if episode is done
                    if result['done']:
                        print(f"Episode finished after {step + 1} steps")
                        print(f"Final score: Left {result['score']['left']} - {result['score']['right']} Right")
                        break
                
                if not self.renderer.is_running():
                    break
                    
        finally:
            self.renderer.cleanup()
    
    def run_training_with_visualization(self, episodes: int = 10, steps_per_episode: int = 1000):
        """Run training with periodic 3D visualization."""
        if not self.renderer.initialize():
            print("Failed to initialize 3D renderer - running without visualization")
            # Fall back to text-only training
            for episode in range(episodes):
                result = self.simulation.run_episode()
                print(f"Episode {episode + 1}: Score {result['final_score']} in {result['steps']} steps")
            return
        
        print(f"Starting 3D Soccer Training with Visualization")
        print(f"Episodes: {episodes}")
        print()
        
        try:
            for episode in range(episodes):
                print(f"Episode {episode + 1}/{episodes}")
                self.simulation.reset_game()
                
                for step in range(steps_per_episode):
                    # Run simulation step
                    result = self.simulation.step()
                    
                    # Render every few frames to maintain performance
                    if step % 3 == 0:  # Render every 3rd step
                        if not self.renderer.render_frame(self.simulation):
                            break
                        self.clock.tick(60)  # 60 FPS for visualization
                    
                    # Check if episode is done
                    if result['done']:
                        print(f"Episode finished after {step + 1} steps")
                        print(f"Final score: Left {result['score']['left']} - {result['score']['right']} Right")
                        stats = self.simulation.get_statistics()
                        print(f"Agent 1 Q-table size: {stats['agent1_q_table_size']}")
                        print(f"Agent 2 Q-table size: {stats['agent2_q_table_size']}")
                        print()
                        break
                
                if not self.renderer.is_running():
                    break
                    
        finally:
            self.renderer.cleanup()


def main():
    """Demo the 3D visualization."""
    viz = Soccer3DVisualization()
    
    print("3D Soccer Simulation Demo")
    print("Choose mode:")
    print("1. Interactive visualization (1 episode)")
    print("2. Training with visualization (5 episodes)")
    print("3. Just render current state")
    
    try:
        choice = input("Enter choice (1-3): ").strip()
    except KeyboardInterrupt:
        print("\nExiting...")
        return
    
    if choice == "1":
        viz.run_interactive(episodes=1, steps_per_episode=1000)
    elif choice == "2":
        viz.run_training_with_visualization(episodes=5, steps_per_episode=500)
    elif choice == "3":
        # Just show a static render
        if viz.renderer.initialize():
            viz.simulation.reset_game()
            print("Rendering static frame - press ESC to close")
            while viz.renderer.render_frame(viz.simulation):
                viz.clock.tick(30)
            viz.renderer.cleanup()
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()