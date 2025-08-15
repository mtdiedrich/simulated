"""
Bot controller for playing Need for Speed: Carbon.

This module implements the actual game controller that uses the trained
neural network to play the game autonomously.
"""

import time
import numpy as np
from typing import List, Optional, Dict, Any
import threading
from queue import Queue

from screen_capture import ScreenCapture
from nfs_bot_model import NFSBotCNN, NFSBotTrainer

try:
    # Try to import input simulation libraries
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False

try:
    from pynput.keyboard import Key, Controller as KeyboardController
    from pynput.mouse import Button, Controller as MouseController
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False


class InputSimulator:
    """Simulates keyboard and mouse inputs for game control."""
    
    def __init__(self):
        """Initialize input simulator."""
        self.method = self._select_input_method()
        
        if self.method == 'pynput':
            self.keyboard = KeyboardController()
            self.mouse = MouseController()
        elif self.method == 'pyautogui':
            pyautogui.FAILSAFE = True  # Move mouse to corner to stop
            
        # Track pressed keys to avoid repeated presses
        self.pressed_keys = set()
        
        # Key mapping for NFS Carbon
        self.key_mapping = {
            'accelerate': 'w',
            'brake': 's', 
            'steer_left': 'a',
            'steer_right': 'd',
            'handbrake': 'space',
            'boost': 'shift',
            'drift': 'ctrl',
            'reset': 'r',
            'pause': 'esc',
            'map': 'tab',
            'camera_change': 'c'
        }
        
    def _select_input_method(self) -> str:
        """Select the best available input simulation method."""
        if PYNPUT_AVAILABLE:
            return 'pynput'
        elif PYAUTOGUI_AVAILABLE:
            return 'pyautogui'
        else:
            return 'none'
            
    def press_key(self, key: str):
        """Press a key."""
        if self.method == 'none':
            return
            
        if key in self.pressed_keys:
            return  # Already pressed
            
        try:
            if self.method == 'pynput':
                if key == 'space':
                    self.keyboard.press(Key.space)
                elif key == 'shift':
                    self.keyboard.press(Key.shift)
                elif key == 'ctrl':
                    self.keyboard.press(Key.ctrl)
                elif key == 'esc':
                    self.keyboard.press(Key.esc)
                elif key == 'tab':
                    self.keyboard.press(Key.tab)
                else:
                    self.keyboard.press(key)
                    
            elif self.method == 'pyautogui':
                pyautogui.keyDown(key)
                
            self.pressed_keys.add(key)
            
        except Exception as e:
            print(f"Error pressing key {key}: {e}")
            
    def release_key(self, key: str):
        """Release a key."""
        if self.method == 'none':
            return
            
        if key not in self.pressed_keys:
            return  # Not pressed
            
        try:
            if self.method == 'pynput':
                if key == 'space':
                    self.keyboard.release(Key.space)
                elif key == 'shift':
                    self.keyboard.release(Key.shift)
                elif key == 'ctrl':
                    self.keyboard.release(Key.ctrl)
                elif key == 'esc':
                    self.keyboard.release(Key.esc)
                elif key == 'tab':
                    self.keyboard.release(Key.tab)
                else:
                    self.keyboard.release(key)
                    
            elif self.method == 'pyautogui':
                pyautogui.keyUp(key)
                
            self.pressed_keys.discard(key)
            
        except Exception as e:
            print(f"Error releasing key {key}: {e}")
            
    def update_inputs(self, predicted_actions: List[str]):
        """
        Update inputs based on predicted actions.
        
        Args:
            predicted_actions: List of predicted game actions
        """
        # Get required keys for predicted actions
        required_keys = set()
        for action in predicted_actions:
            key = self.key_mapping.get(action)
            if key:
                required_keys.add(key)
                
        # Release keys that are no longer needed
        keys_to_release = self.pressed_keys - required_keys
        for key in keys_to_release:
            self.release_key(key)
            
        # Press keys that are newly required
        keys_to_press = required_keys - self.pressed_keys
        for key in keys_to_press:
            self.press_key(key)
            
    def release_all_keys(self):
        """Release all currently pressed keys."""
        keys_to_release = list(self.pressed_keys)
        for key in keys_to_release:
            self.release_key(key)


class NFSBotController:
    """Main controller for the NFS bot."""
    
    def __init__(self, 
                 model_path: str,
                 screen_size: tuple = (224, 224),
                 capture_region: Optional[tuple] = None,
                 prediction_fps: float = 30.0,
                 prediction_threshold: float = 0.5):
        """
        Initialize bot controller.
        
        Args:
            model_path: Path to trained model
            screen_size: Screen capture size
            capture_region: Screen region to capture
            prediction_fps: Prediction frequency
            prediction_threshold: Threshold for action prediction
        """
        self.model_path = model_path
        self.screen_size = screen_size
        self.capture_region = capture_region
        self.prediction_fps = prediction_fps
        self.prediction_threshold = prediction_threshold
        
        # Load model
        self.model = NFSBotCNN(input_size=screen_size, num_actions=11)
        self.trainer = NFSBotTrainer(self.model)
        self.trainer.load_model(model_path)
        
        # Initialize components
        self.screen_capture = ScreenCapture(
            capture_region=capture_region,
            target_size=screen_size,
            capture_fps=prediction_fps * 2  # Capture more frequently than prediction
        )
        
        self.input_simulator = InputSimulator()
        
        # Control state
        self.is_running = False
        self.control_thread = None
        self.frame_queue = Queue(maxsize=5)
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'predictions_made': 0,
            'actions_executed': 0,
            'start_time': None
        }
        
    def start(self):
        """Start the bot controller."""
        if self.is_running:
            return
            
        print("Starting NFS Bot Controller...")
        print(f"Model: {self.model_path}")
        print(f"Screen size: {self.screen_size}")
        print(f"Capture region: {self.capture_region or 'Full screen'}")
        print(f"Prediction FPS: {self.prediction_fps}")
        print(f"Input method: {self.input_simulator.method}")
        
        if self.input_simulator.method == 'none':
            print("WARNING: No input simulation available. Bot will only make predictions.")
            
        self.is_running = True
        self.stats['start_time'] = time.time()
        
        # Start screen capture
        self.screen_capture.start_continuous_capture(self._on_frame_captured)
        
        # Start control thread
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()
        
        print("Bot controller started!")
        
    def stop(self):
        """Stop the bot controller."""
        if not self.is_running:
            return
            
        print("\nStopping bot controller...")
        
        self.is_running = False
        
        # Stop screen capture
        self.screen_capture.stop_continuous_capture()
        
        # Release all keys
        self.input_simulator.release_all_keys()
        
        # Wait for control thread
        if self.control_thread and self.control_thread.is_alive():
            self.control_thread.join(timeout=2.0)
            
        print("Bot controller stopped!")
        self._print_stats()
        
    def _on_frame_captured(self, frame: np.ndarray):
        """Handle captured screen frame."""
        if not self.is_running:
            return
            
        # Add frame to queue (drop oldest if full)
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except:
                pass
                
        self.frame_queue.put(frame)
        self.stats['frames_processed'] += 1
        
    def _control_loop(self):
        """Main control loop."""
        last_prediction_time = 0
        prediction_interval = 1.0 / self.prediction_fps
        
        while self.is_running:
            current_time = time.time()
            
            # Check if it's time for a new prediction
            if current_time - last_prediction_time >= prediction_interval:
                try:
                    # Get latest frame
                    frame = self.frame_queue.get(timeout=0.1)
                    
                    # Make prediction
                    predicted_actions = self.model.predict_actions(
                        frame, 
                        threshold=self.prediction_threshold
                    )
                    
                    # Update inputs
                    self.input_simulator.update_inputs(predicted_actions)
                    
                    # Update statistics
                    self.stats['predictions_made'] += 1
                    if predicted_actions:
                        self.stats['actions_executed'] += len(predicted_actions)
                        
                    last_prediction_time = current_time
                    
                except:
                    # No frame available or other error
                    continue
                    
            # Small sleep to prevent excessive CPU usage
            time.sleep(0.001)
            
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current runtime statistics."""
        if self.stats['start_time']:
            runtime = time.time() - self.stats['start_time']
        else:
            runtime = 0
            
        return {
            'runtime_seconds': runtime,
            'frames_processed': self.stats['frames_processed'],
            'predictions_made': self.stats['predictions_made'],
            'actions_executed': self.stats['actions_executed'],
            'fps': self.stats['frames_processed'] / max(runtime, 1),
            'prediction_rate': self.stats['predictions_made'] / max(runtime, 1)
        }
        
    def _print_stats(self):
        """Print runtime statistics."""
        stats = self.get_current_stats()
        
        print("\nBot Controller Statistics:")
        print(f"  Runtime: {stats['runtime_seconds']:.1f} seconds")
        print(f"  Frames processed: {stats['frames_processed']}")
        print(f"  Predictions made: {stats['predictions_made']}")
        print(f"  Actions executed: {stats['actions_executed']}")
        print(f"  Average FPS: {stats['fps']:.1f}")
        print(f"  Prediction rate: {stats['prediction_rate']:.1f} Hz")
        
    def run_with_monitoring(self, duration_seconds: Optional[float] = None):
        """
        Run the bot with live monitoring.
        
        Args:
            duration_seconds: Optional maximum runtime
        """
        self.start()
        
        try:
            start_time = time.time()
            
            while self.is_running:
                # Print live statistics
                stats = self.get_current_stats()
                print(f"\rRuntime: {stats['runtime_seconds']:.1f}s | "
                      f"FPS: {stats['fps']:.1f} | "
                      f"Predictions: {stats['predictions_made']} | "
                      f"Actions: {stats['actions_executed']}", end="")
                
                # Check duration limit
                if duration_seconds and stats['runtime_seconds'] >= duration_seconds:
                    break
                    
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            
        finally:
            self.stop()


def demo_bot_controller():
    """Demonstrate bot controller with a dummy model."""
    print("Bot Controller Demo")
    print("Note: This is a demo without actual input simulation for safety")
    
    # This would normally use a real trained model
    print("For a real implementation:")
    print("1. Train a model using train_nfs_bot.py")
    print("2. Use the trained model path with NFSBotController")
    print("3. Ensure proper input simulation libraries are installed")
    print("4. Configure screen capture region for your game window")


if __name__ == "__main__":
    demo_bot_controller()