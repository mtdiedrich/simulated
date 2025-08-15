"""
Input capture module for recording keyboard and mouse inputs.

This module provides functionality to monitor and record user inputs
during gameplay for training data collection.
"""

import time
import threading
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass
from queue import Queue
import json

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

try:
    # Try to import pynput for cross-platform input monitoring
    from pynput import keyboard, mouse
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False


@dataclass
class InputEvent:
    """Represents a single input event."""
    timestamp: float
    event_type: str  # 'key_press', 'key_release', 'mouse_move', 'mouse_click', 'mouse_release'
    key: Optional[str] = None
    mouse_pos: Optional[Tuple[int, int]] = None
    mouse_button: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp,
            'event_type': self.event_type,
            'key': self.key,
            'mouse_pos': self.mouse_pos,
            'mouse_button': self.mouse_button
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InputEvent':
        """Create from dictionary."""
        return cls(**data)


class InputCapture:
    """Handles input capture for training data collection."""
    
    def __init__(self, capture_keyboard: bool = True, capture_mouse: bool = True):
        """
        Initialize input capture.
        
        Args:
            capture_keyboard: Whether to capture keyboard events
            capture_mouse: Whether to capture mouse events
        """
        self.capture_keyboard = capture_keyboard
        self.capture_mouse = capture_mouse
        
        self.is_capturing = False
        self.events_queue = Queue()
        self.event_callback = None
        
        # Input state tracking
        self.pressed_keys = set()
        self.mouse_position = (0, 0)
        self.pressed_mouse_buttons = set()
        
        # Listeners
        self.keyboard_listener = None
        self.mouse_listener = None
        
        # Choose input monitoring method
        self.input_method = self._select_input_method()
        
    def _select_input_method(self) -> str:
        """Select the best available input monitoring method."""
        if PYNPUT_AVAILABLE:
            return 'pynput'  # Best cross-platform support
        elif PYGAME_AVAILABLE:
            return 'pygame'  # Good for game development
        else:
            return 'polling'  # Fallback polling method
            
    def get_current_state(self) -> Dict[str, Any]:
        """
        Get current input state snapshot.
        
        Returns:
            Dictionary containing current pressed keys, mouse position, etc.
        """
        return {
            'timestamp': time.time(),
            'pressed_keys': list(self.pressed_keys),
            'mouse_position': self.mouse_position,
            'pressed_mouse_buttons': list(self.pressed_mouse_buttons)
        }
        
    def start_capture(self, event_callback: Optional[Callable[[InputEvent], None]] = None):
        """
        Start input capture.
        
        Args:
            event_callback: Optional callback function called for each input event
        """
        if self.is_capturing:
            return
            
        self.event_callback = event_callback
        self.is_capturing = True
        
        if self.input_method == 'pynput':
            self._start_pynput_capture()
        elif self.input_method == 'pygame':
            self._start_pygame_capture()
        else:
            self._start_polling_capture()
            
    def stop_capture(self):
        """Stop input capture."""
        self.is_capturing = False
        
        if self.keyboard_listener:
            self.keyboard_listener.stop()
            self.keyboard_listener = None
            
        if self.mouse_listener:
            self.mouse_listener.stop()
            self.mouse_listener = None
            
    def _start_pynput_capture(self):
        """Start input capture using pynput."""
        if self.capture_keyboard:
            self.keyboard_listener = keyboard.Listener(
                on_press=self._on_key_press,
                on_release=self._on_key_release
            )
            self.keyboard_listener.start()
            
        if self.capture_mouse:
            self.mouse_listener = mouse.Listener(
                on_move=self._on_mouse_move,
                on_click=self._on_mouse_click
            )
            self.mouse_listener.start()
            
    def _start_pygame_capture(self):
        """Start input capture using pygame (requires event polling)."""
        # This would require pygame event loop integration
        # For now, just set up the framework
        pass
        
    def _start_polling_capture(self):
        """Start input capture using polling method."""
        # Fallback method - would need platform-specific implementation
        pass
        
    def _on_key_press(self, key):
        """Handle keyboard key press event."""
        try:
            key_str = key.char if hasattr(key, 'char') and key.char else str(key)
        except AttributeError:
            key_str = str(key)
            
        self.pressed_keys.add(key_str)
        
        event = InputEvent(
            timestamp=time.time(),
            event_type='key_press',
            key=key_str
        )
        
        self._handle_event(event)
        
    def _on_key_release(self, key):
        """Handle keyboard key release event."""
        try:
            key_str = key.char if hasattr(key, 'char') and key.char else str(key)
        except AttributeError:
            key_str = str(key)
            
        self.pressed_keys.discard(key_str)
        
        event = InputEvent(
            timestamp=time.time(),
            event_type='key_release',
            key=key_str
        )
        
        self._handle_event(event)
        
    def _on_mouse_move(self, x, y):
        """Handle mouse movement event."""
        self.mouse_position = (x, y)
        
        event = InputEvent(
            timestamp=time.time(),
            event_type='mouse_move',
            mouse_pos=(x, y)
        )
        
        self._handle_event(event)
        
    def _on_mouse_click(self, x, y, button, pressed):
        """Handle mouse click/release event."""
        button_str = str(button)
        
        if pressed:
            self.pressed_mouse_buttons.add(button_str)
            event_type = 'mouse_click'
        else:
            self.pressed_mouse_buttons.discard(button_str)
            event_type = 'mouse_release'
            
        event = InputEvent(
            timestamp=time.time(),
            event_type=event_type,
            mouse_pos=(x, y),
            mouse_button=button_str
        )
        
        self._handle_event(event)
        
    def _handle_event(self, event: InputEvent):
        """Handle an input event."""
        self.events_queue.put(event)
        
        if self.event_callback:
            self.event_callback(event)
            
    def get_events(self) -> List[InputEvent]:
        """Get all captured events and clear the queue."""
        events = []
        while not self.events_queue.empty():
            events.append(self.events_queue.get())
        return events
        
    def save_events(self, events: List[InputEvent], filepath: str):
        """Save events to a JSON file."""
        data = [event.to_dict() for event in events]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
    def load_events(self, filepath: str) -> List[InputEvent]:
        """Load events from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return [InputEvent.from_dict(event_data) for event_data in data]


class GameInputMapper:
    """Maps raw input events to game-specific actions."""
    
    def __init__(self, input_mapping: Optional[Dict[str, str]] = None):
        """
        Initialize input mapper.
        
        Args:
            input_mapping: Dictionary mapping raw inputs to game actions
        """
        self.input_mapping = input_mapping or self._get_default_nfs_mapping()
        
    def _get_default_nfs_mapping(self) -> Dict[str, str]:
        """Get default key mapping for Need for Speed: Carbon."""
        return {
            'w': 'accelerate',
            's': 'brake',
            'a': 'steer_left',
            'd': 'steer_right',
            'space': 'handbrake',
            'shift': 'boost',
            'ctrl': 'drift',
            'r': 'reset',
            'esc': 'pause',
            'tab': 'map',
            'c': 'camera_change'
        }
        
    def map_input_event(self, event: InputEvent) -> Optional[str]:
        """
        Map an input event to a game action.
        
        Args:
            event: Input event to map
            
        Returns:
            Game action string or None if no mapping
        """
        if event.event_type in ['key_press', 'key_release'] and event.key:
            key = event.key.lower()
            return self.input_mapping.get(key)
        return None
        
    def get_current_actions(self, input_state: Dict[str, Any]) -> List[str]:
        """
        Get currently active game actions based on input state.
        
        Args:
            input_state: Current input state from InputCapture
            
        Returns:
            List of active game actions
        """
        actions = []
        for key in input_state.get('pressed_keys', []):
            action = self.input_mapping.get(key.lower())
            if action:
                actions.append(action)
        return actions


def test_input_capture():
    """Test the input capture functionality."""
    print("Testing input capture...")
    print(f"Available input methods: {PYNPUT_AVAILABLE and 'pynput' or 'fallback'}")
    
    if not PYNPUT_AVAILABLE:
        print("Warning: pynput not available, input capture will be limited")
        return
        
    capture = InputCapture()
    events_captured = []
    
    def event_callback(event):
        events_captured.append(event)
        print(f"Captured: {event.event_type} - {event.key or event.mouse_button or 'move'}")
        
    print("\nStarting input capture for 5 seconds...")
    print("Please press some keys or move the mouse...")
    
    capture.start_capture(event_callback)
    time.sleep(5)
    capture.stop_capture()
    
    print(f"\nCaptured {len(events_captured)} events")
    
    # Test input mapping
    mapper = GameInputMapper()
    print("\nTesting input mapping:")
    for event in events_captured[:5]:  # Show first 5 events
        action = mapper.map_input_event(event)
        if action:
            print(f"  {event.key} -> {action}")
            
    # Test saving/loading
    if events_captured:
        print("\nTesting save/load...")
        capture.save_events(events_captured, 'test_inputs.json')
        loaded_events = capture.load_events('test_inputs.json')
        print(f"Saved and loaded {len(loaded_events)} events")


if __name__ == "__main__":
    test_input_capture()