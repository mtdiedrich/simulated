"""
Data collection system for recording gameplay sessions.

This module combines screen capture and input monitoring to create
training datasets for the neural network.
"""

import os
import time
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
from queue import Queue

from screen_capture import ScreenCapture
from input_capture import InputCapture, InputEvent, GameInputMapper


@dataclass
class TrainingFrame:
    """Represents a single training frame with screen and input data."""
    timestamp: float
    frame_id: int
    screen_data: np.ndarray  # Screen capture
    input_state: Dict[str, Any]  # Current input state
    actions: List[str]  # Active game actions
    metadata: Dict[str, Any] = None  # Additional metadata
    
    def to_dict(self, save_screen_separately: bool = True) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        if save_screen_separately:
            # Don't include screen data in JSON, save separately
            data.pop('screen_data', None)
        return data


class DataCollector:
    """Collects training data by recording screen and inputs simultaneously."""
    
    def __init__(self, 
                 output_dir: str = "training_data",
                 screen_size: Tuple[int, int] = (224, 224),
                 capture_fps: float = 10.0,
                 capture_region: Optional[Tuple[int, int, int, int]] = None):
        """
        Initialize data collector.
        
        Args:
            output_dir: Directory to save training data
            screen_size: Target size for screen captures
            capture_fps: Frames per second for data collection
            capture_region: Screen region to capture (x, y, width, height)
        """
        self.output_dir = Path(output_dir)
        self.screen_size = screen_size
        self.capture_fps = capture_fps
        self.capture_region = capture_region
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize capture systems
        self.screen_capture = ScreenCapture(
            capture_region=capture_region,
            target_size=screen_size,
            capture_fps=capture_fps
        )
        
        self.input_capture = InputCapture()
        self.input_mapper = GameInputMapper()
        
        # Collection state
        self.is_collecting = False
        self.collection_thread = None
        self.frame_queue = Queue()
        
        # Session data
        self.current_session = None
        self.session_start_time = None
        self.frame_counter = 0
        
    def start_session(self, session_name: Optional[str] = None) -> str:
        """
        Start a new data collection session.
        
        Args:
            session_name: Optional name for the session
            
        Returns:
            Session ID
        """
        if self.is_collecting:
            self.stop_session()
            
        # Generate session ID
        if session_name is None:
            session_name = f"session_{int(time.time())}"
            
        self.current_session = session_name
        self.session_start_time = time.time()
        self.frame_counter = 0
        
        # Create session directory
        session_dir = self.output_dir / session_name
        session_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (session_dir / "frames").mkdir(exist_ok=True)
        (session_dir / "screens").mkdir(exist_ok=True)
        
        print(f"Started data collection session: {session_name}")
        return session_name
        
    def start_collection(self):
        """Start collecting training data."""
        if not self.current_session:
            raise ValueError("No active session. Call start_session() first.")
            
        if self.is_collecting:
            return
            
        self.is_collecting = True
        
        # Start input capture
        self.input_capture.start_capture()
        
        # Start screen capture with frame callback
        self.screen_capture.start_continuous_capture(self._on_frame_captured)
        
        # Start data processing thread
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        
        print("Data collection started")
        
    def stop_collection(self):
        """Stop collecting training data."""
        if not self.is_collecting:
            return
            
        self.is_collecting = False
        
        # Stop capture systems
        self.screen_capture.stop_continuous_capture()
        self.input_capture.stop_capture()
        
        # Wait for collection thread to finish
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=2.0)
            
        print("Data collection stopped")
        
    def stop_session(self):
        """Stop current session and save metadata."""
        if not self.current_session:
            return
            
        self.stop_collection()
        
        # Save session metadata
        session_dir = self.output_dir / self.current_session
        metadata = {
            'session_name': self.current_session,
            'start_time': self.session_start_time,
            'end_time': time.time(),
            'total_frames': self.frame_counter,
            'screen_size': self.screen_size,
            'capture_fps': self.capture_fps,
            'capture_region': self.capture_region
        }
        
        with open(session_dir / "session_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"Session {self.current_session} stopped. Collected {self.frame_counter} frames.")
        self.current_session = None
        
    def _on_frame_captured(self, screen_frame: np.ndarray):
        """Handle captured screen frame."""
        if not self.is_collecting:
            return
            
        # Get current input state
        input_state = self.input_capture.get_current_state()
        
        # Map inputs to game actions
        actions = self.input_mapper.get_current_actions(input_state)
        
        # Create training frame
        training_frame = TrainingFrame(
            timestamp=time.time(),
            frame_id=self.frame_counter,
            screen_data=screen_frame,
            input_state=input_state,
            actions=actions,
            metadata={'session': self.current_session}
        )
        
        # Queue for processing
        self.frame_queue.put(training_frame)
        self.frame_counter += 1
        
    def _collection_loop(self):
        """Main collection loop for processing frames."""
        session_dir = self.output_dir / self.current_session
        frames_data = []
        
        while self.is_collecting or not self.frame_queue.empty():
            try:
                # Get frame from queue (with timeout)
                frame = self.frame_queue.get(timeout=1.0)
                
                # Save screen data
                screen_filename = f"screen_{frame.frame_id:06d}.npy"
                screen_path = session_dir / "screens" / screen_filename
                np.save(screen_path, frame.screen_data)
                
                # Add frame metadata to list
                frame_data = frame.to_dict(save_screen_separately=True)
                frame_data['screen_file'] = screen_filename
                frames_data.append(frame_data)
                
                # Periodically save frame data
                if len(frames_data) % 100 == 0:
                    self._save_frames_data(frames_data, session_dir)
                    
            except:
                # Timeout or other error - continue
                continue
                
        # Save remaining frame data
        if frames_data:
            self._save_frames_data(frames_data, session_dir)
            
    def _save_frames_data(self, frames_data: List[Dict], session_dir: Path):
        """Save frame metadata to JSON file."""
        frames_file = session_dir / "frames_data.json"
        
        # Load existing data if file exists
        existing_data = []
        if frames_file.exists():
            try:
                with open(frames_file, 'r') as f:
                    existing_data = json.load(f)
            except:
                pass
                
        # Append new data
        existing_data.extend(frames_data)
        
        # Save updated data
        with open(frames_file, 'w') as f:
            json.dump(existing_data, f, indent=2)
            
        frames_data.clear()
        
    def get_session_stats(self, session_name: str) -> Dict[str, Any]:
        """Get statistics for a session."""
        session_dir = self.output_dir / session_name
        
        # Load session metadata
        metadata_file = session_dir / "session_metadata.json"
        if not metadata_file.exists():
            return {}
            
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            
        # Load frame data
        frames_file = session_dir / "frames_data.json"
        frame_count = 0
        action_counts = {}
        
        if frames_file.exists():
            with open(frames_file, 'r') as f:
                frames_data = json.load(f)
                frame_count = len(frames_data)
                
                # Count actions
                for frame in frames_data:
                    for action in frame.get('actions', []):
                        action_counts[action] = action_counts.get(action, 0) + 1
                        
        return {
            'metadata': metadata,
            'frame_count': frame_count,
            'action_counts': action_counts,
            'duration': metadata.get('end_time', 0) - metadata.get('start_time', 0)
        }
        
    def list_sessions(self) -> List[str]:
        """List all available sessions."""
        if not self.output_dir.exists():
            return []
            
        sessions = []
        for item in self.output_dir.iterdir():
            if item.is_dir() and (item / "session_metadata.json").exists():
                sessions.append(item.name)
                
        return sorted(sessions)


def demo_data_collection():
    """Demonstrate data collection functionality."""
    print("Starting data collection demo...")
    print("This will collect screen captures and input data for 10 seconds.")
    print("Please interact with your screen during this time.")
    
    collector = DataCollector(
        output_dir="demo_training_data",
        screen_size=(224, 224),
        capture_fps=5.0  # Low FPS for demo
    )
    
    # Start session
    session_id = collector.start_session("demo_session")
    
    try:
        # Collect data for 10 seconds
        collector.start_collection()
        
        for i in range(10):
            print(f"Collecting... {i+1}/10 seconds")
            time.sleep(1)
            
        collector.stop_collection()
        collector.stop_session()
        
        # Show results
        stats = collector.get_session_stats(session_id)
        print(f"\nCollection complete!")
        print(f"Frames collected: {stats['frame_count']}")
        print(f"Duration: {stats['duration']:.1f} seconds")
        print(f"Actions detected: {list(stats['action_counts'].keys())}")
        
        # List all sessions
        sessions = collector.list_sessions()
        print(f"Available sessions: {sessions}")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        collector.stop_session()
    except Exception as e:
        print(f"Error during collection: {e}")
        collector.stop_session()


if __name__ == "__main__":
    demo_data_collection()