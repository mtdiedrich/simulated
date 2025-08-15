"""
Screen capture module for recording game footage.

This module provides functionality to capture screenshots at regular intervals
for training data collection.
"""

import numpy as np
import time
import threading
from typing import Optional, Callable, Tuple
import os

try:
    import pygame
    import pygame.gfxdraw
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

try:
    # Try to import mss for faster screen capture
    from mss import mss
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False

try:
    # Try to import PIL for image processing
    from PIL import Image, ImageGrab
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class ScreenCapture:
    """Handles screen capture for game recording."""
    
    def __init__(self, capture_region: Optional[Tuple[int, int, int, int]] = None,
                 target_size: Tuple[int, int] = (224, 224),
                 capture_fps: float = 30.0):
        """
        Initialize screen capture.
        
        Args:
            capture_region: (x, y, width, height) region to capture, None for full screen
            target_size: (width, height) to resize captured images to
            capture_fps: Target capture rate in frames per second
        """
        self.capture_region = capture_region
        self.target_size = target_size
        self.capture_fps = capture_fps
        self.capture_interval = 1.0 / capture_fps
        
        self.is_capturing = False
        self.capture_thread = None
        self.frame_callback = None
        self.latest_frame = None
        
        # Choose best available capture method
        self.capture_method = self._select_capture_method()
        
        if self.capture_method == 'mss':
            self.sct = mss()
        else:
            self.sct = None
            
    def _select_capture_method(self) -> str:
        """Select the best available screen capture method."""
        if MSS_AVAILABLE:
            return 'mss'  # Fastest option
        elif PIL_AVAILABLE:
            return 'pil'  # Good compatibility
        else:
            return 'pygame'  # Fallback
            
    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame.
        
        Returns:
            RGB image as numpy array of shape (height, width, 3) or None if failed
        """
        try:
            if self.capture_method == 'mss' and self.sct:
                if self.capture_region:
                    monitor = {
                        "top": self.capture_region[1],
                        "left": self.capture_region[0], 
                        "width": self.capture_region[2],
                        "height": self.capture_region[3]
                    }
                else:
                    monitor = self.sct.monitors[1]  # Primary monitor
                    
                screenshot = self.sct.grab(monitor)
                img = np.array(screenshot)
                img = img[:, :, :3]  # Remove alpha channel
                img = img[:, :, ::-1]  # BGR to RGB
                
            elif self.capture_method == 'pil':
                if self.capture_region:
                    img = ImageGrab.grab(bbox=self.capture_region)
                else:
                    img = ImageGrab.grab()
                img = np.array(img)
                if img.shape[2] == 4:  # Remove alpha if present
                    img = img[:, :, :3]
                    
            else:  # pygame fallback
                if not PYGAME_AVAILABLE:
                    return None
                    
                # Initialize pygame display if not done
                if not pygame.get_init():
                    pygame.init()
                    
                # This is a simplified fallback - in practice would need more setup
                return None
                
            # Resize image to target size
            if img is not None:
                img = self._resize_image(img)
                
            return img
            
        except Exception as e:
            print(f"Error capturing frame: {e}")
            return None
            
    def _resize_image(self, img: np.ndarray) -> np.ndarray:
        """Resize image to target size."""
        if PIL_AVAILABLE:
            pil_img = Image.fromarray(img)
            pil_img = pil_img.resize(self.target_size, Image.LANCZOS)
            return np.array(pil_img)
        else:
            # Simple nearest neighbor resize without PIL
            from scipy.ndimage import zoom
            h, w = img.shape[:2]
            target_h, target_w = self.target_size[1], self.target_size[0]
            zoom_h = target_h / h
            zoom_w = target_w / w
            return zoom(img, (zoom_h, zoom_w, 1), order=1).astype(np.uint8)
            
    def start_continuous_capture(self, frame_callback: Callable[[np.ndarray], None]):
        """
        Start continuous screen capture in a separate thread.
        
        Args:
            frame_callback: Function called with each captured frame
        """
        if self.is_capturing:
            return
            
        self.frame_callback = frame_callback
        self.is_capturing = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
    def stop_continuous_capture(self):
        """Stop continuous screen capture."""
        self.is_capturing = False
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
            
    def _capture_loop(self):
        """Main capture loop running in separate thread."""
        last_capture_time = 0
        
        while self.is_capturing:
            current_time = time.time()
            
            if current_time - last_capture_time >= self.capture_interval:
                frame = self.capture_frame()
                if frame is not None:
                    self.latest_frame = frame
                    if self.frame_callback:
                        self.frame_callback(frame)
                        
                last_capture_time = current_time
                
            # Small sleep to prevent excessive CPU usage
            time.sleep(0.001)
            
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get the most recently captured frame."""
        return self.latest_frame
        
    def save_frame(self, frame: np.ndarray, filepath: str):
        """Save a frame to disk."""
        if PIL_AVAILABLE:
            img = Image.fromarray(frame)
            img.save(filepath)
        else:
            # Fallback to numpy save
            np.save(filepath.replace('.png', '.npy'), frame)


def test_screen_capture():
    """Test the screen capture functionality."""
    print("Testing screen capture...")
    
    capture = ScreenCapture(target_size=(224, 224), capture_fps=1.0)
    print(f"Using capture method: {capture.capture_method}")
    
    # Test single frame capture
    frame = capture.capture_frame()
    if frame is not None:
        print(f"Successfully captured frame of size: {frame.shape}")
        
        # Save test frame
        os.makedirs('test_captures', exist_ok=True)
        capture.save_frame(frame, 'test_captures/test_frame.png')
        print("Saved test frame to test_captures/test_frame.png")
    else:
        print("Failed to capture frame")
        
    # Test continuous capture for a few seconds
    frames_captured = []
    
    def frame_callback(frame):
        frames_captured.append(frame)
        print(f"Captured frame {len(frames_captured)}")
        
    print("\nTesting continuous capture for 3 seconds...")
    capture.start_continuous_capture(frame_callback)
    time.sleep(3)
    capture.stop_continuous_capture()
    
    print(f"Captured {len(frames_captured)} frames in 3 seconds")


if __name__ == "__main__":
    test_screen_capture()