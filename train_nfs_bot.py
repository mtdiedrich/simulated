"""
Main training script for Need for Speed: Carbon bot.

This script provides a complete workflow for:
1. Collecting training data while playing
2. Training the neural network on collected data
3. Running the trained bot
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import List, Optional

import torch
import numpy as np

from data_collector import DataCollector
from nfs_bot_model import NFSBotCNN, NFSBotTrainer
from screen_capture import ScreenCapture
from input_capture import GameInputMapper


class NFSBotTrainingPipeline:
    """Complete training pipeline for NFS bot."""
    
    def __init__(self, 
                 data_dir: str = "nfs_training_data",
                 model_dir: str = "nfs_models",
                 screen_size: tuple = (224, 224),
                 capture_fps: float = 10.0):
        """
        Initialize the training pipeline.
        
        Args:
            data_dir: Directory for training data
            model_dir: Directory for saved models
            screen_size: Target screen capture size
            capture_fps: Data collection frame rate
        """
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.screen_size = screen_size
        self.capture_fps = capture_fps
        
        # Create directories
        self.data_dir.mkdir(exist_ok=True)
        self.model_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.data_collector = DataCollector(
            output_dir=str(self.data_dir),
            screen_size=screen_size,
            capture_fps=capture_fps
        )
        
        self.model = None
        self.trainer = None
        
    def collect_training_data(self, 
                            session_name: Optional[str] = None,
                            duration_minutes: float = 5.0,
                            capture_region: Optional[tuple] = None) -> str:
        """
        Collect training data by recording gameplay.
        
        Args:
            session_name: Name for the session
            duration_minutes: How long to collect data
            capture_region: Screen region to capture (x, y, width, height)
            
        Returns:
            Session ID
        """
        print("=" * 60)
        print("TRAINING DATA COLLECTION")
        print("=" * 60)
        
        # Update capture region if specified
        if capture_region:
            self.data_collector.capture_region = capture_region
            self.data_collector.screen_capture.capture_region = capture_region
            
        # Start session
        session_id = self.data_collector.start_session(session_name)
        
        print(f"Session: {session_id}")
        print(f"Duration: {duration_minutes} minutes")
        print(f"Screen size: {self.screen_size}")
        print(f"Capture FPS: {self.capture_fps}")
        
        if capture_region:
            print(f"Capture region: {capture_region}")
        else:
            print("Capture region: Full screen")
            
        print("\nInstructions:")
        print("1. Start Need for Speed: Carbon")
        print("2. Begin racing/driving")
        print("3. Play normally - your inputs will be recorded")
        print("4. The system will automatically stop after the specified duration")
        
        input("\nPress Enter when ready to start data collection...")
        
        try:
            # Start data collection
            self.data_collector.start_collection()
            
            # Collect for specified duration
            duration_seconds = duration_minutes * 60
            start_time = time.time()
            
            while time.time() - start_time < duration_seconds:
                remaining = duration_seconds - (time.time() - start_time)
                minutes = int(remaining // 60)
                seconds = int(remaining % 60)
                
                print(f"\rCollecting data... {minutes:02d}:{seconds:02d} remaining", end="")
                time.sleep(1)
                
            print("\n\nData collection complete!")
            
        except KeyboardInterrupt:
            print("\n\nData collection interrupted by user")
            
        finally:
            # Stop collection and session
            self.data_collector.stop_session()
            
        # Show session statistics
        stats = self.data_collector.get_session_stats(session_id)
        print(f"\nSession Statistics:")
        print(f"  Frames collected: {stats.get('frame_count', 0)}")
        print(f"  Duration: {stats.get('duration', 0):.1f} seconds")
        print(f"  Actions detected: {list(stats.get('action_counts', {}).keys())}")
        
        return session_id
        
    def train_model(self, 
                   session_ids: Optional[List[str]] = None,
                   epochs: int = 20,
                   batch_size: int = 32,
                   learning_rate: float = 1e-3) -> str:
        """
        Train the neural network model.
        
        Args:
            session_ids: List of session IDs to train on (None for all)
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            
        Returns:
            Path to saved model
        """
        print("=" * 60)
        print("MODEL TRAINING")
        print("=" * 60)
        
        # Get available sessions
        available_sessions = self.data_collector.list_sessions()
        
        if not available_sessions:
            raise ValueError("No training sessions found. Collect data first.")
            
        # Use specified sessions or all available
        if session_ids is None:
            session_ids = available_sessions
        else:
            # Validate session IDs
            invalid_sessions = [s for s in session_ids if s not in available_sessions]
            if invalid_sessions:
                raise ValueError(f"Invalid session IDs: {invalid_sessions}")
                
        print(f"Training on sessions: {session_ids}")
        print(f"Training epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {learning_rate}")
        
        # Create model and trainer
        self.model = NFSBotCNN(input_size=self.screen_size, num_actions=11)
        self.trainer = NFSBotTrainer(self.model, learning_rate=learning_rate)
        
        print(f"Model created with {sum(p.numel() for p in self.model.parameters())} parameters")
        print(f"Using device: {self.trainer.device}")
        
        # Prepare session directories
        session_dirs = [str(self.data_dir / session_id) for session_id in session_ids]
        
        # Start training
        print("\nStarting training...")
        history = self.trainer.train(
            session_dirs=session_dirs,
            num_epochs=epochs,
            batch_size=batch_size
        )
        
        # Save trained model
        timestamp = int(time.time())
        model_filename = f"nfs_bot_model_{timestamp}.pth"
        model_path = self.model_dir / model_filename
        
        self.trainer.save_model(str(model_path))
        print(f"\nModel saved to: {model_path}")
        
        return str(model_path)
        
    def run_bot(self, 
                model_path: str,
                capture_region: Optional[tuple] = None):
        """
        Run the trained bot.
        
        Args:
            model_path: Path to trained model
            capture_region: Screen region to capture
        """
        print("=" * 60)
        print("RUNNING NFS BOT")
        print("=" * 60)
        
        # Load model
        self.model = NFSBotCNN(input_size=self.screen_size, num_actions=11)
        self.trainer = NFSBotTrainer(self.model)
        self.trainer.load_model(model_path)
        
        print(f"Model loaded from: {model_path}")
        print(f"Using device: {self.trainer.device}")
        
        # Setup screen capture
        screen_capture = ScreenCapture(
            capture_region=capture_region,
            target_size=self.screen_size,
            capture_fps=30.0  # Higher FPS for real-time play
        )
        
        print("\nInstructions:")
        print("1. Start Need for Speed: Carbon")
        print("2. Position the game window appropriately")
        print("3. The bot will start making predictions")
        print("4. Press Ctrl+C to stop")
        
        if capture_region:
            print(f"5. Make sure the game is visible in region: {capture_region}")
        else:
            print("5. Make sure the game is visible on screen")
            
        input("\nPress Enter when ready to start the bot...")
        
        try:
            print("\nBot starting... (Press Ctrl+C to stop)")
            
            while True:
                # Capture screen
                frame = screen_capture.capture_frame()
                
                if frame is not None:
                    # Get model prediction
                    predicted_actions = self.model.predict_actions(frame)
                    
                    if predicted_actions:
                        print(f"\rPredicted actions: {', '.join(predicted_actions):<40}", end="")
                    else:
                        print(f"\rNo actions predicted{' ' * 30}", end="")
                        
                    # Note: Actual input simulation would require additional libraries
                    # like pynput or pyautogui and is not implemented here for safety
                    
                time.sleep(0.033)  # ~30 FPS
                
        except KeyboardInterrupt:
            print("\n\nBot stopped by user")
            
    def show_sessions(self):
        """Show available training sessions."""
        sessions = self.data_collector.list_sessions()
        
        if not sessions:
            print("No training sessions found.")
            return
            
        print("Available training sessions:")
        print("-" * 40)
        
        for session_id in sessions:
            stats = self.data_collector.get_session_stats(session_id)
            print(f"Session: {session_id}")
            print(f"  Frames: {stats.get('frame_count', 0)}")
            print(f"  Duration: {stats.get('duration', 0):.1f}s")
            print(f"  Actions: {list(stats.get('action_counts', {}).keys())}")
            print()


def main():
    """Main command line interface."""
    parser = argparse.ArgumentParser(description="Train a bot to play Need for Speed: Carbon")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Collect command
    collect_parser = subparsers.add_parser('collect', help='Collect training data')
    collect_parser.add_argument('--session', type=str, help='Session name')
    collect_parser.add_argument('--duration', type=float, default=5.0, help='Duration in minutes')
    collect_parser.add_argument('--region', type=int, nargs=4, metavar=('X', 'Y', 'W', 'H'),
                               help='Capture region (x, y, width, height)')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--sessions', type=str, nargs='+', help='Session IDs to train on')
    train_parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    train_parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    train_parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run the trained bot')
    run_parser.add_argument('model', type=str, help='Path to trained model')
    run_parser.add_argument('--region', type=int, nargs=4, metavar=('X', 'Y', 'W', 'H'),
                           help='Capture region (x, y, width, height)')
    
    # Sessions command
    subparsers.add_parser('sessions', help='Show available sessions')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
        
    # Initialize pipeline
    pipeline = NFSBotTrainingPipeline()
    
    try:
        if args.command == 'collect':
            region = tuple(args.region) if args.region else None
            pipeline.collect_training_data(
                session_name=args.session,
                duration_minutes=args.duration,
                capture_region=region
            )
            
        elif args.command == 'train':
            pipeline.train_model(
                session_ids=args.sessions,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr
            )
            
        elif args.command == 'run':
            region = tuple(args.region) if args.region else None
            pipeline.run_bot(args.model, capture_region=region)
            
        elif args.command == 'sessions':
            pipeline.show_sessions()
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()