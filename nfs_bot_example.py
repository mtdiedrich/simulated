"""
Example usage of the Need for Speed: Carbon bot training system.

This script demonstrates the complete workflow from data collection
to training and running the bot.
"""

import os
import sys
import time
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_nfs_bot import NFSBotTrainingPipeline
from nfs_bot_controller import NFSBotController


def example_workflow():
    """Demonstrate the complete NFS bot workflow."""
    print("=" * 70)
    print("NEED FOR SPEED: CARBON BOT - COMPLETE WORKFLOW EXAMPLE")
    print("=" * 70)
    
    # Initialize the training pipeline
    pipeline = NFSBotTrainingPipeline(
        data_dir="example_nfs_data",
        model_dir="example_models", 
        screen_size=(224, 224),
        capture_fps=10.0
    )
    
    print("\n1. DATA COLLECTION PHASE")
    print("-" * 30)
    print("This phase records your gameplay to create training data.")
    print("The system will capture:")
    print("  - Screen frames at 10 FPS")
    print("  - Your keyboard inputs")
    print("  - Mouse movements and clicks")
    
    choice = input("\nDo you want to collect training data? (y/n): ").lower()
    
    if choice == 'y':
        # Example data collection
        print("\nStarting data collection...")
        print("Instructions:")
        print("1. Start Need for Speed: Carbon")
        print("2. Enter a race or free roam mode") 
        print("3. Drive normally - the system will record everything")
        
        # You can specify a capture region if needed:
        # capture_region = (100, 100, 800, 600)  # x, y, width, height
        capture_region = None  # Full screen
        
        try:
            session_id = pipeline.collect_training_data(
                session_name="example_session",
                duration_minutes=2.0,  # Short demo
                capture_region=capture_region
            )
            print(f"Data collection completed! Session ID: {session_id}")
            
        except Exception as e:
            print(f"Error during data collection: {e}")
            return
    else:
        print("Skipping data collection...")
        # Check if we have existing sessions
        sessions = pipeline.data_collector.list_sessions()
        if not sessions:
            print("No existing training data found. Cannot proceed without data.")
            return
        print(f"Using existing sessions: {sessions}")
        
    print("\n2. MODEL TRAINING PHASE")
    print("-" * 30)
    print("This phase trains a neural network on the collected data.")
    
    choice = input("Do you want to train a model? (y/n): ").lower()
    
    if choice == 'y':
        try:
            print("\nStarting model training...")
            print("This may take several minutes depending on:")
            print("  - Amount of training data")
            print("  - Hardware capabilities (CPU vs GPU)")
            print("  - Number of training epochs")
            
            model_path = pipeline.train_model(
                session_ids=None,  # Use all available sessions
                epochs=10,  # Reduced for demo
                batch_size=16,  # Smaller batch for demo
                learning_rate=1e-3
            )
            
            print(f"Model training completed! Model saved to: {model_path}")
            
        except Exception as e:
            print(f"Error during training: {e}")
            return
    else:
        print("Skipping model training...")
        # Look for existing models
        model_dir = Path("example_models")
        if model_dir.exists():
            models = list(model_dir.glob("*.pth"))
            if models:
                model_path = str(models[-1])  # Use latest model
                print(f"Using existing model: {model_path}")
            else:
                print("No trained models found. Cannot proceed without a model.")
                return
        else:
            print("No model directory found. Cannot proceed.")
            return
            
    print("\n3. BOT EXECUTION PHASE")
    print("-" * 30)
    print("This phase runs the trained bot to play the game.")
    print("WARNING: The bot will simulate keyboard inputs!")
    print("Make sure Need for Speed: Carbon is the active window.")
    
    choice = input("Do you want to run the bot? (y/n): ").lower()
    
    if choice == 'y':
        try:
            print("\nInitializing bot controller...")
            
            # Create bot controller
            bot = NFSBotController(
                model_path=model_path,
                screen_size=(224, 224),
                capture_region=None,  # Full screen
                prediction_fps=30.0,
                prediction_threshold=0.5
            )
            
            print("\nBot ready! Instructions:")
            print("1. Make sure NFS Carbon is running and active")
            print("2. Enter a race or free roam mode")
            print("3. The bot will start controlling the game")
            print("4. Press Ctrl+C to stop the bot")
            print("5. EMERGENCY: Move mouse to top-left corner to stop")
            
            input("Press Enter when ready to start the bot...")
            
            # Run bot for 30 seconds as demo
            bot.run_with_monitoring(duration_seconds=30.0)
            
        except Exception as e:
            print(f"Error running bot: {e}")
            
    print("\n" + "=" * 70)
    print("WORKFLOW COMPLETE!")
    print("=" * 70)
    print("\nWhat you accomplished:")
    print("✓ Learned how to collect training data from gameplay")
    print("✓ Understood the neural network training process")
    print("✓ Saw how the trained bot can control the game")
    print("\nNext steps:")
    print("- Collect more training data for better performance")
    print("- Experiment with different model architectures")
    print("- Fine-tune hyperparameters for your specific setup")
    print("- Add more sophisticated reward functions")


def test_components():
    """Test individual components of the system."""
    print("=" * 60)
    print("TESTING INDIVIDUAL COMPONENTS")
    print("=" * 60)
    
    print("\n1. Testing Screen Capture...")
    try:
        from screen_capture import test_screen_capture
        test_screen_capture()
        print("✓ Screen capture test passed")
    except Exception as e:
        print(f"✗ Screen capture test failed: {e}")
        
    print("\n2. Testing Input Capture...")
    try:
        from input_capture import test_input_capture
        # Note: This requires pynput and user interaction
        print("Note: Input capture test requires pynput and user interaction")
        print("✓ Input capture module loaded successfully")
    except Exception as e:
        print(f"✗ Input capture test failed: {e}")
        
    print("\n3. Testing Neural Network Model...")
    try:
        from nfs_bot_model import demo_model
        demo_model()
        print("✓ Neural network model test passed")
    except Exception as e:
        print(f"✗ Neural network model test failed: {e}")
        
    print("\n" + "=" * 60)
    print("COMPONENT TESTING COMPLETE")
    print("=" * 60)


def show_help():
    """Show help information."""
    print("=" * 60)
    print("NEED FOR SPEED: CARBON BOT - HELP")
    print("=" * 60)
    
    print("\nOVERVIEW:")
    print("This system trains an AI bot to play Need for Speed: Carbon by")
    print("learning from human gameplay. It consists of three main phases:")
    print("")
    print("1. DATA COLLECTION: Records your screen and inputs while playing")
    print("2. MODEL TRAINING: Trains a neural network on the recorded data")
    print("3. BOT EXECUTION: Uses the trained model to play autonomously")
    
    print("\nCOMMAND LINE USAGE:")
    print("python train_nfs_bot.py collect --duration 10    # Collect for 10 minutes")
    print("python train_nfs_bot.py train --epochs 50        # Train for 50 epochs")
    print("python train_nfs_bot.py run model.pth            # Run trained bot")
    print("python train_nfs_bot.py sessions                 # Show sessions")
    
    print("\nFILES AND DIRECTORIES:")
    print("├── train_nfs_bot.py          # Main training script")
    print("├── nfs_bot_controller.py     # Bot controller")
    print("├── nfs_bot_model.py          # Neural network model")
    print("├── data_collector.py         # Data collection system")
    print("├── screen_capture.py         # Screen capture utilities")
    print("├── input_capture.py          # Input monitoring utilities")
    print("├── nfs_training_data/        # Training data directory")
    print("└── nfs_models/               # Saved models directory")
    
    print("\nREQUIREMENTS:")
    print("- Python 3.7+")
    print("- PyTorch (for neural networks)")
    print("- PIL/Pillow (for image processing)")
    print("- MSS (for fast screen capture)")
    print("- pynput (for input monitoring/simulation)")
    print("- Need for Speed: Carbon game")
    
    print("\nTIPS:")
    print("- Collect diverse training data (different tracks, situations)")
    print("- Use a consistent screen resolution and window size")
    print("- Train for more epochs with more data for better performance")
    print("- Adjust prediction threshold based on model confidence")
    
    print("\nSAFETY:")
    print("- The bot simulates keyboard inputs - be careful!")
    print("- Always test in a safe environment first")
    print("- Keep emergency stop methods available")
    print("- Don't use for online competitive play")


def main():
    """Main entry point for examples."""
    print("Need for Speed: Carbon Bot - Example Script")
    print("")
    print("What would you like to do?")
    print("1. Run complete workflow example")
    print("2. Test individual components")
    print("3. Show help information")
    print("4. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            example_workflow()
            break
        elif choice == '2':
            test_components()
            break
        elif choice == '3':
            show_help()
            break
        elif choice == '4':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()