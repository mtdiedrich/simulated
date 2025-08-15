"""
Installation and setup guide for the Need for Speed: Carbon bot.

This script helps users install dependencies and set up the system.
"""

import sys
import subprocess
import importlib
from typing import List, Tuple


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True


def check_dependencies() -> List[Tuple[str, bool, str]]:
    """Check which dependencies are installed."""
    dependencies = [
        # Core dependencies (should already be installed)
        ("torch", "PyTorch for neural networks"),
        ("numpy", "Numerical computing"),
        ("matplotlib", "Plotting and visualization"),
        ("pygame", "Game development library"),
        
        # NFS bot specific dependencies
        ("PIL", "Image processing (from Pillow)"),
        ("mss", "Fast screen capture"),
        ("pynput", "Input monitoring and simulation"),
        ("pyautogui", "Alternative input simulation"),
        ("scipy", "Scientific computing"),
    ]
    
    results = []
    for dep, description in dependencies:
        try:
            importlib.import_module(dep)
            results.append((dep, True, description))
        except ImportError:
            results.append((dep, False, description))
    
    return results


def install_missing_dependencies():
    """Install missing dependencies."""
    print("\n🔧 Installing missing dependencies...")
    
    # Required packages for NFS bot
    packages = [
        "mss>=6.1.0",
        "pillow>=8.0.0", 
        "pynput>=1.7.0",
        "pyautogui>=0.9.50",
        "scipy>=1.7.0"
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            print(f"   You may need to install this manually: pip install {package}")


def setup_directories():
    """Create necessary directories."""
    import os
    
    directories = [
        "nfs_training_data",
        "nfs_models",
        "test_captures"
    ]
    
    print("\n📁 Setting up directories...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created/verified directory: {directory}")


def run_basic_tests():
    """Run basic functionality tests."""
    print("\n🧪 Running basic tests...")
    
    # Test neural network
    try:
        from nfs_bot_model import NFSBotCNN
        model = NFSBotCNN()
        print("✅ Neural network model creation: PASSED")
    except Exception as e:
        print(f"❌ Neural network model creation: FAILED ({e})")
    
    # Test data structures
    try:
        from input_capture import InputEvent, GameInputMapper
        event = InputEvent(0.0, "test")
        mapper = GameInputMapper()
        print("✅ Input capture structures: PASSED")
    except Exception as e:
        print(f"❌ Input capture structures: FAILED ({e})")
    
    # Test main pipeline
    try:
        from train_nfs_bot import NFSBotTrainingPipeline
        pipeline = NFSBotTrainingPipeline()
        print("✅ Training pipeline creation: PASSED")
    except Exception as e:
        print(f"❌ Training pipeline creation: FAILED ({e})")


def show_usage_instructions():
    """Show basic usage instructions."""
    print("\n" + "="*60)
    print("🎮 NEED FOR SPEED: CARBON BOT - SETUP COMPLETE!")
    print("="*60)
    
    print("\n📖 BASIC USAGE:")
    print("1. Start Need for Speed: Carbon game")
    print("2. Collect training data:")
    print("   python train_nfs_bot.py collect --duration 10")
    print("3. Train the neural network:")
    print("   python train_nfs_bot.py train --epochs 20")
    print("4. Run the trained bot:")
    print("   python train_nfs_bot.py run path/to/model.pth")
    
    print("\n📖 INTERACTIVE MODE:")
    print("   python nfs_bot_example.py")
    
    print("\n📖 HELP:")
    print("   python train_nfs_bot.py --help")
    
    print("\n⚠️  IMPORTANT NOTES:")
    print("- The bot will simulate keyboard inputs when running")
    print("- Make sure NFS Carbon is the active window when using the bot")
    print("- Collect diverse training data for better performance")
    print("- Train for more epochs with more data for better results")
    print("- Emergency stop: Move mouse to screen corner or press Ctrl+C")
    
    print("\n🔧 CONFIGURATION:")
    print("- Adjust screen capture region if needed")
    print("- Modify key mappings in input_capture.py")
    print("- Tune neural network parameters in nfs_bot_model.py")
    print("- Change training parameters in train_nfs_bot.py")


def main():
    """Main setup routine."""
    print("🎮 Need for Speed: Carbon Bot - Setup & Installation")
    print("="*60)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Check dependencies
    print("\n📦 Checking dependencies...")
    deps = check_dependencies()
    
    missing_deps = [dep for dep, installed, _ in deps if not installed]
    installed_deps = [dep for dep, installed, _ in deps if installed]
    
    print(f"\n✅ Installed: {', '.join(installed_deps) if installed_deps else 'None'}")
    print(f"❌ Missing: {', '.join(missing_deps) if missing_deps else 'None'}")
    
    # Install missing dependencies
    if missing_deps:
        choice = input("\n🤔 Install missing dependencies? (y/n): ").lower()
        if choice == 'y':
            install_missing_dependencies()
        else:
            print("⚠️  Some features may not work without all dependencies")
    
    # Setup directories
    setup_directories()
    
    # Run tests
    run_basic_tests()
    
    # Show usage instructions
    show_usage_instructions()
    
    print("\n🎉 Setup complete! You're ready to train your NFS Carbon bot!")


if __name__ == "__main__":
    main()