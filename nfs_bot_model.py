"""
Neural network model for Need for Speed: Carbon bot.

This module implements a CNN-based model that learns to map screen captures
to game actions for autonomous gameplay.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import json
from pathlib import Path


class NFSBotCNN(nn.Module):
    """CNN model for predicting game actions from screen captures."""
    
    def __init__(self, 
                 input_size: Tuple[int, int] = (224, 224),
                 num_actions: int = 11,
                 hidden_dim: int = 512):
        """
        Initialize the CNN model.
        
        Args:
            input_size: Input image size (height, width)
            num_actions: Number of possible game actions
            hidden_dim: Hidden layer dimension
        """
        super(NFSBotCNN, self).__init__()
        
        self.input_size = input_size
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim
        
        # Convolutional layers for feature extraction
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
        )
        
        # Calculate the size after conv layers
        self.conv_output_size = self._get_conv_output_size()
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self.conv_output_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim // 2, num_actions)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _get_conv_output_size(self) -> int:
        """Calculate the output size of convolutional layers."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, *self.input_size)
            dummy_output = self.conv_layers(dummy_input)
            return dummy_output.view(1, -1).size(1)
            
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Output tensor of shape (batch_size, num_actions)
        """
        # Extract features with conv layers
        features = self.conv_layers(x)
        
        # Flatten for FC layers
        features = features.view(features.size(0), -1)
        
        # Apply FC layers
        output = self.fc_layers(features)
        
        return output
        
    def predict_actions(self, screen: np.ndarray, threshold: float = 0.5) -> List[str]:
        """
        Predict actions from a screen capture.
        
        Args:
            screen: Screen capture as numpy array (H, W, 3)
            threshold: Threshold for binary action prediction
            
        Returns:
            List of predicted action names
        """
        # Preprocess input
        if len(screen.shape) == 3:
            screen = screen.transpose(2, 0, 1)  # HWC to CHW
        
        # Normalize to [0, 1]
        screen = screen.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        screen_tensor = torch.from_numpy(screen).unsqueeze(0)
        
        # Get model prediction
        self.eval()
        with torch.no_grad():
            logits = self.forward(screen_tensor)
            probabilities = torch.sigmoid(logits)
            
        # Convert to action names
        action_names = self._get_action_names()
        predicted_actions = []
        
        for i, prob in enumerate(probabilities[0]):
            if prob > threshold:
                predicted_actions.append(action_names[i])
                
        return predicted_actions
        
    def _get_action_names(self) -> List[str]:
        """Get list of action names corresponding to output indices."""
        return [
            'accelerate',
            'brake', 
            'steer_left',
            'steer_right',
            'handbrake',
            'boost',
            'drift',
            'reset',
            'pause',
            'map',
            'camera_change'
        ]


class NFSBotTrainer:
    """Trainer for the NFS bot model."""
    
    def __init__(self, 
                 model: NFSBotCNN,
                 learning_rate: float = 1e-3,
                 device: str = 'auto'):
        """
        Initialize trainer.
        
        Args:
            model: The model to train
            learning_rate: Learning rate for optimizer
            device: Device to use ('cpu', 'cuda', or 'auto')
        """
        self.model = model
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.model.to(self.device)
        
        # Setup optimizer and loss function
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.BCEWithLogitsLoss()  # For multi-label classification
        
        # Training history
        self.training_history = {
            'loss': [],
            'accuracy': [],
            'epoch': []
        }
        
    def prepare_data(self, session_dir: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare training data from a session directory.
        
        Args:
            session_dir: Path to session directory
            
        Returns:
            Tuple of (screen_data, action_labels)
        """
        session_path = Path(session_dir)
        
        # Load frame data
        frames_file = session_path / "frames_data.json"
        with open(frames_file, 'r') as f:
            frames_data = json.load(f)
            
        screens = []
        labels = []
        action_names = self.model._get_action_names()
        
        print(f"Loading {len(frames_data)} frames...")
        
        for frame_data in frames_data:
            # Load screen data
            screen_file = session_path / "screens" / frame_data['screen_file']
            screen = np.load(screen_file)
            
            # Normalize and transpose
            screen = screen.astype(np.float32) / 255.0
            if len(screen.shape) == 3:
                screen = screen.transpose(2, 0, 1)  # HWC to CHW
                
            screens.append(screen)
            
            # Create action label vector
            action_vector = np.zeros(len(action_names), dtype=np.float32)
            for action in frame_data['actions']:
                if action in action_names:
                    idx = action_names.index(action)
                    action_vector[idx] = 1.0
                    
            labels.append(action_vector)
            
        # Convert to tensors
        screen_tensor = torch.from_numpy(np.array(screens))
        label_tensor = torch.from_numpy(np.array(labels))
        
        return screen_tensor, label_tensor
        
    def train_epoch(self, 
                   screen_data: torch.Tensor, 
                   labels: torch.Tensor,
                   batch_size: int = 32) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            screen_data: Screen capture data
            labels: Action labels
            batch_size: Batch size for training
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        # Create data indices and shuffle
        indices = torch.randperm(len(screen_data))
        
        for i in range(0, len(screen_data), batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_screens = screen_data[batch_indices].to(self.device)
            batch_labels = labels[batch_indices].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_screens)
            loss = self.criterion(outputs, batch_labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Calculate accuracy (for multi-label)
            predictions = torch.sigmoid(outputs) > 0.5
            accuracy = (predictions == batch_labels).float().mean()
            
            total_loss += loss.item()
            total_accuracy += accuracy.item()
            num_batches += 1
            
        return {
            'loss': total_loss / num_batches,
            'accuracy': total_accuracy / num_batches
        }
        
    def train(self, 
              session_dirs: List[str],
              num_epochs: int = 10,
              batch_size: int = 32,
              validation_split: float = 0.2) -> Dict[str, List[float]]:
        """
        Train the model on multiple sessions.
        
        Args:
            session_dirs: List of session directory paths
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
            
        Returns:
            Training history dictionary
        """
        print(f"Training on {len(session_dirs)} sessions for {num_epochs} epochs")
        
        # Load and combine data from all sessions
        all_screens = []
        all_labels = []
        
        for session_dir in session_dirs:
            screens, labels = self.prepare_data(session_dir)
            all_screens.append(screens)
            all_labels.append(labels)
            
        # Concatenate all data
        screen_data = torch.cat(all_screens, dim=0)
        label_data = torch.cat(all_labels, dim=0)
        
        print(f"Total training samples: {len(screen_data)}")
        
        # Split into train/validation
        num_samples = len(screen_data)
        num_val = int(num_samples * validation_split)
        num_train = num_samples - num_val
        
        indices = torch.randperm(num_samples)
        train_indices = indices[:num_train]
        val_indices = indices[num_train:]
        
        train_screens = screen_data[train_indices]
        train_labels = label_data[train_indices]
        val_screens = screen_data[val_indices]
        val_labels = label_data[val_indices]
        
        # Training loop
        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(train_screens, train_labels, batch_size)
            
            # Validate
            val_metrics = self.validate(val_screens, val_labels, batch_size)
            
            # Record history
            self.training_history['epoch'].append(epoch)
            self.training_history['loss'].append(train_metrics['loss'])
            self.training_history['accuracy'].append(train_metrics['accuracy'])
            
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"Train Loss: {train_metrics['loss']:.4f}, "
                  f"Train Acc: {train_metrics['accuracy']:.4f}, "
                  f"Val Loss: {val_metrics['loss']:.4f}, "
                  f"Val Acc: {val_metrics['accuracy']:.4f}")
                  
        return self.training_history
        
    def validate(self, 
                screen_data: torch.Tensor, 
                labels: torch.Tensor,
                batch_size: int = 32) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            screen_data: Validation screen data
            labels: Validation labels
            batch_size: Batch size for validation
            
        Returns:
            Validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for i in range(0, len(screen_data), batch_size):
                batch_screens = screen_data[i:i + batch_size].to(self.device)
                batch_labels = labels[i:i + batch_size].to(self.device)
                
                outputs = self.model(batch_screens)
                loss = self.criterion(outputs, batch_labels)
                
                predictions = torch.sigmoid(outputs) > 0.5
                accuracy = (predictions == batch_labels).float().mean()
                
                total_loss += loss.item()
                total_accuracy += accuracy.item()
                num_batches += 1
                
        return {
            'loss': total_loss / num_batches,
            'accuracy': total_accuracy / num_batches
        }
        
    def save_model(self, filepath: str):
        """Save the trained model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
            'model_config': {
                'input_size': self.model.input_size,
                'num_actions': self.model.num_actions,
                'hidden_dim': self.model.hidden_dim
            }
        }, filepath)
        
    def load_model(self, filepath: str):
        """Load a trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint.get('training_history', {})


def demo_model():
    """Demonstrate the model functionality."""
    print("Creating NFS Bot CNN model...")
    
    model = NFSBotCNN(input_size=(224, 224), num_actions=11)
    trainer = NFSBotTrainer(model)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Using device: {trainer.device}")
    
    # Test with dummy data
    print("\nTesting model with dummy input...")
    dummy_screen = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Test prediction
    predicted_actions = model.predict_actions(dummy_screen)
    print(f"Predicted actions: {predicted_actions}")
    
    print("\nModel demo complete!")


if __name__ == "__main__":
    demo_model()