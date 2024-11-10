import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
import math

class PrivacyEngine:
    def __init__(
        self,
        model: nn.Module,
        batch_size: int,
        sample_size: int,
        noise_multiplier: float,
        max_grad_norm: float,
        target_delta: float = 1e-5
    ):
        self.model = model
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.target_delta = target_delta
        
        # Initialize privacy accounting
        self.steps = 0
        
    def get_privacy_spent(self):
        """
        Compute epsilon using moment accountant method
        """
        if self.steps == 0:
            return 0.0

        # Sampling rate
        q = self.batch_size / self.sample_size
        
        # Privacy per step
        eps_step = q * math.sqrt(2 * math.log(1.25/self.target_delta)) / self.noise_multiplier
        
        # Total privacy cost (composition)
        epsilon = eps_step * math.sqrt(self.steps)
        
        return epsilon

    def clip_and_noise_gradients(self) -> None:
        """
        Clip gradients and add Gaussian noise
        """
        # Clip gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.max_grad_norm
        )
        
        # Add noise
        with torch.no_grad():
            for param in self.model.parameters():
                if param.requires_grad and param.grad is not None:
                    noise = torch.normal(
                        mean=0,
                        std=self.noise_multiplier * self.max_grad_norm,
                        size=param.grad.shape,
                        device=param.grad.device
                    )
                    param.grad += noise / self.batch_size

        self.steps += 1

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.flatten = nn.Flatten()
        
        self.features = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10)
        )
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.features(x)
        return x

def load_mnist_data(batch_size=64):
    """
    Load and preprocess MNIST dataset
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=True, 
        transform=transform, 
        download=True
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=False, 
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, test_loader, len(train_dataset)

def train_private_model(
    model,
    train_loader,
    test_loader,
    privacy_engine,
    epochs=10,
    learning_rate=0.001
):
    """
    Train the model with differential privacy using Adam optimizer
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    test_accuracies = []
    epsilons = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Apply differential privacy mechanisms
            privacy_engine.clip_and_noise_gradients()
            
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 100 == 99:
                epsilon = privacy_engine.get_privacy_spent()
                print(f'Epoch: {epoch + 1}, Batch: {batch_idx + 1}, '
                      f'Loss: {running_loss / 100:.3f}, '
                      f'ε: {epsilon:.2f}')
                running_loss = 0.0
        
        # Evaluate on test set
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        test_accuracy = 100 * correct / total
        epsilon = privacy_engine.get_privacy_spent()
        
        print(f'Epoch: {epoch + 1}, '
              f'Test Accuracy: {test_accuracy:.2f}%, '
              f'ε: {epsilon:.2f}')
        
        train_losses.append(running_loss)
        test_accuracies.append(test_accuracy)
        epsilons.append(epsilon)
    
    return train_losses, test_accuracies, epsilons

def plot_training_results(train_losses, test_accuracies, epsilons):
    """
    Plot training loss, test accuracy, and privacy budget
    """
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Batch (x100)')
    plt.ylabel('Loss')
    
    plt.subplot(1, 3, 2)
    plt.plot(test_accuracies)
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    
    plt.subplot(1, 3, 3)
    plt.plot(epsilons)
    plt.title('Privacy Budget (ε)')
    plt.xlabel('Epoch')
    plt.ylabel('Epsilon')
    
    plt.tight_layout()
    plt.show()

def main():
    # Hyperparameters
    BATCH_SIZE = 64
    EPOCHS = 10
    LEARNING_RATE = 0.001
    NOISE_MULTIPLIER = 1.1  # Controls privacy guarantee
    MAX_GRAD_NORM = 1.0    # Gradient clipping threshold
    TARGET_DELTA = 1e-5    # Target delta for DP guarantee
    
    # Load data
    train_loader, test_loader, sample_size = load_mnist_data(BATCH_SIZE)
    
    # Initialize model
    model = MNISTNet()
    
    # Initialize privacy engine
    privacy_engine = PrivacyEngine(
        model=model,
        batch_size=BATCH_SIZE,
        sample_size=sample_size,
        noise_multiplier=NOISE_MULTIPLIER,
        max_grad_norm=MAX_GRAD_NORM,
        target_delta=TARGET_DELTA
    )
    
    # Train model with privacy
    train_losses, test_accuracies, epsilons = train_private_model(
        model,
        train_loader,
        test_loader,
        privacy_engine,
        EPOCHS,
        LEARNING_RATE
    )
    
    # Plot results
    plot_training_results(train_losses, test_accuracies, epsilons)

if __name__ == "__main__":
    main()