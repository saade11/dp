import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)

# Data loading and preprocessing
def load_mnist_data(batch_size=64):
    """
    Load and preprocess MNIST dataset
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
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
    
    return train_loader, test_loader

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

def train_model(model, train_loader, test_loader, epochs=10, learning_rate=0.01):
    """
    Train the model using SGD optimizer
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    
    train_losses = []
    test_accuracies = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 100 == 99:
                print(f'Epoch: {epoch + 1}, Batch: {batch_idx + 1}, Loss: {running_loss / 100:.3f}')
                running_loss = 0.0
        
        # Evaluate on test set
        model.eval()
        correct = 0
        total = 0
        test_loss = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        test_accuracy = 100 * correct / total
        avg_test_loss = test_loss / len(test_loader)
        
        print(f'Epoch: {epoch + 1}, Test Accuracy: {test_accuracy:.2f}%')
        
        train_losses.append(running_loss)
        test_accuracies.append(test_accuracy)
        
        scheduler.step(avg_test_loss)
    
    return train_losses, test_accuracies

def plot_training_results(train_losses, test_accuracies):
    """
    Plot training loss and test accuracy
    """
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Batch (x100)')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies)
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    
    plt.tight_layout()
    plt.show()

def main():
    # Hyperparameters
    BATCH_SIZE = 64
    EPOCHS = 10
    LEARNING_RATE = 0.01
    
    # Load data
    train_loader, test_loader = load_mnist_data(BATCH_SIZE)
    
    # Initialize model
    model = MNISTNet()
    
    # Train model
    train_losses, test_accuracies = train_model(
        model, 
        train_loader, 
        test_loader, 
        EPOCHS, 
        LEARNING_RATE
    )
    
    # Plot results
    plot_training_results(train_losses, test_accuracies)

if __name__ == "__main__":
    main()