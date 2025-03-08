import torch

import sys
import os

# Get project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)  # Add to Python path

from src.dataloaders import create_dataloaders
from models.cnn import OzNet  # Import your CNN class
from src.train import train  
from src.train import val      


# Dataset paths
train_dir = r".\dataset\train"  
val_dir = r".\dataset\val"      
test_dir = r".\dataset\val"            

# Output directories
log_dir = "outputs/logs"        # TensorBoard logs
batch_size = 32
num_epochs = 15
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create directories if they don't exist
os.makedirs(log_dir, exist_ok=True)

# Load dataset
train_loader, val_loader, test_loader, class_names = create_dataloaders(
    train_dir=train_dir,
    val_dir=val_dir,
    test_dir=test_dir,
    batch_size=batch_size
)

# Initialize OzNet CNN Model
print("Training OzNet CNN for Stroke Detection...")
num_classes = len(class_names)  # Automatically get class count
model = OzNet(num_classes=num_classes).to(device)

# Define Optimizer & Loss Function
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9) 
criterion = torch.nn.CrossEntropyLoss()

# Train the model
train(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    loss_fn=criterion,
    n_epochs=num_epochs,
    device=device
)


