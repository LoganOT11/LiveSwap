import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# Fix path to allow importing from parent folders
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.loader import AdDataset
from training.model_arch import AdDetectorCNN

# --- CONFIGURATION ---
from settings import PROCESSED_DATASET_DIR as DATASET_PATH
from settings import MODEL_PATH as MODEL_SAVE_PATH

BATCH_SIZE = 16
EPOCHS = 10 # How many times to loop through the entire dataset
LEARNING_RATE = 0.001

def train():
    # 1. Setup Device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # 2. Prepare Data
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset not found at {DATASET_PATH}")
        return

    full_dataset = AdDataset(DATASET_PATH)
    
    # Split: 80% for Training, 20% for Validation (Testing)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_data, val_data = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    # 3. Initialize Model
    model = AdDetectorCNN().to(device)
    criterion = nn.BCELoss() # Binary Cross Entropy (Standard for Yes/No problems)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for specs, labels in train_loader:
            specs, labels = specs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + Backward + Optimize
            outputs = model(specs)
            loss = criterion(outputs.squeeze(1), labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # 5. Validation Step (Check accuracy after every epoch)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for specs, labels in val_loader:
                specs, labels = specs.to(device), labels.to(device)
                outputs = model(specs)
                predicted = (outputs.squeeze(1) > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss/len(train_loader):.4f} | Accuracy: {accuracy:.2f}%")

    # 6. Save the Brain
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()
