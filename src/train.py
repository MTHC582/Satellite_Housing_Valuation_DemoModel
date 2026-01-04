"""
       One might notice a pause during the transition to the next epoch.
   It's no big deal, as i set up the loader (for user exp) only to the Train set
                      But not to the Validation set
Since the validation set is only (1/3)rd or (1/4)th, it takes as much as time as the
                 Trainin set does, in a propotionate way!
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm  # To implement a user view Progress bar.
import matplotlib.pyplot as plt

# One can import your custom modules
# (Since we are in the same folder, simple imports work)
from dataset import SatelliteDataset
from models import ValuationModel

# ================= CONFIGURATION =================
BATCH_SIZE = 16  # Processing 16 houses at a time is found to be a stable count.
LEARNING_RATE = 0.001  # Prolly fast the AI learns. (The alpha - coeff)
EPOCHS = 10  # Number of times to see/loop the whole dataset.
# While too many EPOCHS can most likely lead of easy overfit.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # :| GPU Over CPU.
# Python 313 is not recommended since cuda isnt avail for the latest version of py
# Hence one shall have 311 as it is a LTS Version.
# =================================================


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0

    # Create a progress bar
    loop = tqdm(loader, leave=True)

    for images, features, prices in loop:
        # Move data to GPU (if available)
        images = images.to(DEVICE)
        features = features.to(DEVICE)
        prices = prices.to(DEVICE).unsqueeze(1)  # Reshape [16] -> [16, 1]

        # 1_Forward Pass
        predictions = model(images, features)

        # 2_Calculate Error
        loss = criterion(predictions, prices)

        # 3_Backward Pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update Progress Bar
        running_loss += loss.item()
        loop.set_description(f"Loss: {loss.item():,.0f}")

    return running_loss / len(loader)


def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for images, features, prices in loader:
            images = images.to(DEVICE)
            features = features.to(DEVICE)
            prices = prices.to(DEVICE).unsqueeze(1)

            predictions = model(images, features)
            loss = criterion(predictions, prices)
            running_loss += loss.item()

    return running_loss / len(loader)


def main():
    print(f"Training on Device: {DEVICE}")

    # 1_Prepare Data
    print("Loading Dataset...")
    # Standard ResNet normalization
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Loading ONLY the Training Excel (not Test data prolly!)
    full_dataset = SatelliteDataset(
        "data/train(1).xlsx", "data/images", transform=transform
    )

    # Split: 80% for Training, 20% being Validation-Set (Checking our work)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(
        f"Data Loaded: {len(train_dataset)} Train samples | {len(val_dataset)} Val samples"
    )

    # 2_Setup Model
    model = ValuationModel().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3_Training Loop
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}

    print("\nStarting Training Loop...")

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss = validate(model, val_loader, criterion)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Train Loss: {train_loss:,.0f} | Val Loss: {val_loss:,.0f}")

        # Save Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Best Model Saved!")

    print("\n Training Complete. Best model is 'best_model.pth'")


if __name__ == "__main__":
    main()
