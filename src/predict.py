import torch
import os
from dataset import SatelliteDataset
from models import ValuationModel  # Make sure this matches the class name in models.py
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

# 1_Setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading model on {DEVICE}...")

# 2_Define Transforms (The model expects 224x224 images)
val_transform = transforms.Compose(
    [transforms.Resize((224, 224)), transforms.ToTensor()]
)

# 3_Load the Full Dataset
# We use the same file, but we will pick random items from it to test
full_dataset = SatelliteDataset(
    excel_file="data/train(1).xlsx", image_dir="data/images", transform=val_transform
)

# 4_Split it (Optional: Just to mimic how we trained)
# Let's just grab 100 random items to pretend this is our "Validation Set"
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
_, val_dataset = random_split(full_dataset, [train_size, val_size])

# 5_Create the Loader
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

# 6_Load the Saved Brain
model = ValuationModel().to(DEVICE)

# Check if the file exists before loading
if os.path.exists("best_model.pth"):
    model.load_state_dict(
        torch.load("best_model.pth", map_location=DEVICE, weights_only=True)
    )
    print("Model weights loaded successfully!")
else:
    print("ERROR: 'best_model.pth' not found. Did you run train.py?")
    exit()

model.eval()  # Switch to 'Test Mode'

# 7_Make Predictions
print("\n---  PREDICTION TIME  ---")
print(f"{'ACTUAL PRICE':<15} | {'PREDICTED':<15} | {'ERROR':<15}")
print("-" * 50)

# Let's look at 10 random houses
num_samples = 10
with torch.no_grad():
    for i, (images, features, prices) in enumerate(val_loader):
        if i >= num_samples:
            break

        # Move to GPU
        images, features = images.to(DEVICE), features.to(DEVICE)

        # Predict
        prediction = model(images, features)

        # Convert back to simple numbers
        actual_price = prices.item()
        pred_price = prediction.item()
        error = abs(actual_price - pred_price)

        print(f"${actual_price:<14,.0f} | ${pred_price:<14,.0f} | ${error:,.0f}")

print("\nDone!")
