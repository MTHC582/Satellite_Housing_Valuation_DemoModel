import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class SatelliteDataset(Dataset):
    """
    Custom Dataset to handle paired Data:
    1. Numerical Features (from Excel)
    2. Satellite Image (from Folder)
    """

    def __init__(self, excel_file, image_dir, transform=None, is_test=False):
        # Load the Excel file
        self.df = pd.read_excel(excel_file)
        self.image_dir = image_dir
        self.transform = transform
        self.is_test = is_test

        # We explicitly choose the columns that matter for House Prices.
        # so we r goin to exclude: id, date, price, lat, long, zipcode
        self.feature_cols = [
            "bedrooms",
            "bathrooms",
            "sqft_living",
            "sqft_lot",
            "floors",
            "waterfront",
            "view",
            "condition",
            "grade",
            "sqft_above",
            "sqft_basement",
            "yr_built",
            "yr_renovated",
            "sqft_living15",
            "sqft_lot15",
        ]

        # Filling missing values with 0 to prevent crashes (obviously).
        for col in self.feature_cols:
            self.df[col] = self.df[col].fillna(0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Accessing the Row Data
        row = self.df.iloc[idx]
        prop_id = str(row["id"])

        # Loading the Image
        img_name = os.path.join(self.image_dir, f"{prop_id}.jpg")

        try:
            # We convert to RGB to ensure 3 color channels (Red, Green, Blue)
            image = Image.open(img_name).convert("RGB")
        except (FileNotFoundError, OSError):
            # In case of troubleshoot with that specific image or if it aint downloaded or even corrupted.
            # create a blank (black_) image so the code doesn't crash.
            image = Image.new("RGB", (224, 224), color="black")

        # Apply Resize/Normalization
        if self.transform:
            image = self.transform(image)

        # Get Numerical Features (The Excel numbers)
        features = torch.tensor(row[self.feature_cols].values.astype("float32"))

        # Get Target (Price) - Only if we are Training
        if not self.is_test:
            price = torch.tensor(row["price"], dtype=torch.float32)
            return image, features, price

        # If we are Testing, we don't have the answer key (Price)
        return image, features


# ==========================================
# TEST BLOCK (Runs only when you execute this file directly)
# ==========================================
if __name__ == "__main__":
    print("Testing Dataset Class...")

    # Define a simple resizer (224,224) (Standard for most AI-models)
    simple_transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )

    # Initializing the dataset.
    # Just making sure the filename matches your actual _Excel file name.
    ds = SatelliteDataset(
        excel_file="data/train(1).xlsx",
        image_dir="data/images",
        transform=simple_transform,
    )

    print(f"Dataset Initialized. Found {len(ds)} rows.")

    """
      Try to load the first item
           __Unit Test__
         ** SANITY CHECK **
    """
    print("Grabbing Item 0...")
    img, feats, price = ds[0]

    print(f"   Image Shape: {img.shape} (Should be 3, 224, 224)")
    print(f"   Features: {feats.shape} (Should be 15)")
    print(f"   Price: ${price.item():,.2f}")
    print("\n  SUCCESS: The Dataset code is working!")
