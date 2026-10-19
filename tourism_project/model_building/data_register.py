"""Data registration: validate the dataset stored in the repo and print a summary."""
import os
import sys

import pandas as pd

DATA_PATH = os.getenv("DATA_PATH", "tourism_project/data/tourism.csv")

EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]

if not os.path.exists(DATA_PATH):
    sys.exit(f"Dataset not found at {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
if missing:
    sys.exit(f"Dataset is missing expected columns: {missing}")

if not set(df["ProdTaken"].dropna().unique()).issubset({0, 1}):
    sys.exit("Target column ProdTaken must only contain 0/1")

print(f"Dataset registered: {DATA_PATH}")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print(f"Missing values: {int(df.isna().sum().sum())}")
print(f"Duplicate CustomerIDs: {int(df['CustomerID'].duplicated().sum())}")
print("Target distribution:")
print(df["ProdTaken"].value_counts(normalize=True).round(3).to_string())
