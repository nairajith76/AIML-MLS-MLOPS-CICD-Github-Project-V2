"""Data preparation: clean the dataset and create stratified train/test splits."""
import os

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = os.getenv("DATA_PATH", "tourism_project/data/tourism.csv")
OUT_DIR = os.getenv("SPLIT_DIR", ".")
TARGET = "ProdTaken"

df = pd.read_csv(DATA_PATH)

# Drop the index column and the identifier, which carry no predictive signal
df = df.drop(columns=[c for c in ["Unnamed: 0", "CustomerID"] if c in df.columns])

# Fix data-entry inconsistencies
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

df = df.drop_duplicates()
df = df.dropna(subset=[TARGET])

X = df.drop(columns=[TARGET])
y = df[TARGET].astype(int)

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

os.makedirs(OUT_DIR, exist_ok=True)
Xtrain.to_csv(os.path.join(OUT_DIR, "Xtrain.csv"), index=False)
Xtest.to_csv(os.path.join(OUT_DIR, "Xtest.csv"), index=False)
ytrain.to_csv(os.path.join(OUT_DIR, "ytrain.csv"), index=False)
ytest.to_csv(os.path.join(OUT_DIR, "ytest.csv"), index=False)

print(f"Train: {Xtrain.shape}, Test: {Xtest.shape}")
print(f"Positive rate - train: {ytrain.mean():.3f}, test: {ytest.mean():.3f}")
