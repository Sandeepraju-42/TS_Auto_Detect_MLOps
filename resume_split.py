import pandas as pd
from pathlib import Path
from dependancies.Features_Functions import time_based_split, FORECAST_HORIZON, VALIDATION_HORIZON

input_dir = Path(".").resolve() / "input"
print("reading full features csv...")
df = pd.read_csv(input_dir / "synthetic_fashion_demand_features.csv", parse_dates=["date"])
print("shape:", df.shape)
print("has abc_class/xyz_class:", "abc_class" in df.columns, "xyz_class" in df.columns)
print("date range:", df["date"].min(), df["date"].max())

train, val, test = time_based_split(df)
print("train/val/test shapes:", train.shape, val.shape, test.shape)

train.to_csv(input_dir / "train_features.csv", index=False)
val.to_csv(input_dir / "val_features.csv", index=False)
test.to_csv(input_dir / "test_features.csv", index=False)
print("done writing train/val/test")
