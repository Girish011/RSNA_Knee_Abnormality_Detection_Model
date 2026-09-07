from pathlib import Path
import pandas as pd

DOWNLOADS = Path("/Users/girish11/Downloads")

train = pd.read_csv(DOWNLOADS / "train.csv")
train_series = pd.read_csv(DOWNLOADS / "train_series.csv")

# .shape means (number of rows, number of columns)
print("train shape:", train.shape)
print("train_series shape:", train_series.shape)

print("train columns:", list(train.columns))
print("train_series columns:", list(train_series.columns))

