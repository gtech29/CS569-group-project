import pandas as pd
import os
from sklearn.model_selection import train_test_split

# 1. Setup the path for data
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAMPLE_PATH = os.path.join(ROOT, "data", "processed", "steam_reviews_sample_10k.csv")
OUT_DIR = os.path.join(ROOT, "data", "processed")
TRAIN_PATH = os.path.join(OUT_DIR, "train.csv")
VAL_PATH = os.path.join(OUT_DIR, "val.csv")
TEST_PATH = os.path.join(OUT_DIR, "test.csv")

def main():
    print("Loading the 10k sample dataset...")
    df = pd.read_csv(SAMPLE_PATH)

    # 1. Sort by time (Oldest to Newest)
    # This ensures we aren't "leaking" future reviews into the past
    df = df.sort_values("timestamp_created").reset_index(drop=True)

    # 2. Calculate the "cut points" for 80/10/10
    total_rows = len(df)
    train_end = int(total_rows * 0.8)
    val_end = int(total_rows * 0.9)

    # 3. Slice the data
    # .iloc[start:stop] takes a specific "slice" of the rows
    train_df = df.iloc[:train_end]  # First 80%
    val_df = df.iloc[train_end:val_end]  # Next 10%
    test_df = df.iloc[val_end:]  # Final 10% (The newest reviews)

    # 4. Save
    train_df.to_csv(TRAIN_PATH, index=False)
    val_df.to_csv(VAL_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    print(
        f"Chronological split complete: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test."
    )
    
    
if __name__ == "__main__":
    main()
