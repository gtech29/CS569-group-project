import pandas as pd
import os
from sklearn.model_selection import train_test_split

# 1. Setup the path to find your sample data
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAMPLE_PATH = os.path.join(ROOT, "data", "processed", "steam_reviews_sample_10k.csv")


def main():
    print("Loading the 10k sample dataset...")
    df = pd.read_csv(SAMPLE_PATH)

    # 2. First Split: 80% for Training, 20% Temporary Leftovers
    # test_size=0.2 means 20% goes to the temp pile, leaving 80% for training
    # random_state=42 ensures we get the same random shuffle every time we run it
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)

    # 3. Second Split: Cut the 20% Temporary pile exactly in half
    # test_size=0.5 means split the temp pile 50/50
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    # 4. Verify our math!
    print("--- Data Split Complete ---")
    print(f"Total Original Rows: {len(df)}")
    print(f"Training Set (80%): {len(train_df)} rows")
    print(f"Validation Set (10%): {len(val_df)} rows")
    print(f"Test Set (10%): {len(test_df)} rows")


if __name__ == "__main__":
    main()
