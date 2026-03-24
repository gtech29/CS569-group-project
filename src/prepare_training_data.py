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
    

    # 2. First Split: 80% for Training, 20% Temporary Leftovers
    # test_size=0.2 means 20% goes to the temp pile, leaving 80% for training
    # random_state=42 ensures we get the same random shuffle every time we run it
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)

    # 3. Second Split: Cut the 20% Temporary pile exactly in half
    # test_size=0.5 means split the temp pile 50/50
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    # 4. Verify 
    # print(f"Total Original Rows: {len(df)}")
    # print(f"Training Set (80%): {len(train_df)} rows")
    # print(f"Validation Set (10%): {len(val_df)} rows")
    # print(f"Test Set (10%): {len(test_df)} rows")
    
    train_df.to_csv(TRAIN_PATH, index=False)
    val_df.to_csv(VAL_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    # Stopping here for now might, need to sort the df by timestamp due to chronological splits, 
    # but will do more research on this
    
    
    
    
if __name__ == "__main__":
    main()
