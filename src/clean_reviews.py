import os
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_PATH = os.path.join(ROOT, "data", "raw", "all_reviews", "all_reviews.csv")
OUT_DIR = os.path.join(ROOT, "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

OUT_FULL = os.path.join(OUT_DIR, "steam_reviews_cleaned.csv")      # not committed (too big)
OUT_SAMPLE = os.path.join(OUT_DIR, "steam_reviews_sample_10k.csv") # commit this

def main():
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"Could not find raw dataset at: {RAW_PATH}")

    # Read only needed columns if present (faster + cleaner)
    # If your dataset has different column names, we’ll adjust after you run once.
    usecols = None  # set to a list later if you want to restrict columns

    print("Loading reviews (this can take a while for large files)...")
    df = pd.read_csv(RAW_PATH, low_memory=True)

    # Normalize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Find the review text column
    review_col = None
    for c in df.columns:
        if c in ("review", "review_text", "text"):
            review_col = c
            break
    if review_col is None:
        # heuristic fallback
        for c in df.columns:
            if "review" in c or "text" in c or "body" in c:
                review_col = c
                break
    if review_col is None:
        raise ValueError(f"Could not find a review text column. Columns: {df.columns.tolist()}")

    print("Using review text column:", review_col)

    # Drop empty reviews
    df = df.dropna(subset=[review_col])
    df = df[df[review_col].astype(str).str.strip() != ""]

    # Drop duplicates
    df = df.drop_duplicates(subset=[review_col])

    # Remove likely PII columns if present
    pii_cols = [c for c in df.columns if any(x in c for x in ["user", "username", "steamid", "author", "email", "profile"])]
    if pii_cols:
        print("Dropping possible PII columns:", pii_cols)
        df = df.drop(columns=pii_cols, errors="ignore")

    # Ensure target label exists and is clean if present
    if "recommended" in df.columns:
        # Normalize common representations
        df["recommended"] = df["recommended"].astype(str).str.lower()
        df["recommended"] = df["recommended"].replace({"true": 1, "false": 0, "1": 1, "0": 0, "yes": 1, "no": 0})
        # Keep only valid rows
        df = df[df["recommended"].isin([0, 1])]

    print("Cleaned shape:", df.shape)

    # Save full cleaned (may be big)
    df.to_csv(OUT_FULL, index=False)
    print("Saved full cleaned dataset to:", OUT_FULL)

    # Save GitHub-safe sample
    sample_n = min(10000, len(df))
    sample = df.sample(n=sample_n, random_state=42)
    sample.to_csv(OUT_SAMPLE, index=False)
    print(f"Saved sample ({sample_n} rows) to:", OUT_SAMPLE)

if __name__ == "__main__":
    main()