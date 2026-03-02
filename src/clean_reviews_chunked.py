import os
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_PATH = os.path.join(ROOT, "data", "raw", "all_reviews", "all_reviews.csv")
OUT_DIR = os.path.join(ROOT, "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

OUT_SAMPLE = os.path.join(OUT_DIR, "steam_reviews_sample_10k.csv")  # commit this
# Optional: full cleaned output (likely huge; keep out of GitHub)
OUT_CLEAN = os.path.join(OUT_DIR, "steam_reviews_cleaned.csv")

CHUNKSIZE = 100_000
SAMPLE_TARGET = 10_000

def normalize_cols(df):
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def pick_review_col(cols):
    for c in ["review", "review_text", "text", "content"]:
        if c in cols:
            return c
    for c in cols:
        if "review" in c or "text" in c or "body" in c:
            return c
    return None

def main():
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"Missing raw file: {RAW_PATH}")

    # Peek header/columns quickly
    peek = pd.read_csv(RAW_PATH, nrows=5, dtype=str, encoding_errors="ignore")
    peek = normalize_cols(peek)
    review_col = pick_review_col(peek.columns.tolist())
    if review_col is None:
        raise ValueError(f"Could not find review text column. Columns: {peek.columns.tolist()}")

    print("Using review text column:", review_col)

    # Remove old outputs
    for p in [OUT_SAMPLE, OUT_CLEAN]:
        if os.path.exists(p):
            os.remove(p)

    sample_parts = []
    sample_count = 0
    first_write = True
    total_written = 0

    reader = pd.read_csv(
        RAW_PATH,
        chunksize=CHUNKSIZE,
        dtype=str,                # <<< critical: no int parsing
        low_memory=True,
        encoding_errors="ignore",
        on_bad_lines="skip"       # <<< skips malformed rows safely
        # If you STILL get parsing issues, uncomment next line (slower but robust):
        # , engine="python"
    )

    for i, chunk in enumerate(reader):
        chunk = normalize_cols(chunk)

        if review_col not in chunk.columns:
            rc2 = pick_review_col(chunk.columns.tolist())
            if rc2 is None:
                print(f"Chunk {i+1}: no review column found; skipping.")
                continue
            review_col = rc2

        # basic cleaning
        chunk = chunk.dropna(subset=[review_col])
        chunk = chunk[chunk[review_col].str.strip() != ""]
        chunk = chunk.drop_duplicates(subset=[review_col])

        # drop likely PII columns if present
        pii_cols = [c for c in chunk.columns if any(x in c for x in ["username","steamid","author","email","profile","user_id"])]
        if pii_cols:
            chunk = chunk.drop(columns=pii_cols, errors="ignore")

        # normalize label if present
        if "recommended" in chunk.columns:
            v = chunk["recommended"].str.lower()
            chunk["recommended"] = v.replace({"true":"1","false":"0","yes":"1","no":"0"})
            chunk = chunk[chunk["recommended"].isin(["0","1"])]

        # write full cleaned (optional)
        chunk.to_csv(OUT_CLEAN, mode=("w" if first_write else "a"), index=False, header=first_write)
        first_write = False
        total_written += len(chunk)

        # build sample up to 10k
        if sample_count < SAMPLE_TARGET and len(chunk) > 0:
            needed = SAMPLE_TARGET - sample_count
            take = chunk.sample(n=min(needed, len(chunk)), random_state=42)
            sample_parts.append(take)
            sample_count += len(take)

        print(f"Chunk {i+1}: kept {len(chunk)} rows | total={total_written} | sample={sample_count}")

        # If you only care about the sample for GitHub, stop early:
        if sample_count >= SAMPLE_TARGET:
            print("Sample complete; stopping early to save time.")
            break

    # save sample
    if sample_parts:
        sample_df = pd.concat(sample_parts).drop_duplicates().head(SAMPLE_TARGET)
        sample_df.to_csv(OUT_SAMPLE, index=False)
        print("Saved sample:", OUT_SAMPLE, "rows=", len(sample_df))
    else:
        print("No sample collected.")

    print("Done.")

if __name__ == "__main__":
    main()