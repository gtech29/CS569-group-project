import pandas as pd
import os

# Build the path to the sample file
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAMPLE_PATH = os.path.join(ROOT, "data", "processed", "steam_reviews_sample_10k.csv")

# Read the CSV File
df = pd.read_csv(SAMPLE_PATH)

