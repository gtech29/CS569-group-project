import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# 1. Setup paths
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_PATH = os.path.join(ROOT, "data", "processed", "train.csv")
TEST_PATH = os.path.join(ROOT, "data", "processed", "test.csv")


def main():
    # 2. Load the split data
    print("Loading split datasets...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    print(f"I found these columns: {train_df.columns.tolist()}")

    # Use the 'review' column (or 'review_text' depending on your cleaning script)
    # Filling empty reviews with an empty string prevents errors
    train_df["review"] = train_df["review"].fillna("")
    test_df["review"] = test_df["review"].fillna("")

    # 3. Text to Numbers (TF-IDF)
    print("Converting text to numbers...")
    tfidf = TfidfVectorizer(max_features=5000)
    X_train = tfidf.fit_transform(train_df["review"])
    X_test = tfidf.transform(test_df["review"])

    y_train = train_df["voted_up"]
    y_test = test_df["voted_up"]

    # 4. Train the Baseline
    print("Training Logistic Regression model...")
    model = LogisticRegression(class_weight="balanced")
    model.fit(X_train, y_train)

    # 5. Evaluate results
    predictions = model.predict(X_test)
    print("\n--- Baseline Results ---")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")
    print("\nDetailed Report:")
    print(classification_report(y_test, predictions))
    
    # 6. Get the "Design Insights" (Most negative words)
    feature_names = tfidf.get_feature_names_out()
    coefficients = model.coef_[0]

    # Create a list of words and their "importance" scores
    word_scores = list(zip(feature_names, coefficients))

    # Sort them to see the most negative words
    word_scores.sort(key=lambda x: x[1])

    print("\n--- Top 'Negative' Design Keywords ---")
    for word, score in word_scores[:20]:  # Show top 20 negative words
        print(f"{word}: {score:.4f}")


if __name__ == "__main__":
    main()
