"""
Game Design Insight Dashboard

This Streamlit app is a demo UI for our COMP 569 project.

Main purpose:
- Load real Steam review data from data/processed/test.csv
- Let the user filter and select reviews
- Detect sentiment and possible game design issue areas
- Generate developer-facing recommendations
- Show model result charts from our Training Results folder

Important note:
The design issue detection in this dashboard is currently rule-based.
That means it uses keyword matching for demo purposes.

The actual ML work from the repo is shown in the Model Results section,
where we display training/evaluation charts from our BERT/RoBERTa/etc. experiments.
"""

from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# Page setup
# ============================================================
# This controls the browser tab title, icon, and layout width.
st.set_page_config(
    page_title="Game Design Insight Dashboard",
    page_icon="🎮",
    layout="wide",
)

st.title("Game Design Insight Dashboard")

st.write(
    "A demo UI for turning Steam-style player reviews into actionable game design insights."
)


# ============================================================
# Project paths
# ============================================================
# APP_DIR points to the demo folder.
# ROOT_DIR points to the project root folder.
APP_DIR = Path(__file__).parent
ROOT_DIR = APP_DIR.parent

# Main dataset used by the dashboard.
# This comes from the repo's processed test split.
REAL_DATA_PATH = ROOT_DIR / "data" / "processed" / "test.csv"

# Backup dataset in case the real test dataset is missing.
FALLBACK_DATA_PATH = APP_DIR / "sample_reviews.csv"

# Folder where our model result images are stored.
TRAINING_RESULTS_DIR = ROOT_DIR / "data" / "processed" / "Training Results"


# ============================================================
# Rule-based design issue detection
# ============================================================
# These keywords are used to identify the most likely game design issue.
# This is intentionally simple so the dashboard is reliable during the presentation.
# Later, this could be replaced with a trained classifier.
ISSUE_KEYWORDS = {
    "Tutorial": [
        "tutorial",
        "onboarding",
        "explain",
        "explains",
        "confusing",
        "instructions",
        "teach",
        "crafting",
    ],
    "Inventory": [
        "inventory",
        "items",
        "sorting",
        "sort",
        "backpack",
        "storage",
        "clunky",
    ],
    "Performance": [
        "performance",
        "frame",
        "frames",
        "fps",
        "lag",
        "stutter",
        "runs poorly",
        "slow",
    ],
    "Difficulty": [
        "difficulty",
        "hard",
        "unfair",
        "spike",
        "boss",
        "enemy",
        "enemies",
    ],
    "UI": [
        "ui",
        "interface",
        "menu",
        "quest log",
        "hud",
        "button",
        "buttons",
    ],
    "Bugs": [
        "bug",
        "bugs",
        "crash",
        "crashed",
        "glitch",
        "broken",
        "save file",
    ],
    "Gameplay": [
        "gameplay",
        "loop",
        "repetitive",
        "boring",
        "grind",
        "tedious",
    ],
    "Dialogue": [
        "dialogue",
        "story",
        "choices",
        "characters",
        "narrative",
    ],
}


# Each detected issue maps to a developer-facing recommendation.
# This is what makes the dashboard feel like a game design assistant.
RECOMMENDATIONS = {
    "Tutorial": "Improve onboarding with clearer step-by-step guidance, tooltips, and a short guided tutorial for core mechanics.",
    "Inventory": "Reduce friction by adding sorting, filtering, item categories, and fewer clicks for common inventory actions.",
    "Performance": "Prioritize optimization passes, graphics settings, frame-rate testing, and performance benchmarks on lower-end devices.",
    "Difficulty": "Rebalance difficulty spikes by adjusting enemy stats, checkpoint placement, and early-game progression.",
    "UI": "Simplify menus, improve labels, make important screens easier to find, and reduce visual clutter.",
    "Bugs": "Focus on crash reports, save-file stability, bug reproduction steps, and regression testing.",
    "Gameplay": "Add more variety to the core loop through new objectives, pacing changes, or optional side activities.",
    "Dialogue": "Keep strengthening narrative choice, character writing, and dialogue variety since players are responding to it.",
    "Positive": "This review validates an existing design strength. Preserve this experience and use it as a benchmark for future updates.",
    "General Feedback": "Review the feedback manually and group it with similar comments to identify a clearer design pattern.",
}


# Simple word lists used only when analyzing custom text.
# For dataset reviews, we use the actual voted_up column when available.
POSITIVE_WORDS = [
    "love",
    "best",
    "fun",
    "polished",
    "relaxing",
    "interesting",
    "engaging",
    "great",
    "good",
    "excellent",
    "amazing",
    "recommend",
    "enjoy",
    "enjoyed",
    "beautiful",
    "classic",
]

NEGATIVE_WORDS = [
    "bad",
    "poorly",
    "confusing",
    "clunky",
    "terrible",
    "unfair",
    "boring",
    "repetitive",
    "bugs",
    "bug",
    "crash",
    "crashed",
    "hard",
    "slow",
    "lag",
    "barely",
    "hate",
    "broken",
]


# ============================================================
# Helper functions
# ============================================================
def analyze_review(review_text):
    """
    Analyze a single review.

    This function does three things:
    1. Detects the most likely design issue using keyword matching.
    2. Estimates sentiment using simple positive/negative word counts.
    3. Returns a recommendation based on the detected issue.

    This is used for:
    - Custom review input
    - Auto-labeling issue categories for dataset rows
    """
    text = str(review_text).lower()

    matched_issue = "General Feedback"
    matched_keywords = []

    # Find the first issue category with matching keywords.
    for issue, keywords in ISSUE_KEYWORDS.items():
        hits = [keyword for keyword in keywords if keyword in text]
        if hits:
            matched_issue = issue
            matched_keywords = hits
            break

    # Simple sentiment score for custom text.
    positive_score = sum(1 for word in POSITIVE_WORDS if word in text)
    negative_score = sum(1 for word in NEGATIVE_WORDS if word in text)

    if negative_score > positive_score:
        sentiment = "Negative"
    elif positive_score > negative_score:
        sentiment = "Positive"
    else:
        sentiment = "Mixed"

    # If a review is clearly positive and does not mention an issue,
    # label the design area as Positive instead of General Feedback.
    if sentiment == "Positive" and matched_issue == "General Feedback":
        matched_issue = "Positive"

    recommendation = RECOMMENDATIONS.get(
        matched_issue,
        RECOMMENDATIONS["General Feedback"],
    )

    return {
        "sentiment": sentiment,
        "issue": matched_issue,
        "keywords": matched_keywords,
        "recommendation": recommendation,
        "positive_score": positive_score,
        "negative_score": negative_score,
    }


def sentiment_from_voted_up(value):
    """
    Convert the dataset's voted_up value into a readable label.

    Steam review datasets usually store voted_up as:
    - 1 / True = Positive review
    - 0 / False = Negative review
    """
    value_text = str(value).strip().lower()

    if value_text in ["true", "1", "yes", "positive"]:
        return "Positive"

    if value_text in ["false", "0", "no", "negative"]:
        return "Negative"

    return "Mixed"


def find_review_column(df):
    """
    Find the column that contains review text.

    The real dataset uses 'review', but this makes the app flexible
    in case a teammate uses a slightly different CSV later.
    """
    possible_columns = ["review", "review_text", "text", "content"]

    for column in possible_columns:
        if column in df.columns:
            return column

    return None


def make_dropdown_label(row, review_column):
    """
    Create a readable label for each review in the sidebar dropdown.

    Instead of showing only raw review text, this label includes:
    - row number
    - game name
    - sentiment
    - detected issue
    - shortened review text
    """
    row_number = row.name

    game = str(row["game"]) if "game" in row.index else "Unknown Game"
    sentiment = (
        str(row["_sentiment_label"]) if "_sentiment_label" in row.index else "Unknown"
    )
    issue = str(row["_detected_issue"]) if "_detected_issue" in row.index else "General"

    review_text = str(row[review_column])
    short_review = review_text[:80] + "..." if len(review_text) > 80 else review_text

    return f"Row {row_number} | {game} | {sentiment} | {issue} | {short_review}"


def build_presentation_summary(sentiment, detected_issue, recommendation):
    """
    Build a polished explanation for the presentation.

    This section helps the presenter explain:
    - what the dashboard detected
    - why it matters to game developers
    - what action the team should take
    """
    if sentiment == "Negative":
        impact = (
            "This review points to a possible player frustration that could hurt retention, "
            "reviews, or overall user experience."
        )
    elif sentiment == "Positive":
        impact = (
            "This review highlights a design strength that the team should preserve and use "
            "as a benchmark for future updates."
        )
    else:
        impact = (
            "This review contains mixed or unclear feedback, so it should be grouped with "
            "similar reviews before making a design decision."
        )

    summary = {
        "Player Sentiment": sentiment,
        "Detected Design Area": detected_issue,
        "Why It Matters": impact,
        "Recommended Action": recommendation,
    }

    return summary


# ============================================================
# Load dataset
# ============================================================
# Prefer the real processed test dataset.
# Fall back to demo/sample_reviews.csv if test.csv is not available.
if REAL_DATA_PATH.exists():
    DATA_PATH = REAL_DATA_PATH
    DATA_SOURCE_LABEL = "Real test dataset"
else:
    DATA_PATH = FALLBACK_DATA_PATH
    DATA_SOURCE_LABEL = "Fallback demo dataset"

if not DATA_PATH.exists():
    st.error("Could not find test.csv or demo/sample_reviews.csv.")
    st.stop()

reviews_df = pd.read_csv(DATA_PATH)

# Normalize column names so code can use lowercase consistently.
reviews_df.columns = reviews_df.columns.str.strip().str.lower()

review_column = find_review_column(reviews_df)

if review_column is None:
    st.error("Could not find a review text column.")
    st.write("Expected one of: review, review_text, text, content")
    st.write("Columns found:", reviews_df.columns.tolist())
    st.stop()

# Remove rows with missing review text.
reviews_df = reviews_df.dropna(subset=[review_column]).copy()
reviews_df[review_column] = reviews_df[review_column].astype(str)

# Create readable sentiment labels.
# If voted_up exists, use the real label from the dataset.
# Otherwise, estimate sentiment from review text.
if "voted_up" in reviews_df.columns:
    reviews_df["_sentiment_label"] = reviews_df["voted_up"].apply(
        sentiment_from_voted_up
    )
else:
    reviews_df["_sentiment_label"] = reviews_df[review_column].apply(
        lambda text: analyze_review(text)["sentiment"]
    )

# Create issue labels using the rule-based analyzer.
# This adds a detected issue for every row so we can filter and chart it.
reviews_df["_detected_issue"] = reviews_df[review_column].apply(
    lambda text: analyze_review(text)["issue"]
)


# ============================================================
# Sidebar controls
# ============================================================
st.sidebar.header("Demo Controls")
st.sidebar.caption(f"Data source: {DATA_SOURCE_LABEL}")

review_mode = st.sidebar.radio(
    "Choose input type:",
    ["Dataset review", "Custom review"],
)

# Start with all reviews, then apply filters below.
filtered_df = reviews_df.copy()

if review_mode == "Dataset review":
    st.sidebar.divider()
    st.sidebar.subheader("Dataset Filters")

    # -----------------------------
    # Language filter
    # -----------------------------
    # Default to English because it makes the live presentation easier.
    if "language" in filtered_df.columns:
        language_options = ["All languages"] + sorted(
            filtered_df["language"].dropna().astype(str).unique().tolist()
        )

        default_language_index = 0
        lower_language_options = [language.lower() for language in language_options]

        if "english" in lower_language_options:
            default_language_index = lower_language_options.index("english")

        selected_language = st.sidebar.selectbox(
            "Language:",
            language_options,
            index=default_language_index,
        )

        if selected_language != "All languages":
            filtered_df = filtered_df[
                filtered_df["language"].astype(str) == selected_language
            ]

    # -----------------------------
    # Game filter
    # -----------------------------
    # Lets the presenter focus on one game or view all games.
    if "game" in filtered_df.columns:
        game_options = ["All games"] + sorted(
            filtered_df["game"].dropna().astype(str).unique().tolist()
        )

        selected_game = st.sidebar.selectbox(
            "Game:",
            game_options,
        )

        if selected_game != "All games":
            filtered_df = filtered_df[filtered_df["game"].astype(str) == selected_game]

    # -----------------------------
    # Sentiment filter
    # -----------------------------
    # Useful presentation flow:
    # choose Negative reviews to show pain points and recommendations.
    sentiment_options = ["All sentiments", "Positive", "Negative", "Mixed"]

    selected_sentiment_filter = st.sidebar.selectbox(
        "Sentiment:",
        sentiment_options,
    )

    if selected_sentiment_filter != "All sentiments":
        filtered_df = filtered_df[
            filtered_df["_sentiment_label"] == selected_sentiment_filter
        ]

    # -----------------------------
    # Detected issue filter
    # -----------------------------
    # Lets the presenter quickly find examples involving bugs, UI, performance, etc.
    issue_options = ["All issues"] + sorted(
        filtered_df["_detected_issue"].dropna().astype(str).unique().tolist()
    )

    selected_issue_filter = st.sidebar.selectbox(
        "Detected issue:",
        issue_options,
    )

    if selected_issue_filter != "All issues":
        filtered_df = filtered_df[
            filtered_df["_detected_issue"] == selected_issue_filter
        ]


# ============================================================
# Review selection
# ============================================================
selected_review = ""
selected_row = None

if review_mode == "Dataset review":
    st.sidebar.caption(f"Matching reviews: {len(filtered_df)}")

    if filtered_df.empty:
        st.warning("No reviews match the selected filters. Try changing the filters.")
        st.stop()

    # Limit the dropdown size so Streamlit stays responsive.
    MAX_DROPDOWN_REVIEWS = 300
    display_df = filtered_df.head(MAX_DROPDOWN_REVIEWS).copy()

    review_labels = [
        make_dropdown_label(row, review_column) for _, row in display_df.iterrows()
    ]

    selected_label = st.sidebar.selectbox(
        "Choose a review from the dataset:",
        review_labels,
    )

    selected_position = review_labels.index(selected_label)
    selected_row = display_df.iloc[selected_position]
    selected_review = selected_row[review_column]

else:
    # Custom review mode is useful for a live audience demo.
    selected_review = st.sidebar.text_area(
        "Paste a Steam-style review:",
        height=160,
        placeholder="Example: The combat is fun, but the tutorial is confusing...",
    )


# ============================================================
# Analyze selected review
# ============================================================
sentiment = None
detected_issue = None
recommendation = None
matched_keywords = []

if selected_review.strip():
    analysis = analyze_review(selected_review)

    detected_issue = analysis["issue"]
    recommendation = analysis["recommendation"]
    matched_keywords = analysis["keywords"]

    if review_mode == "Dataset review" and selected_row is not None:
        # For dataset reviews, use the dataset sentiment label.
        # This is better than guessing from keywords.
        sentiment = selected_row["_sentiment_label"]

        # Use the precomputed issue category for consistency with filters/charts.
        detected_issue = selected_row["_detected_issue"]

        recommendation = RECOMMENDATIONS.get(
            detected_issue,
            RECOMMENDATIONS["General Feedback"],
        )
    else:
        # For custom reviews, use the rule-based sentiment estimate.
        sentiment = analysis["sentiment"]


# ============================================================
# Main dashboard: selected review
# ============================================================
st.subheader("Selected Review")

if selected_review.strip():
    if selected_row is not None and "game" in selected_row.index:
        st.caption(f"Game: {selected_row['game']}")

    st.info(selected_review)

    # Three-column summary of the current review.
    col1, col2, col3 = st.columns(3)

    with col1:
        st.caption("Review Sentiment")
        st.markdown(f"### {sentiment}")

    with col2:
        st.caption("Detected Issue")
        st.markdown(f"### {detected_issue}")

    with col3:
        st.caption("Input Type")
        st.markdown(f"### {review_mode}")

    st.divider()

    # Developer recommendation card.
    st.subheader("Developer Recommendation")

    if sentiment == "Positive":
        st.success(recommendation)
    elif sentiment == "Negative":
        st.warning(recommendation)
    else:
        st.info(recommendation)

    # Polished explanation for presentation delivery.
    st.subheader("Presentation Summary")

    presentation_summary = build_presentation_summary(
        sentiment,
        detected_issue,
        recommendation,
    )

    summary_col1, summary_col2 = st.columns(2)

    with summary_col1:
        st.markdown("**What the AI detected:**")
        st.write(
            f"The selected review is classified as **{presentation_summary['Player Sentiment']}** "
            f"and is connected to **{presentation_summary['Detected Design Area']}**."
        )

        st.markdown("**Why it matters:**")
        st.write(presentation_summary["Why It Matters"])

    with summary_col2:
        st.markdown("**Recommended design action:**")
        st.write(presentation_summary["Recommended Action"])

        st.markdown("**Presenter talking point:**")
        st.info(
            "Instead of manually reading thousands of Steam reviews, this dashboard helps "
            "developers quickly identify player pain points and turn them into design actions."
        )

    # Show evidence keywords for custom reviews or rule-based matches.
    if matched_keywords:
        st.caption("Evidence keywords detected: " + ", ".join(matched_keywords))
    else:
        st.caption(
            "No specific issue keywords detected. Treat this as general feedback."
        )

else:
    st.warning("Enter or select a review to analyze.")


# ============================================================
# Dataset overview
# ============================================================
st.divider()

st.subheader("Dataset Overview")

overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)

with overview_col1:
    st.metric("Loaded Rows", len(reviews_df))

with overview_col2:
    filtered_count = (
        len(filtered_df) if review_mode == "Dataset review" else len(reviews_df)
    )
    st.metric("Filtered Rows", filtered_count)

with overview_col3:
    st.metric("Review Column", review_column)

with overview_col4:
    st.metric("Source", DATA_SOURCE_LABEL)

# Two quick charts to help explain the dataset.
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.write("Review sentiment distribution:")
    sentiment_counts = filtered_df["_sentiment_label"].value_counts()
    st.bar_chart(sentiment_counts)

with chart_col2:
    st.write("Detected design issue distribution:")
    issue_counts = filtered_df["_detected_issue"].value_counts()
    st.bar_chart(issue_counts)

# Show a small sample of the loaded data.
# This helps prove the dashboard is using the real dataset.
with st.expander("View loaded dataset sample"):
    columns_to_show = [
        column
        for column in [
            "game",
            "language",
            review_column,
            "voted_up",
            "_sentiment_label",
            "_detected_issue",
        ]
        if column in reviews_df.columns
    ]

    st.dataframe(
        filtered_df[columns_to_show].head(50),
        use_container_width=True,
    )


# ============================================================
# Model results
# ============================================================
# This section displays saved images from model training/evaluation.
# These images connect the UI demo back to the ML work in the repo.
st.divider()

st.subheader("Model Results")

st.write(
    "These charts summarize the model training and evaluation work behind the demo."
)

# List of model result images we want to display.
# If an image does not exist, it is automatically skipped.
model_result_images = {
    "BERT Model Comparison Summary": TRAINING_RESULTS_DIR
    / "BERT Model Comparison Summary.png",
    "ROC Comparison: Baseline vs Best Deep Learning Model": TRAINING_RESULTS_DIR
    / "ROC Comparison - Baseline vs BEST DL Model.png",
    "Final Test Set Confusion Matrix": TRAINING_RESULTS_DIR
    / "Final Test Set Evaluation (Best Model Only)"
    / "Confusion Matrix.png",
    "Final Test Set ROC Curve": TRAINING_RESULTS_DIR
    / "Final Test Set Evaluation (Best Model Only)"
    / "ROC Curve.png",
    "Final Test Set Precision Recall": TRAINING_RESULTS_DIR
    / "Final Test Set Evaluation (Best Model Only)"
    / "Precision Recall.png",
}

# Keep only charts that are actually present on disk.
available_images = {
    title: path for title, path in model_result_images.items() if path.exists()
}

if available_images:
    selected_chart = st.selectbox(
        "Choose a model result chart:",
        list(available_images.keys()),
    )

    st.image(
        str(available_images[selected_chart]),
        caption=selected_chart,
        use_container_width=True,
    )
else:
    st.warning(
        "No model result images were found. Check that the Training Results folder exists."
    )
