import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier


# Load dataset
df = pd.read_csv("cleaned_jobs.csv")

# Fill missing values
df["text"] = df["text"].fillna("")

df["character_count"] = pd.to_numeric(
    df["character_count"], errors="coerce"
).fillna(0)

df["telecommuting"] = pd.to_numeric(
    df["telecommuting"], errors="coerce"
).fillna(0)

df["fraudulent"] = pd.to_numeric(
    df["fraudulent"], errors="coerce"
).fillna(0)


# Calculate location ratio
location_stats = (
    df.groupby("location")["fraudulent"]
    .agg(["sum", "count"])
)

location_stats["real"] = (
    location_stats["count"] - location_stats["sum"]
)

location_stats["ratio"] = (
    location_stats["sum"] /
    location_stats["real"].replace(0, np.nan)
)

location_stats["ratio"] = (
    location_stats["ratio"]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0.5)
)

df["ratio"] = df["location"].map(
    location_stats["ratio"]
).fillna(0.5)


# Text features
X_text = df["text"]
y = df["fraudulent"]


# Numerical features
X_num = df[
    ["telecommuting", "ratio", "character_count"]
]


# Train/test split
indices = np.arange(len(df))

train_idx, test_idx = train_test_split(
    indices,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train_text = X_text.iloc[train_idx]
X_test_text = X_text.iloc[test_idx]

X_train_num = X_num.iloc[train_idx]
X_test_num = X_num.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]


# Text vectorizer
count_vectorizer = CountVectorizer(
    stop_words="english"
)

count_train = count_vectorizer.fit_transform(
    X_train_text
)

count_test = count_vectorizer.transform(
    X_test_text
)


# Naive Bayes
nb_classifier = MultinomialNB()
nb_classifier.fit(
    count_train,
    y_train
)


# Text classifier
clf_log = SGDClassifier(
    loss="log_loss",
    random_state=42
)

clf_log.fit(
    count_train,
    y_train
)


# Numerical classifier
clf_num = SGDClassifier(
    loss="log_loss",
    random_state=42
)

clf_num.fit(
    X_train_num,
    y_train
)


# Save all model components
model_data = {
    "nb_classifier": nb_classifier,
    "clf_log": clf_log,
    "clf_num": clf_num,
    "count_vectorizer": count_vectorizer
}

with open("fake_job_model.pkl", "wb") as f:
    pickle.dump(model_data, f)


print("Model trained successfully!")
print("Model saved as fake_job_model.pkl")
print("Training samples:", len(train_idx))
print("Testing samples:", len(test_idx))
print("Numerical features:", list(X_num.columns))