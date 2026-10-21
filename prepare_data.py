import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# 1. Load the dataset
data_path = Path(__file__).parent / "data" / "SMSSpamCollection"

df = pd.read_csv(
    data_path,
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8"
)

# 2. Remove exact duplicate rows
df = df.drop_duplicates().reset_index(drop=True)

# 3. Separate features and labels
X = df["message"]
y = df["label"]

# 4. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 5. Display the results
print("Total unique rows:", len(df))
print("Training messages:", len(X_train))
print("Testing messages:", len(X_test))

print("\nTraining label distribution:")
print(y_train.value_counts())

print("\nTesting label distribution:")
print(y_test.value_counts())


import re
from collections import Counter

# Tokenize a message into lowercase words
def tokenize(message):
    return re.findall(r"\b\w+\b", message.lower())


# Build word counts using training messages only
word_counts = Counter()

for message in X_train:
    tokens = tokenize(message)
    word_counts.update(tokens)


# Create a vocabulary from the training messages
vocabulary = set(word_counts.keys())


# Display a sample
sample_message = "Win a FREE prize now!"

print("\n--- Tokenization Example ---")
print("Original:", sample_message)
print("Tokens:", tokenize(sample_message))

print("\n--- Vocabulary Statistics ---")
print("Unique words:", len(vocabulary))

print("\n--- Top 20 Most Frequent Words ---")
print(word_counts.most_common(20))
