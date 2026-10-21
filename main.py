
import pandas as pd
from pathlib import Path

# Get the dataset path
data_path = Path(__file__).parent / "data" / "SMSSpamCollection"

# Load the dataset
df = pd.read_csv(
    data_path,
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8"
)

# Explore the data
print("\n--- First 5 Messages ---")
print(df.head())

print("\n--- Dataset Dimensions ---")
print(df.shape)

print("\n--- Label Distribution ---")
print(df["label"].value_counts())

print("\n--- Missing Values ---")
print(df.isnull().sum())


# Step 3A: Check for duplicate rows
print("\n--- Duplicate Analysis ---")

duplicate_count = df.duplicated().sum()
print("Duplicate rows:", duplicate_count)

print("\n--- Class Percentages ---")

class_percentages = (
    df["label"].value_counts(normalize=True) * 100
)

print(class_percentages.round(2))

# Step 3B: Examine example messages
print("\n--- Sample Spam Messages ---")

spam_messages = df[df["label"] == "spam"]
print(spam_messages["message"].head(5).to_string(index=False))

print("\n--- Sample Ham Messages ---")

ham_messages = df[df["label"] == "ham"]
print(ham_messages["message"].head(5).to_string(index=False))

# Step 3C: Examine message lengths
df["message_length"] = df["message"].str.len()

print("\n--- Average Message Length ---")

print(
    df.groupby("label")["message_length"]
    .mean()
    .round(2)
)
