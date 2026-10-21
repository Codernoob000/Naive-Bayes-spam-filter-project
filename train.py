import math
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from naive_bayes import NaiveBayesClassifier


# 1. Load the dataset
data_path = Path(__file__).parent / "data" / "SMSSpamCollection"

df = pd.read_csv(
    data_path,
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8",
)

# 2. Remove exact duplicate rows
df = df.drop_duplicates().reset_index(drop=True)

# 3. Separate messages and labels
X = df["message"]
y = df["label"]

# 4. Create a reproducible, stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# 5. Train our own Naive Bayes classifier
model = NaiveBayesClassifier(alpha=1.0)
model.fit(X_train.tolist(), y_train.tolist())

print("Training completed!")
print("Training messages:", len(X_train))
print("Testing messages:", len(X_test))
print("Vocabulary size:", len(model.vocabulary))

# 6. Predict unseen test messages
predictions = model.predict(X_test.tolist())

# 7. Evaluate the model
print("\n--- Model Evaluation ---")

print(
    f"Accuracy: "
    f"{accuracy_score(y_test, predictions):.4f}"
)

print(
    f"Precision: "
    f"{precision_score(y_test, predictions, pos_label='spam'):.4f}"
)

print(
    f"Recall: "
    f"{recall_score(y_test, predictions, pos_label='spam'):.4f}"
)

print(
    f"F1-score: "
    f"{f1_score(y_test, predictions, pos_label='spam'):.4f}"
)

# Rows = actual labels; columns = predicted labels
labels_order = ["ham", "spam"]

cm = confusion_matrix(
    y_test,
    predictions,
    labels=labels_order,
)

print("\n--- Confusion Matrix ---")
print("Label order: ham, spam")
print(cm)

print("\n--- Classification Report ---")
print(
    classification_report(
        y_test,
        predictions,
        labels=labels_order,
        zero_division=0,
    )
)


# Step 9: Find words most associated with spam or ham

alpha = model.alpha
vocab_size = len(model.vocabulary)

def word_log_ratio(word):
    # Smoothed probability of this word in spam
    spam_probability = (
        model.word_counts["spam"][word] + alpha
    ) / (
        model.total_words["spam"] + alpha * vocab_size
    )

    # Smoothed probability of this word in ham
    ham_probability = (
        model.word_counts["ham"][word] + alpha
    ) / (
        model.total_words["ham"] + alpha * vocab_size
    )

    return math.log(spam_probability / ham_probability)


ranked_words = sorted(
    model.vocabulary,
    key=word_log_ratio,
    reverse=True
)

print("\n--- Top 20 Spam-Associated Words ---")

for word in ranked_words[:20]:
    print(f"{word:20} {word_log_ratio(word):.4f}")

print("\n--- Top 20 Ham-Associated Words ---")

for word in ranked_words[-20:][::-1]:
    print(f"{word:20} {word_log_ratio(word):.4f}")


# Step 12: Save evaluation results and confusion matrix

from pathlib import Path

output_dir = Path(__file__).parent / "results"
output_dir.mkdir(exist_ok=True)

# Save numerical evaluation results
results_text = f"""
NAIVE BAYES SPAM FILTER - EVALUATION RESULTS

Training messages: {len(X_train)}
Testing messages: {len(X_test)}
Vocabulary size: {len(model.vocabulary)}

Accuracy: {accuracy_score(y_test, predictions):.4f}
Spam precision: {precision_score(y_test, predictions, pos_label='spam'):.4f}
Spam recall: {recall_score(y_test, predictions, pos_label='spam'):.4f}
Spam F1-score: {f1_score(y_test, predictions, pos_label='spam'):.4f}

Confusion matrix (rows=actual, columns=predicted):
Label order: ham, spam
{cm}
"""

results_path = output_dir / "evaluation_results.txt"
results_path.write_text(results_text.strip(), encoding="utf-8")

# Plot the confusion matrix
plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels_order,
    yticklabels=labels_order,
)

plt.title("Naive Bayes Spam Filter - Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()

plot_path = output_dir / "confusion_matrix.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close()

print("\n--- Saved Artifacts ---")
print(f"Evaluation report: {results_path}")
print(f"Confusion matrix: {plot_path}")
