
import pandas as pd
from pathlib import Path

from naive_bayes import NaiveBayesClassifier


def load_dataset():
    data_path = (
        Path(__file__).parent
        / "data"
        / "SMSSpamCollection"
    )

    df = pd.read_csv(
        data_path,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="utf-8",
    )

    return df.drop_duplicates().reset_index(drop=True)


def train_model(df):
    model = NaiveBayesClassifier(alpha=1.0)

    model.fit(
        df["message"].tolist(),
        df["label"].tolist(),
    )

    return model


def predict_message(model, message):
    return model.predict_one(message)


def main():
    print("=" * 48)
    print("       NAIVE BAYES SPAM DETECTOR")
    print("=" * 48)

    df = load_dataset()
    model = train_model(df)

    print(f"Model trained on {len(df)} unique messages.")
    print(f"Vocabulary size: {len(model.vocabulary)}")
    print("\nEnter a message to classify.")
    print("Type 'exit' to quit.")

    while True:
        print()
        message = input("Message: ").strip()

        if message.lower() == "exit":
            print("Thank you for using the Spam Detector!")
            break

        if not message:
            print("Please enter a non-empty message.")
            continue

        prediction = predict_message(model, message)

        if prediction == "spam":
            print("Prediction: SPAM")
            print("Warning: This message may be unwanted.")
        else:
            print("Prediction: HAM (NOT SPAM)")
            print("This message appears legitimate.")


if __name__ == "__main__":
    main()
