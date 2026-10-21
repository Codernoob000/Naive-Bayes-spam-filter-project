
from naive_bayes import NaiveBayesClassifier

messages = [
    "free prize win",
    "win free money",
    "hello friend",
    "hello how are you",
]

labels = [
    "spam",
    "spam",
    "ham",
    "ham",
]

model = NaiveBayesClassifier(alpha=1.0)
model.fit(messages, labels)

test_messages = [
    "free money prize",
    "hello friend",
]

for message in test_messages:
    prediction = model.predict_one(message)
    print(f"Message: {message}")
    print(f"Prediction: {prediction}\n")
