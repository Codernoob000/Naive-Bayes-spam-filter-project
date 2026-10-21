
import re
import math
from collections import Counter


def tokenize(message):
    """Convert a message into lowercase word tokens."""
    return re.findall(r"\b\w+\b", message.lower())


class NaiveBayesClassifier:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classes = []
        self.class_counts = Counter()
        self.word_counts = {}
        self.total_words = Counter()
        self.vocabulary = set()
        self.class_log_prior = {}

    def fit(self, messages, labels):
        """Learn class priors and word frequencies."""
        self.classes = sorted(set(labels))

        self.class_counts = Counter(labels)
        self.word_counts = {
            label: Counter() for label in self.classes
        }
        self.total_words = Counter()
        self.vocabulary = set()

        # Count words separately for each class
        for message, label in zip(messages, labels):
            tokens = tokenize(message)

            self.word_counts[label].update(tokens)
            self.total_words[label] += len(tokens)
            self.vocabulary.update(tokens)

        # Learn class prior probabilities in log space
        total_messages = len(labels)

        self.class_log_prior = {
            label: math.log(
                self.class_counts[label] / total_messages
            )
            for label in self.classes
        }

        return self

    def predict_one(self, message):
        """Predict the class of one message."""
        tokens = tokenize(message)
        vocabulary_size = len(self.vocabulary)

        scores = {}

        for label in self.classes:
            score = self.class_log_prior[label]

            denominator = (
                self.total_words[label]
                + self.alpha * vocabulary_size
            )

            for word in tokens:
                count = self.word_counts[label][word]

                probability = (
                    count + self.alpha
                ) / denominator

                score += math.log(probability)

            scores[label] = score

        return max(scores, key=scores.get)

    def predict(self, messages):
        """Predict classes for multiple messages."""
        return [
            self.predict_one(message)
            for message in messages
        ]
