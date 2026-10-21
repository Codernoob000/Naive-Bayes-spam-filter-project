
import unittest

from naive_bayes import NaiveBayesClassifier


class TestNaiveBayesClassifier(unittest.TestCase):

    def setUp(self):
        self.model = NaiveBayesClassifier(alpha=1.0)

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

        self.model.fit(messages, labels)

    def test_spam_prediction(self):
        result = self.model.predict_one("free money prize")
        self.assertEqual(result, "spam")

    def test_ham_prediction(self):
        result = self.model.predict_one("hello friend")
        self.assertEqual(result, "ham")

    def test_unseen_words(self):
        result = self.model.predict_one(
            "unfamiliar vocabulary xyz"
        )
        self.assertIn(result, ["spam", "ham"])

    def test_empty_message(self):
        result = self.model.predict_one("")
        self.assertIn(result, ["spam", "ham"])


if __name__ == "__main__":
    unittest.main()
