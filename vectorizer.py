from collections import Counter


class BagOfWordsVectorizer:
    def __init__(self):
        self.vocabulary = {}

    def fit(self, messages):
        # Build vocabulary using training messages only
        unique_words = set()

        for message in messages:
            tokens = message.lower().split()
            unique_words.update(tokens)

        # Assign each word a numerical index
        self.vocabulary = {
            word: index
            for index, word in enumerate(sorted(unique_words))
        }

    def transform(self, messages):
        # Convert each message into a word-count vector
        vectors = []

        for message in messages:
            counts = Counter(message.lower().split())

            vector = [0] * len(self.vocabulary)

            for word, count in counts.items():
                if word in self.vocabulary:
                    index = self.vocabulary[word]
                    vector[index] = count

            vectors.append(vector)

        return vectors

    def fit_transform(self, messages):
        self.fit(messages)
        return self.transform(messages)