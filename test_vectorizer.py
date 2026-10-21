from vectorizer import BagOfWordsVectorizer

messages = [
    "free prize free",
    "hello prize",
    "win free"
]

vectorizer = BagOfWordsVectorizer()
vectors = vectorizer.fit_transform(messages)

print("Vocabulary:", vectorizer.vocabulary)
print("\nVectors:")

for message, vector in zip(messages, vectors):
    print(message, "->", vector)