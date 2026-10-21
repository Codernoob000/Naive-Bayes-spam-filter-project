# PROJECT.md — Naive Bayes Spam Filter from Scratch

> **Purpose:** Complete technical documentation, onboarding guide, implementation reference, and viva preparation manual for the Naive Bayes Spam Filter course-end project.
>
> **Project type:** Machine Learning / Natural Language Processing  
> **Programming language:** Python  
> **Primary algorithm:** Multinomial Naive Bayes  
> **Application interface:** Interactive terminal  
> **Dataset:** UCI SMS Spam Collection  
> **Project status:** Working prototype with training, evaluation, word-association analysis, terminal predictions, and automated tests.

---

# Table of Contents

1. Project Overview
2. Problem Statement
3. Objectives and Scope
4. Features and Requirements
5. Technology Stack
6. Repository Structure
7. Prerequisites
8. Clone and Run the Project
9. Dataset Documentation
10. Machine Learning Fundamentals
11. Mathematical Foundation
12. System Architecture
13. Data Processing Pipeline
14. Implementation Guide
15. Model Training and Prediction
16. Evaluation Methodology
17. Understanding the Results
18. Spam-Associated Word Analysis
19. Terminal Application
20. Automated Testing
21. Configuration and Dependencies
22. Troubleshooting
23. Limitations and Future Improvements
24. Development Workflow
25. GitHub and Collaboration Guidelines
26. Viva Preparation: Questions and Answers
27. Presentation Guide
28. Reference Materials
29. Final Verification Checklist
30. Important Implementation Notes

---

# 1. Project Overview

## 1.1 What is this project?

The Naive Bayes Spam Filter is a text-classification application that predicts whether an SMS message is:

- **Spam:** An unwanted, unsolicited, promotional, or potentially deceptive message.
- **Ham:** A legitimate message that is not classified as spam.

The project uses a Multinomial Naive Bayes classifier implemented manually in Python. Instead of importing a ready-made Naive Bayes classifier, the implementation calculates class priors, word frequencies, smoothed word probabilities, and classification scores itself.

The application learns patterns from labeled SMS messages and uses those patterns to classify new messages.

### Example

Input:

`Congratulations! Claim your free prize now.`

Expected prediction:

`SPAM`

Input:

`Hey, are we meeting at college tomorrow?`

Expected prediction:

`HAM (NOT SPAM)`

These are examples from the demonstrated terminal application. Predictions depend on the trained model and may occasionally be incorrect.

## 1.2 Why is the project useful?

Spam messages can contain unsolicited advertisements, misleading prize claims, and suspicious offers. An automated classifier can help organize messages by identifying text patterns associated with unwanted communication.

This project also serves an educational purpose: it demonstrates how a basic machine learning algorithm works internally rather than treating the model as a black box.

## 1.3 What makes this a from-scratch implementation?

The custom classifier in `naive_bayes.py` implements:

- Text tokenization.
- Vocabulary collection.
- Class-specific word-frequency counting.
- Class prior estimation.
- Laplace smoothing.
- Log-probability scoring.
- Class prediction.

Scikit-learn is used for data splitting and evaluation metrics, not as the project's Naive Bayes classifier.

The standalone `vectorizer.py` demonstrates bag-of-words vectorization. The current classifier counts tokens directly and does not depend on that vectorizer.

---

# 2. Problem Statement

Manual identification of spam messages can be time-consuming and inconsistent. A machine learning model can learn statistical patterns from labeled examples and apply those patterns to unseen messages.

The problem is to develop a Python-based system that:

1. Accepts labeled SMS messages for training.
2. Learns differences between spam and legitimate messages.
3. Predicts the category of an unseen message.
4. Measures performance using standard classification metrics.
5. Identifies words statistically associated with each class.
6. Provides an interactive interface for testing messages.

The system is intended as an educational spam-classification prototype, not a production-grade security or anti-phishing service.

---

# 3. Objectives and Scope

## 3.1 Objectives

1. Understand Bayes' theorem and conditional probability.
2. Learn the principles of Multinomial Naive Bayes.
3. Process and explore a real-world text dataset.
4. Implement tokenization and word counting.
5. Calculate class priors and word likelihoods.
6. Handle unseen words using Laplace smoothing.
7. Evaluate predictions on held-out data.
8. Interpret a confusion matrix and classification metrics.
9. Identify words associated with spam.
10. Build an interactive terminal-based spam detector.
11. Write automated tests for important classifier behaviors.
12. Document the project for reproducibility and collaboration.

## 3.2 Scope

The current version supports binary SMS classification, training from a labeled dataset, evaluation, word-association analysis, and terminal predictions.

It does not currently implement:

- Email server integration.
- Real-time mailbox monitoring.
- Sender reputation analysis.
- URL reputation checking.
- A deployed web application.
- Deep learning or transformer-based classification.
- Production-grade threat detection.

These are possible future extensions, not existing features.

---

# 4. Features and Requirements

## 4.1 Implemented features

| Feature | Description |
|---|---|
| Dataset loading | Reads tab-separated SMS data |
| Duplicate removal | Removes exact duplicate label/message rows |
| Train/test split | Uses an 80:20 stratified split |
| Text tokenization | Converts messages into lowercase word tokens |
| Vocabulary learning | Collects unique words |
| Naive Bayes training | Calculates class priors and word counts |
| Laplace smoothing | Prevents zero word probabilities |
| Prediction | Selects the highest-scoring class |
| Evaluation | Calculates accuracy, precision, recall and F1-score |
| Confusion matrix | Shows correct and incorrect predictions |
| Word analysis | Ranks words by spam-versus-ham association |
| Terminal interface | Accepts interactive message input |
| Automated tests | Checks core prediction and edge-case behavior |
| Saved results | Exports evaluation metrics and a confusion matrix image |

## 4.2 Functional requirements

- The program must load labeled SMS messages.
- The model must learn from the training data.
- The model must classify a user-entered message.
- The training pipeline must evaluate predictions on held-out data.
- The program must display evaluation results.
- The test suite must validate core classifier behavior.

## 4.3 Non-functional requirements

- The project should run on a typical laptop.
- The code should be understandable and modular.
- The training/test split should be reproducible.
- Dependencies should be documented.
- Predictions should not require an internet connection after setup.
- The project should provide meaningful error messages for common setup failures.

---

# 5. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Dataset loading and tabular data handling |
| NumPy | Numerical computing dependency |
| scikit-learn | Dataset splitting and evaluation metrics |
| Matplotlib | Plot generation |
| Seaborn | Confusion matrix visualization |
| `re` | Regular-expression tokenization |
| `math` | Logarithms and numerical calculations |
| `collections.Counter` | Word-frequency and class counting |
| `pathlib` | File path management |
| `unittest` | Automated tests |
| VS Code | Recommended development environment |
| Git | Version control |
| GitHub | Repository hosting and collaboration |

## Why these tools?

### Python

Python provides readable syntax and a large ecosystem for data analysis and machine learning.

### Pandas

Pandas loads the SMS dataset into a DataFrame, making it easy to inspect labels, messages, duplicates, and class distributions.

### scikit-learn

The project uses selected utilities from scikit-learn:

- `train_test_split` for reproducible stratified splitting.
- `accuracy_score` for accuracy.
- `precision_score` for precision.
- `recall_score` for recall.
- `f1_score` for F1-score.
- `confusion_matrix` for error counts.
- `classification_report` for a combined metric summary.

It does not use `sklearn.naive_bayes.MultinomialNB` for the primary model.

### Matplotlib and Seaborn

These libraries generate the confusion matrix visualization saved in the `results/` directory.

### Python standard library

`re`, `math`, `collections`, `pathlib`, and `unittest` provide core functionality without requiring additional installations.

---

# 6. Repository Structure

The project has evolved incrementally. The following structure reflects the files developed so far.

```text
naive-bayes-spam-filter/
│
├── data/
│   └── SMSSpamCollection
│
├── results/
│   ├── confusion_matrix.png
│   └── evaluation_results.txt
│
├── tests/
│   └── test_classifier.py
│
├── naive_bayes.py
├── vectorizer.py
├── prepare_data.py
├── main.py
├── train.py
├── app_terminal.py
├── test_naive_bayes.py
├── test_vectorizer.py
│
├── requirements.txt
├── README.md
├── PROJECT.md
├── .gitignore
└── .venv/
```

The `.venv/` directory is created locally and should not be committed to Git.

## 6.1 File responsibilities

### `naive_bayes.py`

Contains the custom Naive Bayes classifier.

It is responsible for learning class-specific word counts, class priors, smoothed likelihoods, and predictions.

### `vectorizer.py`

Contains the standalone `BagOfWordsVectorizer` class.

It demonstrates how messages can be converted into fixed-length word-count vectors. The current classifier does not import this class.

### `prepare_data.py`

Loads the dataset, removes exact duplicate rows, performs the training/test split, and explores word frequencies.

It also contains the initial tokenization and vocabulary exploration code developed during the project.

### `train.py`

Runs the main training and evaluation pipeline.

It:

1. Loads the dataset.
2. Removes exact duplicate rows.
3. Splits the data.
4. Trains the custom classifier.
5. Predicts test labels.
6. Calculates evaluation metrics.
7. Ranks spam-associated and ham-associated words.
8. Saves evaluation artifacts.

### `app_terminal.py`

Trains the classifier on all unique labeled messages and provides an interactive terminal interface.

### `main.py`

Contains the initial dataset exploration program.

### `test_naive_bayes.py`

Provides a small manually understandable dataset for testing predictions.

### `test_vectorizer.py`

Tests the standalone bag-of-words vectorizer.

### `tests/test_classifier.py`

Contains automated unit tests for spam prediction, ham prediction, unseen words, and empty messages.

### `requirements.txt`

Records the installed Python package versions used during development.

### `results/`

Stores the evaluation report and confusion matrix image generated by the training pipeline.

---

# 7. Prerequisites

Before cloning the project, install:

- Python 3.10 or a compatible newer version.
- Git.
- A terminal or command prompt.
- A code editor such as Visual Studio Code.

Verify Python:

```powershell
python --version
```

Verify Git:

```powershell
git --version
```

The exact Python version used during development should be recorded before claiming that a particular version has been verified.

---

# 8. Clone and Run the Project

## 8.1 Clone the repository

Replace the example URL with the actual GitHub repository URL.

```powershell
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd naive-bayes-spam-filter
```

Do not use the placeholder literally.

## 8.2 Create a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

A successful activation usually displays `(.venv)` in the terminal prompt.

If PowerShell blocks activation, use Command Prompt:

```bat
.venv\Scripts\activate.bat
```

Alternatively, run the virtual environment's Python executable directly.

## 8.3 Install dependencies

```powershell
python -m pip install -r requirements.txt
```

Verify the essential imports:

```powershell
python -c "import pandas, numpy, sklearn, matplotlib, seaborn; print('Environment ready!')"
```

If installation fails, consult the troubleshooting section.

## 8.4 Obtain the dataset

Dataset source:

https://archive.ics.uci.edu/dataset/228/sms+spam+collection

Download and extract the SMS Spam Collection dataset.

Place the file named `SMSSpamCollection` in:

```text
data/SMSSpamCollection
```

The application expects a tab-separated file with two fields per record: a label and a message.

The repository should include the dataset only if its applicable distribution terms and attribution requirements permit it. Otherwise, document the download procedure and keep the dataset out of the repository.

## 8.5 Run dataset exploration

```powershell
python main.py
```

This displays the first five records, dataset dimensions, label counts, and missing-value counts.

## 8.6 Run data preparation

```powershell
python prepare_data.py
```

This performs the split and prints the training/test class distributions, vocabulary statistics, and other exploratory information.

The script also includes the standalone vectorizer experiment code added during development.

## 8.7 Train and evaluate the classifier

```powershell
python train.py
```

This is the main experimental pipeline.

It prints evaluation metrics, a confusion matrix, a classification report, and word-association rankings.

It also generates:

```text
results/evaluation_results.txt
results/confusion_matrix.png
```

## 8.8 Launch the terminal detector

```powershell
python app_terminal.py
```

Enter a message at the prompt.

Example:

```text
Message: Congratulations! Claim your free prize now.
Prediction: SPAM
Warning: This message may be unwanted.
```

Type `exit` to stop the application.

The terminal application trains on all unique labeled messages. Its predictions are intended for demonstration, not held-out evaluation.

## 8.9 Run the tests

```powershell
python -m unittest discover -s tests -v
```

The current four classifier tests should pass.

Additional demonstration scripts can be run with:

```powershell
python test_naive_bayes.py
python test_vectorizer.py
```

---

# 9. Dataset Documentation

## 9.1 Dataset source

The project uses the UCI SMS Spam Collection.

Official page:

https://archive.ics.uci.edu/dataset/228/sms+spam+collection

The dataset consists of labeled SMS text messages collected for spam-classification research.

## 9.2 Dataset format

Each record contains two fields:

| Field | Meaning |
|---|---|
| Label | `ham` or `spam` |
| Message | SMS message text |

The file is tab-separated and does not contain a header row.

The loading code is:

```python
df = pd.read_csv(
    data_path,
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8",
)
```

`sep="\t"` tells Pandas that a tab separates the label and message.

`header=None` indicates that the file does not start with column names.

`names=["label", "message"]` assigns meaningful column names.

## 9.3 Dataset statistics

The original local dataset contained 5,572 rows.

After removing exact duplicate label/message rows, the working dataset contained 5,169 unique rows.

The verified class counts are:

| Class | Unique messages | Percentage |
|---|---:|---:|
| Ham | 4,516 | 87.17% |
| Spam | 653 | 12.83% |
| Total | 5,169 | 100% |

These values describe the deduplicated dataset, before the train/test split.

## 9.4 Duplicate handling

The project uses:

```python
df = df.drop_duplicates().reset_index(drop=True)
```

This removes exact duplicate rows, including repeated label/message pairs.

It does not automatically remove different messages with similar wording.

If the same message text appears with different labels, those rows are not necessarily duplicates under this operation.

For rigorous evaluation, future versions should investigate repeated message text across classes and avoid related messages leaking across train/test partitions.

## 9.5 Class imbalance

Ham messages are much more common than spam messages.

The original model therefore cannot be evaluated using accuracy alone.

For example, a classifier that always predicts ham would achieve approximately 87.17% accuracy on the deduplicated full dataset but detect no spam messages.

Precision, recall, F1-score, and the confusion matrix provide additional information about the model's performance on the minority class.

---

# 10. Machine Learning Fundamentals

## 10.1 What is machine learning?

Machine learning is a field of computing in which a program learns patterns from data and uses those patterns to make predictions or decisions.

In this project, the training data consists of SMS messages with known labels.

The classifier learns statistical relationships between words and message categories.

## 10.2 What is supervised learning?

Supervised learning uses examples that contain both input features and correct output labels.

For this project:

- Input: SMS message.
- Target: `spam` or `ham`.
- Learning objective: predict the label of a new message.

## 10.3 What is text classification?

Text classification assigns a category to a text document.

Examples include:

- Spam detection.
- Sentiment analysis.
- Topic categorization.
- Email routing.
- Document classification.

This project performs binary text classification.

## 10.4 What is Naive Bayes?

Naive Bayes is a family of probabilistic classifiers based on Bayes' theorem.

The algorithm estimates the probability of each class given the observed features and chooses the class with the highest score.

It is called *naive* because it makes a conditional-independence assumption about features given the class.

For text classification, this means the model treats word occurrences as conditionally independent given the message class.

Natural language does not actually obey this assumption perfectly. Nevertheless, Naive Bayes can be effective for text classification.

## 10.5 What is Multinomial Naive Bayes?

Multinomial Naive Bayes is a Naive Bayes variant designed for count-like features.

In this project, each message is represented conceptually by counts of its words.

For example:

```text
Message: free prize free

free  -> 2
prize -> 1
```

The model learns how often each word occurs within spam and ham training messages.

The current classifier performs these counts directly rather than requiring a separate numerical vector for every message.

## 10.6 What is a prior probability?

A prior probability represents the estimated probability of a class before considering the words in a particular message.

The class prior is calculated as:

\[
P(C)=\frac{N_C}{N}
\]

Where:

- \(C\) is a class.
- \(N_C\) is the number of training messages in that class.
- \(N\) is the total number of training messages.

The training split contains 522 spam messages and 3,613 ham messages.

Therefore:

\[
P(\text{spam})=\frac{522}{4135}
\]

\[
P(\text{ham})=\frac{3613}{4135}
\]

The prior favors ham because ham is more common in the training data.

## 10.7 What is likelihood?

A likelihood measures how probable the observed evidence is under a particular class.

For this project, the model estimates the probability of a word given the class:

\[
P(w\mid C)
\]

For example, it may learn that the word `prize` is more likely to occur in spam messages than in ham messages.

The exact probabilities are estimated from training counts and smoothing.

## 10.8 What is conditional independence?

Naive Bayes assumes that the word features are conditionally independent given the class.

In practical terms, the model multiplies the class-specific probability contribution of each observed word.

This is a simplifying assumption, not a claim that words are truly independent in natural language.

## 10.9 What is a vocabulary?

The vocabulary is the set of unique tokens found in the training messages.

In the training pipeline, the vocabulary contains 7,720 unique tokens.

The interactive application, which trains on all 5,169 unique messages, learns a vocabulary of 8,753 tokens.

These sizes differ because the training data differs.

## 10.10 What is tokenization?

Tokenization splits a message into word-like units.

The custom tokenizer is:

```python
def tokenize(message):
    return re.findall(r"\b\w+\b", message.lower())
```

It performs two main operations:

1. Converts text to lowercase.
2. Extracts sequences of word characters using a regular expression.

For example:

```text
Original: Win a FREE prize now!
Tokens: ['win', 'a', 'free', 'prize', 'now']
```

The tokenizer retains digits and underscores as word characters and does not preserve punctuation as separate tokens.

---

# 11. Mathematical Foundation

## 11.1 Bayes' theorem

Bayes' theorem is:

\[
P(C\mid M)=\frac{P(M\mid C)P(C)}{P(M)}
\]

Where:

- \(C\): a class, such as spam.
- \(M\): the observed message.
- \(P(C\mid M)\): posterior probability of the class given the message.
- \(P(M\mid C)\): likelihood of observing the message under the class.
- \(P(C)\): prior probability of the class.
- \(P(M)\): probability of the message.

The model compares spam and ham using these probabilities.

## 11.2 Multinomial Naive Bayes

Let a message contain words \(w_1,w_2,\ldots,w_n\).

The unnormalized class score is:

\[
P(C)\prod_{i=1}^{n}P(w_i\mid C)
\]

If a word occurs multiple times, it contributes multiple times to the score.

The model predicts the class with the highest score.

## 11.3 Why use logarithms?

Multiplying many small probabilities can produce numerical underflow.

For example:

\[
0.01\times0.02\times0.03
\]

becomes extremely small as the number of terms increases.

Taking logarithms converts multiplication into addition:

\[
\log(ab)=\log(a)+\log(b)
\]

Therefore, the model computes:

\[
\operatorname{score}(C,M)=
\log P(C)+
\sum_{i=1}^{n}\log P(w_i\mid C)
\]

The class with the highest log score is selected.

Log scores are ranking values, not calibrated probabilities.

## 11.4 Laplace smoothing

Without smoothing, a word that never appears in one class would have estimated probability zero.

That zero would make the entire unsmoothed product zero for that class whenever the word appears.

Laplace smoothing addresses this problem:

\[
P(w\mid C)=
\frac{\operatorname{count}(w,C)+\alpha}
{N_C^{words}+\alpha|V|}
\]

Where:

- \(\operatorname{count}(w,C)\): count of the word in the class.
- \(N_C^{words}\): total token occurrences in the class.
- \(\alpha\): smoothing parameter.
- \(|V|\): vocabulary size.

The project uses \(\alpha=1.0\).

This is called add-one smoothing.

## 11.5 Worked example

Consider this miniature training dataset:

| Message | Label |
|---|---|
| free prize win | Spam |
| win free money | Spam |
| hello friend | Ham |
| hello how are you | Ham |

The vocabulary is:

\[
V=\{\text{free, prize, win, money, hello, friend, how, are, you}\}
\]

Therefore, the vocabulary size is 9.

Spam contains 6 total token occurrences; ham contains 5.

For the word `free` in spam:

\[
P(\text{free}\mid\text{spam})
=\frac{2+1}{6+9}
=\frac{3}{15}
\]

For `hello` in spam:

\[
P(\text{hello}\mid\text{spam})
=\frac{0+1}{6+9}
=\frac{1}{15}
\]

The second probability is not zero because of Laplace smoothing.

This example illustrates how word counts are converted into probabilities. It is not a calculation of the full posterior probability for an arbitrary message.

---

# 12. System Architecture

The project has two main execution paths: offline training/evaluation and interactive terminal prediction.

## 12.1 Training and evaluation architecture

```text
              SMS Dataset
                   |
                   v
            Load with Pandas
                   |
                   v
          Remove Exact Duplicates
                   |
                   v
          Stratified Train/Test Split
                   |
           +-------+-------+
           |               |
           v               v
       Training Set     Test Set
           |               |
           v               |
       Tokenization         |
           |               |
           v               |
      Word/Class Counts     |
           |               |
           v               |
        Class Priors        |
           |               |
           v               |
    Smoothed Probabilities  |
           |               |
           v               |
     Custom Naive Bayes <---+
           |
           v
       Predictions
           |
           v
   Evaluation Metrics
           |
           v
      Saved Results
```

## 12.2 Terminal prediction architecture

```text
       User Enters Message
                |
                v
        Terminal Application
                |
                v
         Load SMS Dataset
                |
                v
         Train Naive Bayes
                |
                v
       Tokenize Input Message
                |
                v
        Calculate Class Scores
                |
                v
        Select Highest Score
                |
                v
        Display Spam or Ham
```

## 12.3 Design decisions

The current project prioritizes readability and educational value.

- The classifier is implemented independently of the user interface.
- The training pipeline evaluates the model on held-out data.
- The terminal application is a separate demonstration.
- The evaluation report and visualization are saved as files.
- The standalone vectorizer is available for learning but is not integrated into the current classifier.

---

# 13. Data Processing Pipeline

## 13.1 Load the dataset

The project reads the tab-separated file using Pandas.

The columns are named `label` and `message`.

## 13.2 Remove exact duplicates

The training pipeline removes exact duplicate rows before splitting.

This reduces repeated identical label/message records.

## 13.3 Separate features and labels

```python
X = df["message"]
y = df["label"]
```

`X` contains the message text.

`y` contains the correct target label.

## 13.4 Split the data

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)
```

The split has these properties:

- 80% training data.
- 20% test data.
- Fixed random seed for reproducibility.
- Stratification to preserve class proportions.

The resulting split contains 4,135 training messages and 1,034 test messages.

## 13.5 Avoiding data leakage

The vocabulary and word counts used by the evaluated model are learned from training messages only.

The test set is used for prediction and evaluation, not for fitting the classifier.

This helps measure generalization to held-out examples.

However, a random split does not guarantee that related or near-duplicate messages are separated. That issue should be investigated in future evaluation work.

---

# 14. Implementation Guide

## 14.1 Tokenization

In `naive_bayes.py`:

```python
def tokenize(message):
    return re.findall(r"\b\w+\b", message.lower())
```

This creates lowercase word tokens from the input text.

## 14.2 Model initialization

```python
class NaiveBayesClassifier:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classes = []
        self.class_counts = Counter()
        self.word_counts = {}
        self.total_words = Counter()
        self.vocabulary = set()
        self.class_log_prior = {}
```

The attributes store:

- `alpha`: smoothing parameter.
- `classes`: available labels.
- `class_counts`: number of training messages in each class.
- `word_counts`: token frequencies per class.
- `total_words`: total token occurrences per class.
- `vocabulary`: all unique training tokens.
- `class_log_prior`: log prior probabilities.

## 14.3 Training with `fit()`

The `fit()` method:

1. Identifies the classes.
2. Counts training messages per class.
3. Initializes separate word counters.
4. Tokenizes every training message.
5. Updates the relevant class word counts.
6. Tracks the total token occurrences per class.
7. Builds the vocabulary.
8. Calculates class log priors.

The model stores sufficient statistics for later predictions.

## 14.4 Prediction with `predict_one()`

For one message, the classifier:

1. Tokenizes the message.
2. Starts with the class log prior.
3. Computes a smoothed probability for each token.
4. Adds the token log probabilities to the class score.
5. Compares scores for spam and ham.
6. Returns the class with the highest score.

## 14.5 Prediction with `predict()`

```python
def predict(self, messages):
    return [
        self.predict_one(message)
        for message in messages
    ]
```

This applies `predict_one()` to every message in a collection.

It returns a list of predicted labels.

## 14.6 Unknown words

If a token is absent from the learned vocabulary, its count in each class is zero.

Laplace smoothing assigns a nonzero probability to that token under each class.

The implementation therefore handles previously unseen words without crashing.

## 14.7 Empty messages

An empty message produces no tokens.

The classifier then compares class log priors. In the current implementation, it predicts the class with the larger prior.

An empty message is not meaningful evidence of spam or ham. A future interface should reject empty input rather than presenting the prior-based result as a useful classification.

---

# 15. Model Training and Prediction

## 15.1 Training pipeline

The command is:

```powershell
python train.py
```

The training pipeline:

1. Loads the dataset.
2. Removes exact duplicates.
3. Creates a stratified train/test split.
4. Instantiates `NaiveBayesClassifier(alpha=1.0)`.
5. Fits the classifier on training messages.
6. Predicts labels for the test messages.
7. Calculates evaluation metrics.
8. Prints the classification report.
9. Ranks class-associated words.
10. Saves evaluation results and the confusion matrix.

## 15.2 Terminal application

The command is:

```powershell
python app_terminal.py
```

The application trains on all unique labeled messages, then accepts repeated user inputs.

It stops when the user enters `exit`.

Because the application trains on the full labeled dataset, its model is not the same fitted model that produced the held-out test results.

## 15.3 Why not evaluate the terminal model on the training data?

A model is expected to perform well on data it has already seen.

Evaluation on training data can therefore exaggerate its ability to generalize.

The reported metrics should come from the held-out test predictions produced by `train.py`, not from the interactive application.

---

# 16. Evaluation Methodology

The project uses several metrics to measure classification performance.

## 16.1 Confusion matrix

The confusion matrix is:

```text
                 Predicted
              Ham       Spam

Actual Ham     888        15
Actual Spam      6       125
```

Rows represent actual labels; columns represent predicted labels.

| Term | Meaning | Result |
|---|---|---:|
| True negative | Ham correctly predicted as ham | 888 |
| False positive | Ham incorrectly predicted as spam | 15 |
| False negative | Spam incorrectly predicted as ham | 6 |
| True positive | Spam correctly predicted as spam | 125 |

The terms above treat spam as the positive class.

## 16.2 Accuracy

Accuracy measures the fraction of all predictions that are correct.

\[
Accuracy=\frac{TP+TN}{TP+TN+FP+FN}
\]

For this project:

\[
Accuracy=\frac{125+888}{1034}
\approx0.9797
\]

Accuracy is approximately 97.97%.

## 16.3 Precision

Precision measures how many predicted spam messages were actually spam.

\[
Precision=\frac{TP}{TP+FP}
\]

\[
Precision=\frac{125}{125+15}
\approx0.8929
\]

Spam precision is approximately 89.29%.

## 16.4 Recall

Recall measures how many actual spam messages were detected.

\[
Recall=\frac{TP}{TP+FN}
\]

\[
Recall=\frac{125}{125+6}
\approx0.9542
\]

Spam recall is approximately 95.42%.

## 16.5 F1-score

The F1-score is the harmonic mean of precision and recall.

\[
F1=2\frac{Precision\times Recall}{Precision+Recall}
\]

The spam F1-score is approximately 92.25%.

## 16.6 Why all four metrics matter

A high accuracy score can hide poor performance on a minority class.

Precision matters because false positives can cause legitimate messages to be treated as spam.

Recall matters because false negatives allow spam to pass through undetected.

F1-score summarizes the balance between precision and recall.

No single metric fully describes all the practical consequences of a classification system.

---

# 17. Understanding the Results

The recorded evaluation results are:

| Metric | Result |
|---|---:|
| Training messages | 4,135 |
| Testing messages | 1,034 |
| Training vocabulary size | 7,720 |
| Accuracy | 97.97% |
| Spam precision | 89.29% |
| Spam recall | 95.42% |
| Spam F1-score | 92.25% |

The model correctly classified 888 ham messages and 125 spam messages.

It incorrectly flagged 15 ham messages and missed 6 spam messages.

These results are encouraging for a small educational project. They do not establish that the model will achieve the same performance on modern spam campaigns or messages from other datasets.

## 17.1 Results artifact

The confusion matrix is saved at:

```text
results/confusion_matrix.png
```

The text report is saved at:

```text
results/evaluation_results.txt
```

The matrix image can be included in the README or project report.

---

# 18. Spam-Associated Word Analysis

The project ranks words using a smoothed log-likelihood ratio.

\[
S(w)=\log\frac{P(w\mid spam)}{P(w\mid ham)}
\]

A positive score indicates a stronger association with spam.

A negative score indicates a stronger association with ham.

A score near zero indicates more similar estimated probabilities.

## 18.1 Recorded spam-associated words

The first training run reported the following words:

| Word | Log-ratio score |
|---|---:|
| claim | 5.3588 |
| prize | 5.2580 |
| 150p | 4.8526 |
| tone | 4.6793 |
| co | 4.6793 |
| www | 4.6379 |
| 18 | 4.6237 |
| 500 | 4.5341 |
| guaranteed | 4.5341 |
| 1000 | 4.4357 |

These scores are from the recorded training run and should be regenerated if the data or implementation changes.

## 18.2 Why are these words associated with spam?

Words such as `claim`, `prize`, and `guaranteed` often occur in promotional or prize-related messages.

Tokens such as `150p` and `150ppm` may reflect premium-rate SMS promotions in the dataset.

These patterns are dataset-specific associations, not universal rules.

A legitimate message may contain the word `prize`, and a spam message may contain none of the top-ranked words.

## 18.3 Important interpretation

The ranking describes word-level associations learned from training data.

It does not measure a word's causal effect, guarantee a prediction, or provide a complete explanation of a particular message's classification.

---

# 19. Terminal Application

The terminal interface is implemented in `app_terminal.py`.

It loads the dataset, removes exact duplicates, trains a classifier, and repeatedly prompts the user for a message.

## 19.1 Example session

```text
================================================
       NAIVE BAYES SPAM DETECTOR
================================================
Model trained on 5169 unique messages.
Vocabulary size: 8753

Enter a message to classify.
Type 'exit' to quit.

Message: Congratulations! Claim your free prize now.
Prediction: SPAM
Warning: This message may be unwanted.

Message: Hey, are we meeting at college tomorrow?
Prediction: HAM (NOT SPAM)
This message appears legitimate.
```

The vocabulary size is 8,753 because the application trains on all 5,169 unique messages, rather than only the 4,135 training messages.

## 19.2 How predictions work

The terminal application calls:

```python
prediction = model.predict_one(message)
```

It then displays a user-friendly label based on the returned class.

The interface does not currently display calibrated confidence scores, and its text messages should not be interpreted as proof that a message is safe.

---

# 20. Automated Testing

The project includes four unit tests in `tests/test_classifier.py`.

Run them using:

```powershell
python -m unittest discover -s tests -v
```

## 20.1 Test cases

| Test | Purpose |
|---|---|
| Spam prediction | Checks an obvious spam example |
| Ham prediction | Checks a legitimate message example |
| Unseen words | Checks that unknown vocabulary does not crash prediction |
| Empty message | Checks that empty input is handled without an exception |

The recorded test output was:

```text
Ran 4 tests in 0.002s

OK
```

## 20.2 What do these tests prove?

They demonstrate that selected expected behaviors work on a small controlled dataset.

They do not prove that the model is correct for every possible message or that the evaluation results will generalize to every dataset.

Further tests should cover invalid labels, mismatched input lengths, repeated calls to `fit()`, and empty training data.

---

# 21. Configuration and Dependencies

## 21.1 Smoothing parameter

The classifier is initialized with:

```python
model = NaiveBayesClassifier(alpha=1.0)
```

The parameter `alpha` controls additive smoothing.

Changing it can affect word probabilities and predictions. Any comparison between different smoothing values should use a validation strategy rather than repeatedly tuning against the held-out test set.

## 21.2 Random seed

The split uses:

```python
random_state=42
```

This makes the split reproducible for the same dataset and compatible library behavior.

The seed is not a model parameter and does not make the classifier intrinsically more accurate.

## 21.3 Requirements

The environment uses Pandas, NumPy, scikit-learn, Matplotlib, and Seaborn.

The standard library provides the remaining tools.

The `requirements.txt` file records the installed package versions from the development environment. A future release should consider a minimal dependency file containing only the packages the project actually requires.

---

# 22. Troubleshooting

## 22.1 `FileNotFoundError`

**Cause:** The dataset is missing or stored in the wrong directory.

**Solution:** Confirm that the file exists at:

```text
data/SMSSpamCollection
```

Run commands from the project root.

## 22.2 `ModuleNotFoundError`

**Cause:** A dependency is missing or the wrong Python interpreter is active.

**Solution:**

```powershell
python -m pip install -r requirements.txt
```

Verify the active interpreter:

```powershell
python -c "import sys; print(sys.executable)"
```

It should point to the project's `.venv` environment when that environment is active.

## 22.3 PowerShell activation fails

Try Command Prompt activation:

```bat
.venv\Scripts\activate.bat
```

Or run the environment's Python executable directly.

## 22.4 Pip errors after an upgrade

Check:

```powershell
python -m pip --version
```

If pip is broken, try:

```powershell
python -m ensurepip --upgrade
```

Use `python -m pip` to target the selected interpreter consistently.

If the environment remains broken, recreate `.venv` and reinstall the dependencies. Preserve the dataset and project source files.

## 22.5 Unexpected evaluation results

Check that:

- The dataset has not changed.
- Exact duplicate removal is applied consistently.
- The same split parameters are used.
- The model is trained only on training messages.
- The evaluation uses the corresponding test labels.
- No preprocessing step has accidentally used test data to learn model parameters.

## 22.6 Confusion matrix is missing

Install the visualization dependencies:

```powershell
python -m pip install matplotlib seaborn
```

Then rerun:

```powershell
python train.py
```

Confirm that the output directory is `results/`.

---

# 23. Limitations and Future Improvements

## 23.1 Current limitations

### Bag-of-words assumptions

The model counts words but does not understand their full meaning or order.

### Conditional independence

The algorithm assumes conditional independence between word features, which is an approximation for natural language.

### Dataset limitations

The classifier learns from a particular SMS dataset. Its performance may change when applied to modern messages, other languages, or other communication channels.

### Tokenization

The current tokenizer uses a simple regular expression. It may treat certain punctuation patterns, contractions, URLs, and Unicode text differently than a more specialized text processor.

### Class imbalance

The dataset contains substantially more ham than spam messages. This makes class-sensitive evaluation important.

### Limited context

The model does not consider sender identity, URL reputation, external threat intelligence, or message history.

### Probability calibration

The classifier uses log-probability scores for ranking classes. These scores are not calibrated confidence estimates.

### Evaluation leakage risks

Exact duplicate rows are removed, but repeated message text and near-duplicates should be examined to assess whether related messages cross the training/test boundary.

## 23.2 Future improvements

1. Integrate the standalone vectorizer into a cleaner modular architecture.
2. Add explicit input validation and clearer errors.
3. Improve tokenization and investigate URL normalization.
4. Add tests for invalid inputs and retraining behavior.
5. Compare against a scikit-learn MultinomialNB baseline.
6. Evaluate different smoothing parameters using a validation set.
7. Investigate cross-validation and duplicate-aware evaluation.
8. Add a Streamlit interface.
9. Save and load trained model parameters.
10. Explore TF-IDF or other feature representations.
11. Evaluate the system on additional datasets.
12. Add explainability features showing which words contributed to a prediction.

These are proposed improvements, not features that should be claimed as completed.

---

# 24. Development Workflow

## 24.1 Recommended workflow for a new contributor

1. Clone the repository.
2. Create a virtual environment.
3. Install dependencies.
4. Obtain the dataset.
5. Run the automated tests.
6. Run the training pipeline.
7. Inspect the evaluation artifacts.
8. Run the terminal application.
9. Read the classifier implementation.
10. Make changes on a separate Git branch.
11. Run tests after making changes.
12. Submit changes through a pull request.

## 24.2 Recommended order for understanding the code

Read the source files in this order:

1. `test_naive_bayes.py` — understand the smallest working example.
2. `naive_bayes.py` — understand the algorithm.
3. `test_classifier.py` — understand expected behavior.
4. `prepare_data.py` — understand data preparation.
5. `train.py` — understand the full evaluation pipeline.
6. `app_terminal.py` — understand interactive predictions.
7. `vectorizer.py` — understand standalone bag-of-words vectorization.

This order moves from a simple example to the full system.

---

# 25. GitHub and Collaboration Guidelines

## 25.1 Files to exclude

The `.gitignore` should exclude:

```gitignore
.venv/
__pycache__/
*.py[cod]
.pytest_cache/
.coverage
.vscode/
```

Do not commit your local virtual environment or Python bytecode caches.

## 25.2 Branching strategy

Create a separate branch for each meaningful change:

```powershell
git checkout -b feature/improve-tokenizer
```

After implementing the change:

```powershell
git status
git add .
git commit -m "Improve text tokenization"
git push -u origin feature/improve-tokenizer
```

Review the changes before committing. Avoid adding generated temporary files or unrelated changes.

## 25.3 Contribution guidelines

Contributors should:

- Preserve the from-scratch implementation.
- Avoid introducing hidden data leakage.
- Add tests for new behavior.
- Update documentation when functionality changes.
- Record experimental parameters.
- Report actual measured results.
- Avoid claiming that a feature is implemented before verifying it.

---

# 26. Viva Preparation: Questions and Answers

## Q1. What is the purpose of this project?

The project automatically classifies SMS messages into spam and ham categories using a Multinomial Naive Bayes classifier implemented from scratch in Python.

## Q2. Why did you choose Naive Bayes?

Naive Bayes is relatively simple, computationally efficient, and well suited to word-count-based text classification. It also allows us to understand the mathematical principles behind the prediction process.

## Q3. What is Bayes' theorem?

Bayes' theorem updates the probability of a hypothesis after observing evidence. In this project, the hypothesis is that a message belongs to a particular class, and the evidence is the message's words.

## Q4. Why is it called Naive Bayes?

It uses Bayes' theorem with a simplifying assumption that features are conditionally independent given the class.

## Q5. Why Multinomial Naive Bayes?

Multinomial Naive Bayes works naturally with word-count features, where repeated occurrences of a word contribute repeatedly to the class score.

## Q6. What is the difference between spam and ham?

Spam is the unwanted-message category; ham is the legitimate-message category.

## Q7. What is a prior probability?

It is the estimated probability of a class before considering a particular message's words. In this implementation, it is calculated from class frequencies in the training set.

## Q8. What is likelihood?

Likelihood represents how probable the observed word evidence is under a particular class.

## Q9. Why is Laplace smoothing necessary?

It prevents words absent from a class's training messages from receiving zero probability.

## Q10. Why do you use logarithms?

Logarithms convert products of small probabilities into sums of log probabilities, preventing numerical underflow and simplifying computation.

## Q11. What is a vocabulary?

The vocabulary is the set of unique tokens learned from training messages. The evaluated model learned 7,720 tokens.

## Q12. Why split data into training and test sets?

Training data is used to learn model parameters. Test data is held out to estimate how well the trained model predicts unseen examples.

## Q13. Why use stratification?

Stratification preserves approximately the same class proportions in training and test sets.

## Q14. What is data leakage?

Data leakage occurs when information that would not be available during genuine prediction influences model fitting or evaluation. For example, learning the vocabulary from the entire dataset before splitting would expose the training pipeline to test-set information.

## Q15. Why remove duplicate rows?

Exact duplicates can give repeated records disproportionate influence and may inflate evaluation if copies appear in both training and test sets.

## Q16. What is accuracy?

Accuracy is the fraction of all predictions that are correct. It can be misleading when classes are imbalanced.

## Q17. What is precision?

Precision measures the fraction of predicted spam messages that were actually spam.

The project's spam precision is 89.29%.

## Q18. What is recall?

Recall measures the fraction of actual spam messages that were detected.

The project's spam recall is 95.42%.

## Q19. What is the F1-score?

It is the harmonic mean of precision and recall. The project's spam F1-score is 92.25%.

## Q20. Explain the confusion matrix.

The matrix contains 888 correctly predicted ham messages, 15 false positives, 6 false negatives, and 125 correctly predicted spam messages.

## Q21. Why is spam recall important?

Low recall means more spam messages go undetected. However, maximizing recall without considering precision can cause too many legitimate messages to be flagged.

## Q22. What is a false positive?

A legitimate message incorrectly classified as spam. The model produced 15 false positives in the recorded test run.

## Q23. What is a false negative?

A spam message incorrectly classified as ham. The model produced 6 false negatives in the recorded test run.

## Q24. What are the top spam-associated words?

In the recorded run, words such as `claim`, `prize`, `150p`, and `guaranteed` ranked highly according to a smoothed log-likelihood ratio.

## Q25. Does a word like `prize` always mean spam?

No. It is only a statistical association. The classifier considers the full set of tokens in a message, and its prediction can still be wrong.

## Q26. Why is the terminal application's vocabulary larger?

The training pipeline learns vocabulary from 4,135 training messages, while the terminal application trains on all 5,169 unique messages.

## Q27. Why not use a pretrained classifier?

The educational objective is to understand how Naive Bayes works internally. The classifier's training and prediction logic is therefore implemented manually.

## Q28. What is the role of scikit-learn?

It is used for stratified splitting and evaluation metrics, not for the primary Naive Bayes classifier.

## Q29. What are the main limitations?

The model ignores much of the semantic context, assumes conditional independence between words, and learns from a specific dataset. Its performance may differ on newer or substantially different messages.

## Q30. What would you improve next?

I would improve tokenization, investigate repeated messages across splits, integrate the vectorizer cleanly, add a web interface, compare against a standard baseline, and evaluate on additional datasets.

---

# 27. Presentation Guide

A concise project demonstration can follow this sequence:

1. Introduce the spam-detection problem.
2. Explain the dataset and the spam/ham labels.
3. Describe Bayes' theorem and Multinomial Naive Bayes.
4. Explain tokenization and word counts.
5. Demonstrate class priors, likelihoods, and Laplace smoothing.
6. Show the custom classifier code.
7. Run the training pipeline.
8. Present the confusion matrix and evaluation metrics.
9. Show the top spam-associated words.
10. Demonstrate the terminal interface.
11. Discuss limitations and future improvements.

## Suggested short explanation

"This project implements a Multinomial Naive Bayes SMS spam classifier from scratch in Python. It learns class-specific word frequencies and prior probabilities from labeled messages, applies Laplace smoothing to handle unseen words, and predicts classes using log-probability scores. I evaluated it on a held-out test set of 1,034 messages and obtained 97.97% accuracy and a spam F1-score of 92.25%. The project also includes word-association analysis, a confusion matrix, automated tests, and an interactive terminal application."

Only present the reported metrics as results from the recorded evaluation run.

---

# 28. Reference Materials

1. **UCI SMS Spam Collection:**  
   https://archive.ics.uci.edu/dataset/228/sms+spam+collection

2. **scikit-learn documentation — train/test splitting:**  
   https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

3. **scikit-learn documentation — classification metrics:**  
   https://scikit-learn.org/stable/modules/model_evaluation.html

4. **Python documentation:**  
   https://docs.python.org/3/

5. **Pandas documentation:**  
   https://pandas.pydata.org/docs/

6. **Matplotlib documentation:**  
   https://matplotlib.org/stable/

7. **Seaborn documentation:**  
   https://seaborn.pydata.org/

These references support further learning and verification of the libraries and dataset used by the project.

---

# 29. Final Verification Checklist

Before submitting or sharing the repository, verify each item.

## Environment and setup

- [ ] A teammate can clone the repository.
- [ ] The Python environment can be created from scratch.
- [ ] Dependencies install successfully.
- [ ] Dataset download instructions are documented.
- [ ] The expected dataset path is correct.

## Implementation

- [ ] `naive_bayes.py` contains the custom classifier.
- [ ] Training uses only the training split to learn word statistics.
- [ ] Laplace smoothing is applied.
- [ ] Log probabilities are used for scoring.
- [ ] The terminal application accepts user input.
- [ ] Empty input is handled appropriately.

## Evaluation

- [ ] The training/test split is reproducible.
- [ ] Evaluation metrics are generated from held-out predictions.
- [ ] Confusion matrix counts match the saved report.
- [ ] Reported metrics match the recorded run.
- [ ] Duplicate and near-duplicate leakage risks are understood.

## Documentation and collaboration

- [ ] `README.md` explains setup and basic usage.
- [ ] `PROJECT.md` explains the full system.
- [ ] `requirements.txt` contains the needed dependencies.
- [ ] `.gitignore` excludes local environment files.
- [ ] Tests pass on a fresh environment.
- [ ] Limitations and future work are documented.

---

# 30. Important Implementation Notes

This documentation describes the current prototype and distinguishes existing behavior from proposed improvements.

Before treating the repository as a finalized release, verify the following:

1. **Reproducibility:** Run `python train.py` and confirm that the printed metrics match the recorded values.
2. **Dependency hygiene:** Check `requirements.txt` and remove unnecessary development packages if appropriate.
3. **Data integrity:** Confirm the dataset version and document any changes.
4. **Evaluation integrity:** Investigate repeated message text across training and test sets.
5. **Vectorizer architecture:** Decide whether to integrate `vectorizer.py` into the main model pipeline or retain it as a standalone educational demonstration.
6. **Input validation:** Consider rejecting empty messages rather than returning a class based only on its prior.
7. **Repository hygiene:** Ensure `.venv/` and cache files are excluded from version control.
8. **Performance claims:** Treat the existing accuracy and F1-score as measured results for the recorded split, not guarantees for future datasets.

The goal of this project is not only to achieve a high evaluation score but also to build a classifier that can be understood, tested, reproduced, and explained.

