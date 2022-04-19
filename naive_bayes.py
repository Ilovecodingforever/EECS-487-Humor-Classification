import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


class NaiveBayes:
    """Naive Bayes classifier."""

    def __init__(self):
        super().__init__()
        self.ngram_count = []
        self.total_count = []
        self.category_prob = []

    def fit(self, data):
        # store ngram counts for each category in self.ngram_count
        self.vectorizer = CountVectorizer(ngram_range=(1, 2), max_df=0.8, min_df=3)
        self.vectorizer.fit(data.text)
        for label_i in range(len(pd.unique(data.is_humor))):
            data_i = data[data.is_humor == label_i]
            raw_counts = self.vectorizer.transform(data_i.text)
            self.ngram_count.append(np.asarray(raw_counts.sum(axis=0)))
            self.total_count.append(raw_counts.sum())
            self.category_prob.append(len(data_i) / len(data))

    def calculate_prob(self, docs, c_i, alpha):
        prob = None
        # calculate probability of category c_i given each article in docs
        raw_counts = self.vectorizer.transform(docs)
        ngram_prob = (self.ngram_count[c_i] + alpha) / (self.total_count[c_i] +
                                                        alpha * len(self.vectorizer.vocabulary_))
        prob = raw_counts.multiply(np.log(ngram_prob)).toarray()
        prob = prob.sum(axis=1) + np.log(self.category_prob[c_i])

        return prob

    def predict(self, docs, alpha):
        prediction = [None] * len(docs)
        # predict categories for the docs
        probs = [self.calculate_prob(docs, ci, alpha)[:, np.newaxis] for ci in range(len(self.category_prob))]
        probs = np.hstack(probs)
        prediction = np.argmax(probs, axis=1)

        return prediction


def evaluate(predictions, labels):
    accuracy, mac_f1, mic_f1 = None, None, None

    predictions, labels = np.array(predictions), np.array(labels)
    num_category = labels.max() + 1
    accuracy = (predictions == labels).sum() / len(labels)
    # get confusion matrix
    confusion_matrix = np.zeros((num_category, num_category))
    for true_ in range(num_category):
        for pred_ in range(num_category):
            confusion_matrix[true_, pred_] = ((labels == true_) & (predictions == pred_)).sum()
    # calculate macro f1
    f1s = []
    for ci in range(num_category):
        precision = confusion_matrix[ci, ci] / confusion_matrix[:, ci].sum()
        recall = confusion_matrix[ci, ci] / confusion_matrix[ci].sum()
        f1s.append(2 * precision * recall / (precision + recall))
    mac_f1 = sum(f1s) / len(f1s)
    # calculate micro f1
    mic_matrix = np.zeros((2, 2))  # [[tp, fn], [fp, tn]]
    for ci in range(num_category):
        mic_matrix[0, 0] += confusion_matrix[ci, ci]
        mic_matrix[0, 1] += confusion_matrix[ci].sum() - confusion_matrix[ci, ci]
        mic_matrix[1, 0] += confusion_matrix[:, ci].sum() - confusion_matrix[ci, ci]
        mic_matrix[1, 1] += confusion_matrix.sum() - confusion_matrix[ci].sum() \
                            - confusion_matrix[:, ci].sum() + confusion_matrix[ci, ci]
    precision = mic_matrix[0, 0] / (mic_matrix[:, 0].sum())
    recall = mic_matrix[0, 0] / (mic_matrix[0].sum())
    mic_f1 = 2 * precision * recall / (precision + recall)

    return accuracy, mac_f1, mic_f1
