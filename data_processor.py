from typing import List

import pandas as pd
from torch.utils.data import Dataset
from gensim.models import KeyedVectors
import gensim.downloader
import torch
from nltk.tokenize import word_tokenize
from os.path import exists
from torch.utils.data import DataLoader


def load_data(filename):
    """reads csv and return lists"""
    df = pd.read_csv(filename, header=0,
                     names=("id", "text", "is_humor", "humor_rating", "humor_controversy", "offense_rating"))
    return df["text"].tolist(), df["is_humor"].tolist()


def load_data_nb(filename):
    """same function, but for naive bayes, returns df"""
    df = pd.read_csv(filename, header=0,
                     names=("id", "text", "is_humor", "humor_rating", "humor_controversy", "offense_rating"))
    return df[['text', 'is_humor']]


class HumorDataset(Dataset):
    """
    return:
        sentences: list of sentence embeddings, N x * x 300
        label: tensor of one-hot encodings, N x #classes
    """

    def __init__(self, data_text: List, data_label: List):

        if not exists('vectors.kv'):
            self.embed = gensim.downloader.load('glove-wiki-gigaword-300')
            word_vectors = self.embed
            word_vectors.save('vectors.kv')
        else:
            self.embed = KeyedVectors.load('vectors.kv')

        self.data_text = data_text
        self.data_label = data_label
        self.sentences = []
        self.label = None

        for l in data_label:
            temp = torch.zeros(3, dtype=torch.int64)
            temp[int(l)] = 1
            if self.label is None:
                self.label = temp
            else:
                self.label = torch.cat((self.label, temp))

        for t in data_text:
            w = word_tokenize(t.lower())
            lst = []
            for x in w:
                if x in self.embed:
                    lst.append(torch.tensor(self.embed[x]))
                else:
                    lst.append(torch.tensor(self.embed['unk']))
            self.sentences.append(torch.vstack(lst))

    def __len__(self):
        return len(self.data_text)

    def __getitem__(self, idx):
        return {"sentences": self.sentences[idx], "labels": self.label[idx]}


def basic_collate_fn(batch):
    """Collate function for basic setting."""
    sentences = [i['sentences'] for i in batch]
    labels = [i['labels'] for i in batch]
    return sentences, labels


if __name__ == "__main__":
    x, y = load_data("train.csv")
    dataset = HumorDataset(x, y)
    train_loader = DataLoader(dataset, batch_size=2, collate_fn=basic_collate_fn, shuffle=False)
    sentence, labels = next(iter(train_loader))
    pass
