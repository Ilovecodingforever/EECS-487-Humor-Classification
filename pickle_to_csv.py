import pickle as pkl
import pandas as pd
# import pickle
import base64
import csv

# change file names to convert for wiki and reuters

if __name__ == "__main__":
    with open("wiki_sentences.pickle", "rb") as f:
        u = pkl._Unpickler(f)
        u.encoding = 'latin1'
        object = u.load()

    df = pd.DataFrame(object)
    df.to_csv(r'wiki.csv')
