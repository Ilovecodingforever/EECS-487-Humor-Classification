import pickle as pkl
import pandas as pd
# import pickle
import base64
import csv

# with open('mnist.pkl', 'rb') as f:
#     u = pickle._Unpickler(f)
#     u.encoding = 'latin1'
#     p = u.load()
#     print(p)


if __name__ == "__main__":
    with open("wiki_sentences.pickle", "rb") as f:
        u = pkl._Unpickler(f)
        u.encoding = 'latin1'
        object = u.load()

    df = pd.DataFrame(object)
    df.to_csv(r'reuters.csv')




    #
    # your_pickle_obj = pickle.loads(open('wiki_sentences.pickle', 'rb').read())
    # with open('output.csv', 'a', encoding='utf8') as csv_file:
    #     wr = csv.writer(csv_file, delimiter='|')
    #     pickle_bytes = pickle.dumps(your_pickle_obj)  # unsafe to write
    #     b64_bytes = base64.b64encode(pickle_bytes)  # safe to write but still bytes
    #     b64_str = b64_bytes.decode('utf8')  # safe and in utf8
    #     wr.writerow(['col1', 'col2', b64_str])