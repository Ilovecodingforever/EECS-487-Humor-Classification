import torch
from torch import nn
import torch.nn.functional as F
from typing import List


class Model(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size=300, hidden_size=hidden_dim, batch_first=True, dropout=0.5, num_layers=2, bidirectional=True)
        self.dense = nn.Linear(self.hidden_dim * 2, self.hidden_dim, device=device)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(self.hidden_dim, 3, device=device)
        self.dropout = nn.Dropout(0.1)
        self.init_weights()

    def init_weights(self):
        pass

    def forward(self, sentences: List[torch.Tensor]):
        """
        return:
        output: N x 3 tensor
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        output = None

        # c_inp = []
        # each_c_len = []
        # for i in range(len(sentences)):
        #   c_inp = c_inp + sentences[i]
        #   each_c_len = [len(sentences[i])] + each_c_len
        c_inp_len = [i.shape[0] for i in sentences]
        c_inp = nn.utils.rnn.pad_sequence(sentences, batch_first=True).to(device=device)
        c_inp = nn.utils.rnn.pack_padded_sequence(c_inp, c_inp_len, batch_first=True, enforce_sorted=False).to(device=device)
        out, (h_t, c_t) = self.lstm(c_inp)
        c_output, c_output_len = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        last_seq_idxs = torch.tensor([x - 1 for x in c_output_len], device=device, dtype=torch.int64)
        last_seq_items = c_output[range(c_output.shape[0]), last_seq_idxs, :]
        out = (last_seq_items)

        output = self.dense2(self.dropout(self.relu(self.dense(out))))

        return output
