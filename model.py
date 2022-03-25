import torch
from torch import nn
import torch.nn.functional as F
from typing import List


class Model(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.hidden_dim = hidden_dim

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

        return output
