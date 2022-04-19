from transformers import AutoTokenizer

from datasets import load_dataset
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
import numpy as np
from datasets import load_metric
from transformers import TrainingArguments
import torch.nn as nn
import torch


from transformers import BertModel


class CustomBERTModel(nn.Module):
    def __init__(self):
          super(CustomBERTModel, self).__init__()
          self.bert = BertModel.from_pretrained("bert-base-cased")       # try distiled
          ### New layers:
          self.linear1 = nn.Linear(768, 256)
          self.linear2 = nn.Linear(256, 3) ## 3 is the number of classes in this example
          self.relu = torch.nn.ReLU()
          self.dropout = torch.nn.Dropout(0.1)
          self.batchnorm = torch.nn.BatchNorm1d(256)

    def forward(self, input_ids, attention_mask, token_type_ids, labels):
          sequence_output, pooled_output  = self.bert(
               input_ids, 
               attention_mask=attention_mask,
               return_dict=False)
          # print(sequence_output[0].shape)
          # sequence_output has the following shape: (batch_size, sequence_length, 768)
          linear1_output = self.linear1(self.relu(sequence_output[:,0,:].view(-1,768))) ## extract the 1st token's embeddings

          linear2_output = self.linear2(self.batchnorm(self.dropout(self.relu(linear1_output))))

          return linear2_output



class TransformerDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)




