from transformers import AutoTokenizer

from datasets import load_dataset
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
import numpy as np
from datasets import load_metric
from transformers import TrainingArguments
import torch.nn as nn
import torch


# class Transformer(torch.nn.Module):
#     def __init__(self):
#           super(Transformer, self).__init__()
#           self.bert = AutoModelForSequenceClassification.from_pretrained("bert-base-cased")
#           ### New layers:
#           self.linear1 = torch.nn.Linear(768, 256)
#           self.linear2 = torch.nn.Linear(256, 3) ## 3 is the number of classes in this example
#           self.relu = torch.nn.ReLU()

#     def forward(self, input_ids, attention_mask, token_type_ids, labels):
#           sequence_output = self.bert(
#                input_ids, 
#                attention_mask=attention_mask)

#           # sequence_output has the following shape: (batch_size, sequence_length, 768)
#           linear1_output = self.linear1(sequence_output[:,0,:].view(-1,768)) ## extract the 1st token's embeddings

#           linear2_output = self.linear2(self.relu(linear1_output))

#           return linear2_output




from transformers import BertModel
class CustomBERTModel(nn.Module):
    def __init__(self):
          super(CustomBERTModel, self).__init__()
          self.bert = BertModel.from_pretrained("bert-base-cased")       # try distiled
          ### New layers:
          self.linear1 = nn.Linear(768, 3)
          self.linear2 = nn.Linear(256, 3) ## 3 is the number of classes in this example
          self.relu = torch.nn.ReLU()
          self.dropout = torch.nn.Dropout(0.1)
          self.batchnorm = torch.nn.BatchNorm1d(256)

    def forward(self, input_ids, attention_mask):
          sequence_output, pooled_output  = self.bert(
               input_ids, 
               attention_mask=attention_mask,
               return_dict=False)
          # print(sequence_output[0].shape)
          # sequence_output has the following shape: (batch_size, sequence_length, 768)
          linear1_output = self.linear1(self.relu(sequence_output[:,0,:].view(-1,768))) ## extract the 1st token's embeddings

          # linear2_output = self.linear2(self.batchnorm(self.dropout(self.relu(linear1_output))))

          return linear1_output


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





# def tokenize_function(examples):
#     return tokenizer(examples["text"], padding="max_length", truncation=True)

# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)

# # dataset = load_dataset("yelp_review_full")
# # dataset[100]


# def train(dataset):


#   tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

#   tokenized_datasets = dataset.map(tokenize_function, batched=True)


#   small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
#   small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))


#   model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)



#   training_args = TrainingArguments(output_dir="test_trainer")


#   metric = load_metric("accuracy")



#   training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")




#   trainer = Trainer(
#       model=model,
#       args=training_args,
#       train_dataset=small_train_dataset,
#       eval_dataset=small_eval_dataset,
#       compute_metrics=compute_metrics,
#   )



#   trainer.train()


