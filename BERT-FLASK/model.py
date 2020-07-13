import config
import transformers
import torch.nn as nn
from transformers import BertModel
import torch

# Code block to avoid repeated truncation warning
import logging
logging.basicConfig(level=logging.ERROR)



class SentimentClassifier(nn.Module):
  def __init__(self, n_classes):
    super(SentimentClassifier, self).__init__()
    self.bert = BertModel.from_pretrained(config.PRE_TRAINED_MODEL_NAME)
    self.drop = nn.Dropout(p=0.3)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
    self.softmax = nn.Softmax(dim=1)
  def forward(self, input_ids, attention_mask):
    _, pooled_output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    output = self.drop(pooled_output)
    output = self.out(output)
    return self.softmax(output)


model = SentimentClassifier(len(config.class_names))
model.load_state_dict(torch.load(config.MODEL_PATH))
model.to(config.device)