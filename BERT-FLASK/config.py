import transformers
import torch

MAX_LEN = 150
MODEL_PATH = "..\Bert_Sentiment_Model.pt"
PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
tokenizer = transformers.BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
class_names = ['negative', 'neutral', 'positive']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Batch_size = 16