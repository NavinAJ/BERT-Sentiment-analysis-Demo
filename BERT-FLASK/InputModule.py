import config
import model as md
import TextPreprocessing as tp
import torch

import Predictions as pred
import Dataloader as dl


import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader

def GetSentimentForText(_text):
    review = str(_text)
    review = tp.expand_contractions(review)
    review = tp.scrub_words(review)
    review = tp.remove_accented_chars(review)

    encoding = config.tokenizer.encode_plus(
    review,
    max_length=config.MAX_LEN,
    add_special_tokens=True, # Add '[CLS]' and '[SEP]'
    return_token_type_ids=False,
    pad_to_max_length=True,
    return_attention_mask=True,
    return_tensors='pt',  # Return PyTorch tensors
    )

    input_ids = encoding['input_ids'].to(config.device)
    attention_mask = encoding['attention_mask'].to(config.device)
    output = md.model(input_ids, attention_mask)
    _, prediction = torch.max(output, dim=1)
    print(f'Review text: {review}')
    print(f'Sentiment  : {config.class_names[prediction]}')
    return config.class_names[prediction]

def Unseen_Dataloader(df):
  ds = dl.UnseenTweetReviewDataset(
    reviews=df.text.to_numpy()
  )
  return DataLoader(
    ds
  )

def GetSentimentForDF(_df):
    _df['text'] = _df.text.apply(tp.expand_contractions)
    _df['text'] = _df.text.apply(tp.scrub_words)
    _df['text'] = _df.text.apply(tp.remove_accented_chars)
    print(_df)
    review_data_loader = Unseen_Dataloader(_df)
    data = next(iter(review_data_loader))
    print(data.keys())
    unsen_review_texts, unseen_pred, unseen_pred_probs = pred.Get_unseen_Prediction(md.model,review_data_loader)
    return unseen_pred


def ReadText(_text):
  review = _text
  sentiment = GetSentimentForText(review)
  return sentiment

def ReadFile(_file,_filename):
  input_df = pd.read_csv(_file)
  review_df = pd.DataFrame()
  review_df = input_df[['text']]
  sentiments = GetSentimentForDF(review_df)
  review_df['BERT Sentiment Prediction'] = [config.class_names[i] for i in sentiments]
  return review_df




