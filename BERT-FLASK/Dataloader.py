from torch.utils.data import Dataset, DataLoader
import config

class UnseenTweetReviewDataset(Dataset):
  def __init__(self, reviews):
    self.reviews = reviews
  def __len__(self):
    return len(self.reviews)
  def __getitem__(self, item):
    review = str(self.reviews[item])
    encoding = config.tokenizer.encode_plus(
      review,
      add_special_tokens=True,
      max_length=config.MAX_LEN,
      return_token_type_ids=False,
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt',
    )
    return {
      'review_text': review,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten()
    }