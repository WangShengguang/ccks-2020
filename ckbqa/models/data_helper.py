from typing import List

import torch
from pytorch_transformers import BertTokenizer

from ckbqa.utils.decorators import singleton
from ckbqa.utils.sequence import pad_sequences
from config import Config


@singleton
class DataHelper(object):
    def __init__(self, load_tokenizer=True):
        self.device = Config.device
        if load_tokenizer:
            self.load_tokenizer()

    def load_tokenizer(self):
        self.tokenizer = BertTokenizer.from_pretrained(
            Config.pretrained_model_name_or_path, do_lower_case=True)

    def batch_sent2tensor(self, sent_texts: List[str], pad=False):
        batch_token_ids = [self.sent2ids(sent_text) for sent_text in sent_texts]
        if pad:
            batch_token_ids = pad_sequences(batch_token_ids, maxlen=Config.max_len, padding='post')
        batch_token_ids = torch.tensor(batch_token_ids, dtype=torch.long).to(self.device)
        return batch_token_ids

    def sent2ids(self, sent_text):
        sent_tokens = ['[CLS]'] + self.tokenizer.tokenize(sent_text) + ["[SEP]"]
        token_ids = self.tokenizer.convert_tokens_to_ids(sent_tokens)
        return token_ids

    def data2tensor(self, batch_token_ids, pad=True, maxlen=Config.max_len):
        if pad:
            batch_token_ids = pad_sequences(batch_token_ids, maxlen=maxlen, padding='post')
        batch_token_ids = torch.tensor(batch_token_ids, dtype=torch.long).to(self.device)
        return batch_token_ids
