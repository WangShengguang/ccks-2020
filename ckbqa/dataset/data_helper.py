import logging

import numpy as np
import pandas as pd
import torch
from pytorch_transformers import BertTokenizer
from typing import List

from ckbqa.utils.sequence import pad_sequences
from ckbqa.utils.tools import singleton
from config import Config


@singleton
class DataHelper(object):
    def __init__(self, load_tokenizer=False):
        self.config = Config()
        if load_tokenizer:
            self.load_tokenizer()

    def load_tokenizer(self):
        self.tokenizer = BertTokenizer.from_pretrained(
            self.config.pretrained_model_name_or_path, do_lower_case=True)

    def batch_sent2tensor(self, sent_texts: List[str], pad=False):
        batch_token_ids = [self.__sent2ids(sent_text) for sent_text in sent_texts]
        if pad:
            batch_token_ids = pad_sequences(batch_token_ids, maxlen=Config.max_len, padding='post')
        batch_token_ids = torch.tensor(batch_token_ids, dtype=torch.long).to(Config.device)
        return batch_token_ids

    def sent2ids(self, sent_text):
        sent_tokens = ['[CLS]'] + self.tokenizer.tokenize(sent_text) + ["[SEP]"]
        token_ids = self.tokenizer.convert_tokens_to_ids(sent_tokens)
        return token_ids

    def data2tensor(self, batch_token_ids, pad=True):
        if pad:
            batch_token_ids = pad_sequences(batch_token_ids, maxlen=self.config.max_len, padding='post')
        batch_token_ids = torch.tensor(batch_token_ids, dtype=torch.long).to(Config.device)
        return batch_token_ids

    def batch_iter(self, data_type, batch_size, _shuffle=True, fixed_seq_len=None):
        self.load_tokenizer()
        file_path = Config.get_sample_csv_path(data_type=data_type, neg_rate=3)
        logging.info(f'* load data from {file_path}')
        data_df = pd.read_csv(file_path)
        x_data = data_df['question'].apply(self.sent2ids)
        y_data = data_df['sim_question'].apply(self.sent2ids)
        labels = data_df['label']
        data_size = len(x_data)
        order = list(range(data_size))
        if _shuffle:
            np.random.shuffle(order)
        for batch_step in range(data_size // batch_size + 1):
            batch_idxs = order[batch_step * batch_size:(batch_step + 1) * batch_size]
            if len(batch_idxs) != batch_size:  # batch size 不可过大; 不足batch_size的数据丢弃（最后一batch）
                continue
            q_sents = self.data2tensor([x_data[idx] for idx in batch_idxs])
            a_sents = self.data2tensor([y_data[idx] for idx in batch_idxs])
            batch_labels = self.data2tensor([labels[idx] for idx in batch_idxs], pad=False)
            yield q_sents, a_sents, batch_labels
