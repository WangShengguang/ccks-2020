"""
主实体关联的关系打分，得到topn
"""
import torch
from pytorch_transformers import BertModel
from torch import nn

from config import Config


class BertMatch2(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(Config.pretrained_model_name_or_path)
        self.soft_max = nn.Softmax(dim=-1)
        self.fc = nn.Linear(768 * 3, 2)
        self.dropout = nn.Dropout(0.8)
        self.loss = nn.CrossEntropyLoss()

    def encode(self, x):
        outputs = self.bert(x)
        pooled_out = outputs[1]
        return pooled_out

    def forward(self, input1_ids, input2_ids, labels=None):
        # shape: (batch_size, seq_length, hidden_size)
        seq1_output = self.encode(input1_ids)
        seq2_output = self.encode(input2_ids)
        feature = torch.cat([seq1_output, seq2_output, seq1_output - seq2_output], dim=-1)
        logistic = self.fc(feature)
        if labels is None:
            output = self.soft_max(logistic)
            pred = torch.argmax(output, dim=-1)
            return pred
        else:
            loss = self.loss(logistic, labels)
            return logistic, loss


class BertMatch(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(Config.pretrained_model_name_or_path)
        self.soft_max = nn.Softmax(dim=-1)
        self.fc1 = nn.Linear(768 * 3, 128)
        self.fc2 = nn.Linear(128, 2)
        self.dropout1 = nn.Dropout(0.8)
        self.dropout2 = nn.Dropout(0.8)
        self.loss = nn.CrossEntropyLoss()

    def encode(self, x):
        outputs = self.bert(x)
        pooled_out = outputs[1]
        return pooled_out

    def forward(self, input1_ids, input2_ids, labels=None):
        # shape: (batch_size, seq_length, hidden_size)
        seq1_output = self.encode(input1_ids)
        seq2_output = self.encode(input2_ids)
        feature = torch.cat([seq1_output, seq2_output, seq1_output - seq2_output], dim=-1)
        feature1 = self.fc1(feature)
        logistic = self.fc2(feature1)
        if labels is None:
            output = self.soft_max(logistic)
            pred = torch.argmax(output, dim=-1)
            return pred
        else:
            loss = self.loss(logistic, labels)
            return logistic, loss
