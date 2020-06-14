"""BERT + CRF for named entity recognition"""

import torch.nn as nn
from pytorch_transformers import BertPreTrainedModel, BertModel

from .crf import CRF


class BERTCRF(BertPreTrainedModel):
    """BERT model for token-level classification.
    This module is composed of the BERT model with a linear layer on top of
    the full hidden state of the last layer.

    Args:
        config: a BertConfig class instance with the configuration to build a new model.
        num_labels: the number of classes for the classifier. Default = 2.

    Inputs:
        input_ids: a torch.LongTensor of shape (batch_size, seq_length)
            with the word token indices in the vocabulary.
        token_type_ids: an optional torch.LongTensor of shape (batch_size, seq_length) with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token.
        attention_mask: an optional torch.LongTensor of shape (batch_size, seq_length) with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        labels: labels for the classification output: torch.LongTensor of shape (batch_size, seq_length)
            with indices selected in [0, ..., num_labels].

    Returns:
        if labels is not None:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if labels is None:
            Outputs the classification tag indices of shape (batch_size, seq_length).
    """

    def __init__(self, config, num_labels):
        super(BERTCRF, self).__init__(config)
        self.num_labels = num_labels  # ent label num
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.fc = nn.Linear(config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        self.apply(self.init_bert_weights)
        # self.fc.weight.data.normal_(mean=0.0, std=0.02)
        # self.fc.bias.data.zero_()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        # shape: (batch_size, seq_length, hidden_size)
        outputs = self.bert(input_ids, token_type_ids, attention_mask)
        seq_output = outputs[0]
        # shape: (batch_size, seq_length, hidden_size)
        seq_output = self.dropout(seq_output)
        # shape: (batch_size, seq_length, num_labels)
        fc_output = self.fc(seq_output)

        if attention_mask is None:
            pred = self.crf.decode(fc_output)
        else:
            pred = self.crf.decode(fc_output, attention_mask)

        if labels is None:
            return pred
        else:
            if attention_mask is not None:
                # shape: (1,)
                loss = -self.crf(fc_output, labels, attention_mask, reduction='token_mean')
            else:
                loss = -self.crf(fc_output, labels, reduction='token_mean')
            return pred, loss
