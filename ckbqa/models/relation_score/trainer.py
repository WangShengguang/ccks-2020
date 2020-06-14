import logging
import os
import random
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ckbqa.dataset.data_prepare import load_data, question_patten
from ckbqa.models.base_trainer import BaseTrainer
from ckbqa.models.data_helper import DataHelper
from ckbqa.models.relation_score.model import BertMatch, BertMatch2
from ckbqa.utils.saver import Saver
from ckbqa.utils.tools import json_load
from config import Config


class RelationScoreTrainer(BaseTrainer):
    """关系打分模块训练"""

    def __init__(self, model_name):
        super().__init__(model_name)
        self.model_name = model_name
        self.data_helper = DataHelper()
        self.saver = Saver(model_name=model_name)

    def data2samples(self, neg_rate=3, test_size=0.1):
        if os.path.isfile(Config.get_relation_score_sample_csv('train', neg_rate)):
            return
        questions = []
        sim_questions = []
        labels = []
        all_relations = list(json_load(Config.relation2id))
        _entity_pattern = re.compile(r'["<](.*?)[>"]')
        for q, sparql, a in load_data(tqdm_prefix='data2samples '):
            q_text = question_patten.findall(q)[0]
            q_entities = _entity_pattern.findall(sparql)
            questions.append(q_text)
            sim_questions.append('的'.join(q_entities))
            labels.append(1)
            #
            for neg_relation in random.sample(all_relations, neg_rate):
                questions.append(q_text)
                neg_question = '的'.join(q_entities[:-1] + [neg_relation])  # 随机替换 <关系>
                sim_questions.append(neg_question)
                labels.append(0)
        data_df = pd.DataFrame({'question': questions, 'sim_question': sim_questions, 'label': labels})
        data_df.to_csv(Config.relation_score_sample_csv, encoding='utf_8_sig', index=False)
        train_df, test_df = train_test_split(data_df, test_size=test_size)
        test_df.to_csv(Config.get_relation_score_sample_csv('test', neg_rate), encoding='utf_8_sig', index=False)
        train_df.to_csv(Config.get_relation_score_sample_csv('train', neg_rate), encoding='utf_8_sig', index=False)

    def batch_iter(self, data_type, batch_size, _shuffle=True, fixed_seq_len=None):
        file_path = Config.get_relation_score_sample_csv(data_type=data_type, neg_rate=3)
        logging.info(f'* load data from {file_path}')
        data_df = pd.read_csv(file_path)
        x_data = data_df['question'].apply(self.data_helper.sent2ids)
        y_data = data_df['sim_question'].apply(self.data_helper.sent2ids)
        labels = data_df['label']
        data_size = len(x_data)
        order = list(range(data_size))
        if _shuffle:
            np.random.shuffle(order)
        for batch_step in range(data_size // batch_size + 1):
            batch_idxs = order[batch_step * batch_size:(batch_step + 1) * batch_size]
            if len(batch_idxs) != batch_size:  # batch size 不可过大; 不足batch_size的数据丢弃（最后一batch）
                continue
            q_sents = self.data_helper.data2tensor([x_data[idx] for idx in batch_idxs])
            a_sents = self.data_helper.data2tensor([y_data[idx] for idx in batch_idxs])
            batch_labels = self.data_helper.data2tensor([labels[idx] for idx in batch_idxs], pad=False)
            yield q_sents, a_sents, batch_labels

    def train_match_model(self, mode='train'):
        """
        :mode: train,test
        """
        self.data2samples(neg_rate=3, test_size=0.1)
        if self.model_name == 'bert_match':  # 单层
            model = BertMatch()
        elif self.model_name == 'bert_match2':  # 双层
            model = BertMatch2()
        else:
            raise ValueError()
        for para in model.bert.parameters():
            para.requires_grad = False  # bert参数不训练
        model = self.init_model(model)
        model_path, epoch, step = self.saver.load_model(model, fail_ok=True)
        self.global_step = step
        for q_sents, a_sents, batch_labels in self.batch_iter(
                data_type='train', batch_size=32):
            pred, loss = model(q_sents, a_sents, batch_labels)
            # print(f' {str(datetime.datetime.now())[:19]} global_step: {self.global_step}, loss:{loss.item():.04f}')
            logging.info(f' global_step: {self.global_step}, loss:{loss.item():.04f}')
            self.backfoward(loss, model)
            if self.global_step % 100 == 0:
                model_path = self.saver.save(model, epoch=1, step=self.global_step)
                logging.info(f'save to {model_path}')
            # if mode == 'test':
            # output = torch.softmax(pred, dim=-1)
            # pred = torch.argmax(output, dim=-1)
            # print('labels: ', batch_labels)
            # print('pred: ', pred.tolist())
            # print('--' * 10 + '\n\n')
