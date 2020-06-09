import logging

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from ckbqa.dataset.data_helper import DataHelper
from ckbqa.models.bert_match import BertMatch, BertMatch2
from ckbqa.utils.saver import Saver
from config import Config


class BaseTrainer(object):
    def __init__(self):
        self.global_step = 0
        self.device = Config.device

    def init_model(self, model):
        model.to(self.device)  # without this there is no error, but it runs in CPU (instead of GPU).
        if Config.gpu_nums > 1 and Config.multi_gpu:
            model = torch.nn.DataParallel(model)
        self.optimizer = Adam(model.parameters(), lr=Config.learning_rate)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1 / (1 + 0.05 * epoch))
        return model

    def backfoward(self, loss, model):
        if Config.gpu_nums > 1 and Config.multi_gpu:
            loss = loss.mean()  # mean() to average on multi-gpu
        if Config.gradient_accumulation_steps > 1:
            loss = loss / Config.gradient_accumulation_steps
        # https://zhuanlan.zhihu.com/p/79887894
        loss.backward()
        self.global_step += 1
        if self.global_step % Config.gradient_accumulation_steps == 0:
            nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=Config.clip_grad)
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss


class BertmatchTrainer(BaseTrainer):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.data_helper = DataHelper()
        self.saver = Saver(model_name=model_name)

    def get_model(self):
        if self.model_name == 'bert_match':  # 20200607双层
            model = BertMatch()
            # model.bert
        elif self.model_name == 'bert_match2':  # 20200607单层
            model = BertMatch2()
        else:
            raise ValueError()
        return model

    def train(self):
        model = self.get_model()
        model = self.init_model(model)
        for para in model.bert.parameters():
            para.requires_grad = False  # bert参数不训练
        model_path, epoch, step = self.saver.load_model(model, fail_ok=True)
        self.global_step = step
        for q_sents, a_sents, batch_labels in self.data_helper.batch_iter(
                data_type='train', batch_size=32):
            pred, loss = model(q_sents, a_sents, batch_labels)
            # print(f' {str(datetime.datetime.now())[:19]} global_step: {self.global_step}, loss:{loss.item():.04f}')
            logging.info(f' global_step: {self.global_step}, loss:{loss.item():.04f}')
            self.backfoward(loss, model)
            if self.global_step % 100 == 0:
                model_path = self.saver.save(model, epoch=1, step=self.global_step)
                logging.info(f'save to {model_path}')
            # print('labels: ', batch_labels)
            # print('pred: ', pred)

    # def _predict(self, q_sents, a_sents, batch_labels):

    def predict(self):
        model = self.get_model()
        model_path, epoch, step = self.saver.load_model(model, fail_ok=False)
        logging.info(f'* load model from {model_path}')
        labels = []
        preds = []
        for q_sents, a_sents, batch_labels in self.data_helper.batch_iter(
                data_type='test', batch_size=256):
            distances = model(q_sents, a_sents)
            labels.extend(batch_labels.tolist())
            preds.extend(distances.tolist())
            print('aaa')
        test_df = pd.read_csv(Config.get_sample_csv_path('test', neg_rate=3), nrows=len(preds))
        test_df['pred'] = preds
        test_df.to_csv('./pred.csv', encoding='utf_8_sig', index=False)
