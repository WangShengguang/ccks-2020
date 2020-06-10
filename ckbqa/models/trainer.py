import logging

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from ckbqa.dataset.data_helper import DataHelper
from ckbqa.models.relation_score import BertMatch, BertMatch2
from ckbqa.utils.saver import Saver
from config import Config


class BaseTrainer(object):
    def __init__(self, model_name):
        self.global_step = 0
        self.device = Config.device
        self.model_name = model_name

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

    def init_model(self, model):
        model.to(self.device)  # without this there is no error, but it runs in CPU (instead of GPU).
        if Config.gpu_nums > 1 and Config.multi_gpu:
            model = torch.nn.DataParallel(model)
        self.optimizer = Adam(model.parameters(), lr=Config.learning_rate)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1 / (1 + 0.05 * epoch))
        return model


class Trainer(BaseTrainer):
    def __init__(self, model_name):
        super().__init__(model_name)
        self.model_name = model_name
        self.data_helper = DataHelper()
        self.saver = Saver(model_name=model_name)

    def train_match_model(self, mode='train'):
        """
        :mode: train,test
        """
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
        for q_sents, a_sents, batch_labels in self.data_helper.batch_iter(
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
