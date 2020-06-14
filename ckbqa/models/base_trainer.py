import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

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
