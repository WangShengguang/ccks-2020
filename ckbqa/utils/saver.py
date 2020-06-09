import logging
import os

import torch

from config import Config, ckpt_dir


class Saver(object):
    def __init__(self, model_name):
        # self.dataset = dataset
        self.model_name = model_name
        self.model_dir = os.path.join(ckpt_dir, model_name)

    def load_model(self, model: torch.nn.Module, mode="max_step", fail_ok=False, map_location=Config.device):
        model_path = os.path.join(self.model_dir, mode, f"{self.model_name}.bin")
        if os.path.isfile(model_path):
            ckpt = torch.load(model_path, map_location=map_location)
            model.load_state_dict(ckpt["net"])
            step = ckpt["step"]
            epoch = ckpt["epoch"]
            logging.info("* load model from file: {}，epoch：{}， step：{}".format(model_path, epoch, step))
        else:
            if fail_ok:
                epoch = 0
                step = 0
            else:
                raise ValueError(f'model path : {model_path} is not exist')
            logging.info("* Fail load model from file: {}".format(model_path))
        return model_path, epoch, step

    def save(self, model, epoch, step=-1, mode="max_step", parms_dic: dict = None):
        model_path = os.path.join(self.model_dir, mode, f"{self.model_name}.bin")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        state = {"net": model.state_dict(), 'epoch': epoch, "step": step}
        if isinstance(parms_dic, dict):
            state.update(parms_dic)
        torch.save(state, model_path)
        return model_path
