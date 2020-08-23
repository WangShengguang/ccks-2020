"""现有模型封装；提供给预测使用"""
import logging
from typing import List

from ckbqa.models.data_helper import DataHelper
from ckbqa.models.relation_score.model import BertMatch, BertMatch2
from ckbqa.utils.saver import Saver


class RelationScorePredictor(object):
    def __init__(self, model_name):
        logging.info(f'BertMatchPredictor loaded sim_model init ...')
        self.data_helper = DataHelper(load_tokenizer=True)
        self.model = self.load_sim_model(model_name)
        logging.info(f'loaded sim_model: {model_name}')

    def load_sim_model(self, model_name: str):
        """文本相似度模型"""
        assert model_name in ['bert_match', 'bert_match2']
        saver = Saver(model_name=model_name)
        if model_name == 'bert_match':
            model = BertMatch()
        elif model_name == 'bert_match2':
            model = BertMatch2()
        else:
            raise ValueError()
        saver.load_model(model)
        return model

    def iter_sample(self, q_text, sim_texts):
        batch_size = 32
        q_ids = self.data_helper.sent2ids(q_text)
        batch_q_sent_token_ids = [q_ids for _ in range(min(batch_size, len(sim_texts)))]
        q_tensors = self.data_helper.data2tensor(batch_q_sent_token_ids)
        batch_sim_texts = [sim_texts[i:i + batch_size] for i in range(0, len(sim_texts), batch_size)]
        for batch_sim_text in batch_sim_texts:
            sim_sent_token_ids = [self.data_helper.sent2ids(sim_text) for sim_text in batch_sim_text]
            sim_tensors = self.data_helper.data2tensor(sim_sent_token_ids)
            yield q_tensors[:len(sim_tensors)], sim_tensors

    def predict(self, q_text: str, sim_texts: List[str]):
        all_preds = []
        for q_tensors, sim_tensors in self.iter_sample(q_text, sim_texts):
            preds = self.model(q_tensors, sim_tensors)
            preds = preds[:, 1]
            all_preds.extend(preds.tolist())
        return all_preds
