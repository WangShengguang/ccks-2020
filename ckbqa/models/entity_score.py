'''
从所有候选实体中筛选出topn个候选实体
'''

import re
import os
from sklearn import linear_model
import numpy as np
from ckbqa.dataset.data_prepare import question_patten, entity_patten, load_data
from ckbqa.qa.recognizer import Recognizer
from ckbqa.utils.tools import pkl_dump, pkl_load
from config import Config

ent_attr_partten = re.compile('["<](.*?)[>"]')


class EntityScore(object):
    def __init__(self, load_model=False):
        if load_model:
            self.model = pkl_load(Config.entity_score_model_pkl)

    def gen_train_data(self):
        X_train = []
        Y_label = []
        recognizer = Recognizer()
        for q, sparql, a in load_data(tqdm_prefix='EntityScore train data '):
            a_entities = entity_patten.findall(a)
            q_entities = ent_attr_partten.findall(sparql)  # attr
            q_text = question_patten.findall(q)[0]
            candidate_entities = recognizer.get_candidate_entities(q_text)
            for ent_name, feature_dict in candidate_entities.items():
                feature = feature_dict['feature']
                label = 1 if ent_name in q_entities else 0  # 候选实体有的不在答案中
                X_train.append(feature)
                Y_label.append(label)
        pkl_dump({'x_data': X_train, 'y_label': Y_label}, Config.entity_score_data_pkl)

    def train(self):
        if not os.path.isfile(Config.entity_score_data_pkl):
            self.gen_train_data()
        data = pkl_load(Config.entity_score_data_pkl)
        X_train, Y_label = data['x_data'], data['y_label']
        model = linear_model.LogisticRegression(C=1e5)
        model.fit(np.array(X_train), np.array(Y_label))
        pkl_dump(model, Config.entity_score_model_pkl)

    def predict(self, features):
        preds = self.model(features)
        return preds
