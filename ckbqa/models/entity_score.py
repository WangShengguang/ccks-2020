'''
从所有候选实体中筛选出topn个候选实体
'''

import re

from sklearn import linear_model
import numpy as np
from ckbqa.dataset.data_prepare import question_patten, entity_patten, load_data
from ckbqa.qa.recognizer import Recognizer
from ckbqa.utils.tools import pkl_dump
from config import Config

ent_attr_partten = re.compile('["<](.*?)[>"]')


class EntityScore(object):
    def __init__(self):
        self.recognizer = Recognizer()

    def train(self):
        X_train = []
        Y_label = []
        for q, sparql, a in load_data():
            a_entities = entity_patten.findall(a)
            q_entities = ent_attr_partten.findall(sparql)  # attr
            q_text = question_patten.findall(q)[0]
            candidate_entities = self.recognizer.get_candate_entities(q_text)
            for ent_name, feature_dict in candidate_entities.items():
                feature = feature_dict['feature']
                label = 1 if ent_name in q_entities else 0  # 候选实体有的不在答案中
                X_train.append(feature)
                Y_label.append(label)
        model = linear_model.LogisticRegression(C=1e5)
        model.fit(np.array(X_train), np.array(Y_label))
        pkl_dump(model, Config.entity_score_model_pkl)
