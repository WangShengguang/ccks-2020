'''
从所有候选实体中筛选出topn个候选实体
'''

import os

import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression

from ckbqa.dataset.data_prepare import question_patten, entity_pattern, attr_pattern, load_data
from ckbqa.utils.tools import pkl_dump, pkl_load
from config import Config


class EntityScore(object):
    """
        主实体打分排序；
    """

    def __init__(self, load_pretrain_model=False):
        if load_pretrain_model:
            self.model: LogisticRegression = pkl_load(Config.entity_score_model_pkl)

    def gen_train_data(self):
        X_train = []
        Y_label = []
        from ckbqa.qa.el import EL  # 避免循环导入
        el = EL()
        for q, sparql, a in load_data(tqdm_prefix='EntityScore train data '):
            # a_entities = entity_pattern.findall(a)
            q_entities = set(entity_pattern.findall(sparql) + attr_pattern.findall(sparql))  # attr
            q_text = question_patten.findall(q)[0]
            candidate_entities = el.el(q_text)
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
        X_train = preprocessing.scale(np.array(data['x_data']))
        # Y_label = np.eye(2)[data['y_label']]  # class_num==2
        Y_label = np.array(data['y_label'])  # class_num==2
        print(f"X_train : {X_train.shape}, Y_label: {Y_label.shape}")
        print(f"sum(Y_label): {sum(Y_label)}")
        model = LogisticRegression(C=1e5, verbose=1)
        model.fit(X_train, Y_label)
        # model.predict()
        accuracy_score = model.score(X_train, Y_label)
        print(f"accuracy_score: {accuracy_score:.4f}")
        pkl_dump(model, Config.entity_score_model_pkl)

    def predict(self, features):
        preds = self.model.predict(np.asarray(features))
        return preds
