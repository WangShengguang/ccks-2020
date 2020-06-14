# -*- coding: utf-8 -*-
import logging
from typing import Tuple, Dict, List

import numpy as np

from ckbqa.models.relation_score.predictor import RelationScorePredictor
from ckbqa.qa.neo4j_graph import GraphDB
from ckbqa.utils.decorators import singleton


@singleton
class RelationExtractor(object):

    def __init__(self):
        self.graph_db = GraphDB()
        self.sim_predictor = RelationScorePredictor(model_name='bert_match')  # bert_match,bert_match2

    def get_relations(self, candidate_entities, ent_name, direction='out') -> Tuple:
        onehop_relations = self.graph_db.get_onehop_relations_by_entName(ent_name, direction=direction)
        twohop_relations = self.graph_db.get_twohop_relations_by_entName(ent_name, direction=direction)
        mention = candidate_entities[ent_name]['mention']
        candidate_paths, candidate_sents = [], []
        for rel_name in onehop_relations:
            _candidate_path = '的'.join([mention, rel_name[1:-1]])  # 查询的关系有<>,需要去掉
            candidate_sents.append(_candidate_path)
            candidate_paths.append([ent_name, rel_name])
        for rels in twohop_relations:
            pure_rel_names = [rel_name[1:-1] for rel_name in rels]  # python-list 关系名列表
            _candidate_path = '的'.join([mention] + pure_rel_names)
            candidate_sents.append(_candidate_path)
            candidate_paths.append([ent_name] + rels)
        return candidate_paths, candidate_sents

    def get_ent_relations(self, q_text: str, candidate_entities: Dict) -> Tuple:
        """
        :param q_text:
        :top_k 10 ; out 10, in 10
        :param candidate_entities: {ent:[mention, feature1, feature2, ...]}
        :return:
        """
        candidate_out_sents, candidate_out_paths = [], []
        candidate_in_sents, candidate_in_paths = [], []
        for entity in candidate_entities:
            candidate_paths, candidate_sents = self.get_relations(candidate_entities, entity, direction='out')
            candidate_out_sents.extend(candidate_sents)
            candidate_out_paths.extend(candidate_paths)
            candidate_paths, candidate_sents = self.get_relations(candidate_entities, entity, direction='in')
            candidate_in_sents.extend(candidate_sents)
            candidate_in_paths.extend(candidate_paths)
        if not candidate_out_sents and not candidate_in_sents:
            logging.info('* candidate_out_paths Empty ...')
            return candidate_out_paths, candidate_in_paths
        # 模型打分排序
        _out_paths = self.relation_score_topn(q_text, candidate_out_paths, candidate_out_sents)
        _in_paths = self.relation_score_topn(q_text, candidate_out_paths, candidate_out_sents)
        return _out_paths, _in_paths

    def relation_score_topn(self, q_text: str, candidate_paths, candidate_sents, top_k=10) -> List:
        top_k = len(candidate_sents)  # TODO 后需删除；不做筛选，保留所有路径；目前筛选效果不好
        if top_k >= len(candidate_sents):  # 太少则不作筛选
            return candidate_paths
        if candidate_sents:
            sim_scores = self.sim_predictor.predict(q_text, candidate_sents)  # 目前算法不好，score==1
            sim_indexs = np.array(sim_scores).argsort()[-top_k:][::-1]
            _paths = [candidate_paths[i] for i in sim_indexs]
        else:
            _paths = []
        return _paths
        # for i in in_sim_indexs:
        #     try:
        #         candidate_in_paths[i]
        #     except:
        #         import traceback
        #         logging.info(traceback.format_exc())
        #         print(f'i: {i}, candidate_in_paths: {len(candidate_in_paths)}')
        #         import ipdb
        #         ipdb.set_trace()
