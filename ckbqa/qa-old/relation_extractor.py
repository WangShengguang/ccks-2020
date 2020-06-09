# -*- coding: utf-8 -*-
import logging

from ckbqa.qa.neo4j_graph import GraphDB
from ckbqa.utils.decorators import singleton


@singleton
class RelationExtractor(object):

    def __init__(self):
        self.graph_db = GraphDB()
        # self.sim_predictor = BertMatchPredictor(model_name='bert_match')  # bert_match,bert_match2

    def extract_ent_rel(self, q_text, candidate_entities):
        """
        :param q_text:
        :param candidate_entities: {ent:[mention, feature1, feature2, ...]}
        :return:
        """
        logging.info('extract_ent_rel start ...')
        candidate_sents = []
        candidate_paths = []
        for entity in candidate_entities:
            onehop_relations = self.graph_db.get_onehop_relations_by_entName(entity)
            twohop_relations = self.graph_db.get_twohop_relations_by_entName(entity)
            mention = candidate_entities[entity]['mention']
            for rel_name in onehop_relations:
                _candidate_path = '的'.join([mention, rel_name[1:-1]])  # 查询的关系有<>,需要去掉
                candidate_sents.append(_candidate_path)
                candidate_paths.append([entity, rel_name])
            for rels in twohop_relations:
                pure_rel_names = [rel_name[1:-1] for rel_name in rels]  # python-list 关系名列表
                _candidate_path = '的'.join([mention] + pure_rel_names)
                candidate_sents.append(_candidate_path)
                candidate_paths.append([entity] + rels)
        if not candidate_sents:
            logging.info('* candidate_paths Empty ...')
            return
        logging.info(f'sim predict start , candidate_paths : {len(candidate_paths)}...')
        # sim_scores = self.sim_predictor.predict(q_text, candidate_paths) #目前算法不好，score==1
        return candidate_paths

    def relation_score_topn(self):
        pass
