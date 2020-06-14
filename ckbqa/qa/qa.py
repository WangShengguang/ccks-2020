import logging

from ckbqa.qa.algorithms import Algorithms
from ckbqa.qa.cache import Memory
from ckbqa.qa.el import EL
from ckbqa.qa.neo4j_graph import GraphDB
from ckbqa.qa.relation_extractor import RelationExtractor
from ckbqa.utils.async_tools import async_init_singleton_class


class QA(object):
    def __init__(self):
        async_init_singleton_class([Memory, GraphDB, RelationExtractor, EL])
        logging.info('QA init start ...')
        self.graph_db = GraphDB()
        self.memory = Memory()
        self.el = EL()
        self.relation_extractor = RelationExtractor()
        self.algo = Algorithms()
        logging.info('QA init Done ...')

    def query_path(self, path, direction='out'):
        if len(path) == 2:
            ent_name, rel_name = path
            answer = self.graph_db.search_by_2path(ent_name, rel_name, direction=direction)
        elif len(path) == 3:
            ent_name, rel1_name, rel2_name = path
            answer = self.graph_db.search_by_3path(ent_name, rel1_name, rel2_name, direction=direction)
        elif len(path) == 4:
            ent1_name, rel1_name, rel2_name, ent2_name = path
            answer = self.graph_db.search_by_4path(ent1_name, rel1_name, rel2_name, ent2_name, direction=direction)
        else:
            logging.info(f'这个查询路径不规范: {path}')
            answer = []
        return answer

    def run(self, q_text, return_candidates=False):
        logging.info(f"* q_text: {q_text}")
        candidate_entities = self.el.el(q_text)
        logging.info(f'* get_candidate_entities {len(candidate_entities)} ...')
        candidate_out_paths, candidate_in_paths = self.relation_extractor.get_ent_relations(q_text, candidate_entities)
        logging.info(f'* get candidate_out_paths: {len(candidate_out_paths)},'
                     f'candidate_in_paths: {len(candidate_in_paths)} ...')
        # 生成cypher语句并查询
        top_out_path, max_out_score = self.algo.get_most_overlap_path(q_text, candidate_out_paths)
        logging.info(f'* get_most_overlap out path：{max_out_score:.4f}， {top_out_path} ...')
        top_in_path, max_in_score = self.algo.get_most_overlap_path(q_text, candidate_in_paths)
        logging.info(f'* get_most_overlap in path：{max_in_score:.4f}， {top_in_path} ...')
        # self.graph_db.cache()  # 缓存neo4j查询结果
        if max_out_score >= max_in_score:  # 分数相同时取out
            direction = 'out'
            top_path = top_out_path
        else:
            direction = 'in'
            top_path = top_in_path
        logging.info(f'* get_most_overlap_path: {top_path} ...')
        result_ents = self.query_path(top_path, direction=direction)
        if not result_ents and len(top_path) > 2:
            if direction == 'out':
                top_path = top_path[:2]
            else:
                top_path = top_path[-2:]
            result_ents = self.query_path(top_path)
            if not result_ents:
                top_path = top_path[0] + top_path[-1]
                result_ents = self.query_path(top_path, direction=direction)
        logging.info(f"* cypher result_ents: {result_ents}" + '\n' + '--' * 10 + '\n\n')
        if return_candidates:
            return result_ents, candidate_entities, candidate_out_paths, candidate_in_paths
        return result_ents

    # def evaluate(self, question, subject_entities, result_ents_entities):
    #     q_text = question_patten.findall(question)[0]
    #     candidate_entities = self.recognizer.get_candidate_entities(q_text)
    #     precision, recall, f1 = get_metrics(subject_entities, candidate_entities)
    #     logging.info(f'get_candidate_entities, precision: {precision}, recall: {recall}, f1: {f1}')
