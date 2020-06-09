import logging
import threading

from ckbqa.qa.algorithms import Algorithm
from ckbqa.qa.cache import Memory
from ckbqa.qa.lac_tools import BaiduLac
from ckbqa.qa.neo4j_graph import GraphDB
from ckbqa.qa.recognizer import Recognizer
from ckbqa.qa.relation_extractor import RelationExtractor


# for sc in A.__subclasses__():#所有子类
#     print(sc.__name__)
def preprocess_init():
    logging.info('preprocess singleton_class init start ...')
    thrs = [threading.Thread(target=singleton_class)
            for singleton_class in [Memory, Recognizer, GraphDB,
                                    RelationExtractor, BaiduLac]]
    for t in thrs:
        t.start()
    for t in thrs:
        t.join()  # 等待所有线程结束再往下继续
    logging.info('preprocess singleton_class init done ...')


class QA(object):
    def __init__(self):
        preprocess_init()
        logging.info('QA init start ...')
        self.graph_db = GraphDB()
        self.memory = Memory()
        self.recognizer = Recognizer()
        self.relation_extractor = RelationExtractor()
        self.algo = Algorithm()
        logging.info('QA init Done ...')

    def run(self, q_text):
        logging.info('--' * 20)
        logging.info(f"* q_text: {q_text}")
        candidate_entities = self.recognizer.get_candidate_entities(q_text)
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
        self.graph_db.update_total_queue()
        if max_out_score > max_in_score:
            direction = 'out'
            top_path = top_out_path
        else:
            direction = 'in'
            top_path = top_in_path
        logging.info(f'* get_most_overlap_path: {top_path} ...')
        if len(top_path) == 2:
            ent_name, rel_name = top_path
            answer = self.graph_db.search_by_2path(ent_name, rel_name, direction=direction)
        elif len(top_path) == 3:
            ent_name, rel1_name, rel2_name = top_path
            answer = self.graph_db.search_by_3path(ent_name, rel1_name, rel2_name, direction=direction)
        elif len(top_path) == 4:
            ent1_name, rel1_name, rel2_name, ent2_name = top_path
            answer = self.graph_db.search_by_4path(ent1_name, rel1_name, rel2_name, ent2_name, direction=direction)
        else:
            print('这个查询路径不规范')
            answer = []
        logging.info(f"* cypher answer: {answer}")
        return answer

    # def evaluate(self, question, subject_entities, answer_entities):
    #     q_text = question_patten.findall(question)[0]
    #     candidate_entities = self.recognizer.get_candidate_entities(q_text)
    #     precision, recall, f1 = get_metrics(subject_entities, candidate_entities)
    #     logging.info(f'get_candidate_entities, precision: {precision}, recall: {recall}, f1: {f1}')
