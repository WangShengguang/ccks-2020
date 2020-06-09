import logging
import threading

from ckbqa.qa.algorithms import Algorithm
from ckbqa.qa.cache import Memory
from ckbqa.qa.lac_tools import BaiduLac
from ckbqa.qa.neo4j_graph import GraphDB
from ckbqa.qa.recognizer import Recognizer
from ckbqa.qa.relation_extractor import RelationExtractor
from ckbqa.utils.async_tools import apply_async


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
        t.join()  # 等待所有线程结束再往下
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
        logging.info('* get_candate_entities start ...')
        candidate_entities = self.recognizer.get_candate_entities(q_text)
        logging.info('* relation_extractor.extract_ent_rel start ...')
        candidate_paths = self.relation_extractor.extract_ent_rel(q_text, candidate_entities)
        # 生成cypher语句并查询
        logging.info(f'* get_most_overlap_path start , candidate_paths: {len(candidate_paths)}...')
        top_path = self.algo.get_most_overlap_path(q_text, candidate_paths)
        # self.graph_db.cache()  # 缓存neo4j查询结果
        apply_async(self.graph_db.cache)
        logging.info(f'* candidate_paths: {len(candidate_paths)}')
        if len(top_path) == 2:
            ent_name, rel_name = top_path
            answer = self.graph_db.search_by_2path(ent_name, rel_name)
        elif len(top_path) == 3:
            ent_name, rel1_name, rel2_name = top_path
            answer = self.graph_db.search_by_3path(ent_name, rel1_name, rel2_name)
        elif len(top_path) == 4:
            ent1_name, rel1_name, rel2_name, ent2_name = top_path
            answer = self.graph_db.search_by_4path(ent1_name, rel1_name, rel2_name, ent2_name)
        else:
            print('这个查询路径不规范')
            answer = []
        # print(q_text)
        # print(f'***** answer : {len(answer)}, ** {answer[:5]}')
        # import ipdb
        # ipdb.set_trace()
        return answer
    # def answer(self, q_text: str):
    #     nv_words = self.recognizer.get_nv_words(q_text)
    #     ent_rels = []
    #     for w in nv_words:
    #         if w in self.entity2id:
    #             ent_rels.append(w)
    #         elif w in self.relations_set:
    #             ent_rels.append(w)
    #     entity_ids = [self.entity2id[w] for w in nv_words if w in self.entity2id]
    #
    #     triples_vec = self.bert_encoder(self.sent2ids(['']))
    #     text_vec = self.bert_encoder(self.sent2ids([q_text]))
    #     import ipdb
    #     ipdb.set_trace()
