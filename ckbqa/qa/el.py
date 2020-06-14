"""实体链接模块"""
import gc
from collections import defaultdict
from typing import List, Dict

import numpy as np

from ckbqa.models.entity_score.model import EntityScore
from ckbqa.qa.algorithms import sequences_set_similar
from ckbqa.qa.cache import Memory
from ckbqa.qa.lac_tools import Ngram, BaiduLac
from ckbqa.qa.neo4j_graph import GraphDB
from ckbqa.utils.decorators import singleton


@singleton
class CEG(object):
    """ Candidate Entity Generation： NER
        从mention出发，找到KB中所有可能的实体，组成候选实体集 (candidate entities)；
        最主流也最有效的方法就是Name Dictionary，说白了就是配别名
    """

    def __init__(self):
        # async_init_singleton_class([JiebaLac, BaiduLac, Memory])
        print('start ...')
        # self.jieba = JiebaLac(load_custom_dict=True)
        gc.collect()
        print('jieba init done ...')
        self.lac = BaiduLac(mode='lac', _load_customization=True)
        self.memory = Memory()
        self.ngram = Ngram()
        print(' memo init  done ...')
        self.mention_stop_words = {'是什么', '在哪里', '哪里', '什么', '提出的', '有什么', '国家', '哪个', '所在',
                                   '培养出', '为什么', '什么时候', '人', '你知道', '都包括', '是谁', '告诉我',
                                   '又叫做'}
        self.entity_stop_words = {'有', '是', '《', '》', '率', '·'}

        self.main_tags = {'n', 'an', 'nw', 'nz', 'PER', 'LOC', 'ORG', 'TIME',
                          'vn', 'v'}

    def get_ent2mention(self, q_text):
        '''
                标签	含义	标签	含义	标签	含义	标签	含义
                n	普通名词	f	方位名词	s	处所名词	nw	作品名
                nz	其他专名	v	普通动词	vd	动副词	vn	名动词
                a	形容词	ad	副形词	an	名形词	d	副词
                m	数量词	q	量词	r	代词	p	介词
                c	连词	u	助词	xc	其他虚词	w	标点符号
                PER	人名	LOC	地名	ORG	机构名	TIME	时间
        '''
        entity2mention = {}
        words, tags = self.lac.run(q_text)
        main_text = ''.join([word for word, tag in zip(words, tags) if tag in self.main_tags])
        main_text_ngrams = list(self.ngram.get_all_grams(main_text))
        q_text_grams = list(self.ngram.get_all_grams(q_text))
        for gram in set(q_text_grams + main_text_ngrams):
            if f"<{gram}>" in self.memory.entity2id:
                entity2mention[f'<{gram}>'] = gram
            if gram in self.memory.entity2id:
                entity2mention[gram] = gram
            if gram in self.memory.mention2entity and gram not in self.mention_stop_words:
                for ent in self.memory.mention2entity[gram]:
                    if f"<{ent}>" in self.memory.entity2id:
                        entity2mention[f'<{ent}>'] = gram
                    if ent in self.memory.entity2id:
                        entity2mention[ent] = gram
            # print(gram)
        return entity2mention

    def seg_text(self, text):
        return self.lac.run(text)[0]

    # def _get_ent2mention(self, text: str) -> Dict:
    #     entity2mention = {}
    #     lac_words, tags = self.lac.run(text)
    #     jieba_words = [w for w in self.jieba.cut_for_search(text)]
    #     for word in set(lac_words + jieba_words):
    #         # if word in self.entity_stop_words:
    #         #     continue
    #         if word in self.memory.mention2entity and word not in self.mention_stop_words:
    #             for ent in self.memory.mention2entity[word]:
    #                 if f"<{ent}>" in self.memory.entity2id:
    #                     entity2mention[f'<{ent}>'] = word
    #                 if ent in self.memory.entity2id:
    #                     entity2mention[ent] = word
    #         if f"<{word}>" in self.memory.entity2id:
    #             entity2mention[f'<{word}>'] = word
    #         if word in self.memory.entity2id:
    #             entity2mention[word] = word
    #     return entity2mention


@singleton
class ED(object):
    """ Entity Disambiguation：
        从candidate entities中，选择最可能的实体作为预测实体。
    """

    def __init__(self):
        self.memory = Memory()
        self.graph_db = GraphDB()
        self.ceg = CEG()
        self.mention_stop_words = {'是什么', '在哪里', '哪里', '什么', '提出的', '有什么', '国家', '哪个', '所在',
                                   '培养出', '为什么', '什么时候', '人', '你知道', '都包括', '是谁', '告诉我',
                                   '又叫做'}
        self.entity_stop_words = {'有', '是', '《', '》', '率', '·'}
        # self.entity_score = EntityScore()
        self.entity_score = EntityScore(load_pretrain_model=True)

    def ent_rel_similar(self, question: str, entity: str, relations: List):
        '''
        抽取每个实体或属性值2hop内的所有关系，来跟问题计算各种相似度特征
        input:
            question: python-str
            entity: python-str <entityname>
            relations: python-dic key:<rname>
        output：
            [word_overlap,char_overlap,word_embedding_similarity,char_overlap_ratio]
        '''
        # 得到主语-谓词的tokens及chars
        rel_tokens = set()
        for rel_name in set(relations):
            rel_tokens.update(self.ceg.seg_text(rel_name[1:-1]))
        rel_chars = set([char for rel_name in relations for char in rel_name])
        #
        question_tokens = set(self.ceg.seg_text(question))
        question_chars = set(question)
        #
        entity_tokens = set(self.ceg.seg_text(entity[1:-1]))
        entity_chars = set(entity[1:-1])
        #
        qestion_ent_sim = (sequences_set_similar(question_tokens, entity_tokens) +
                           sequences_set_similar(question_chars, entity_chars))
        qestion_rel_sim = (sequences_set_similar(question_tokens, rel_tokens) +
                           sequences_set_similar(question_chars, rel_chars))
        # 实体名和问题的overlap除以实体名长度的比例
        return qestion_ent_sim + qestion_rel_sim

    def get_entity_feature(self, q_text, ent_name):
        # 得到实体两跳内的所有关系
        one_hop_out_rel_names = self.graph_db.get_onehop_relations_by_entName(ent_name, direction='out')
        two_hop_out_rel_names = self.graph_db.get_twohop_relations_by_entName(ent_name, direction='out')
        out_rel_names = [rel for rels in two_hop_out_rel_names for rel in rels] + one_hop_out_rel_names
        #
        one_hop_in_rel_names = self.graph_db.get_onehop_relations_by_entName(ent_name, direction='in')
        two_hop_in_rel_names = self.graph_db.get_twohop_relations_by_entName(ent_name, direction='in')
        in_rel_names = [rel for rels in two_hop_in_rel_names for rel in rels] + one_hop_in_rel_names
        # 计算问题和主语实体及其两跳内关系间的相似度
        similar_feature = self.ent_rel_similar(q_text, ent_name, out_rel_names + in_rel_names)
        # 实体的流行度特征
        popular_feature = self.graph_db.get_onehop_relCount_by_entName(ent_name)
        return similar_feature, popular_feature

    def get_candidate_entities(self, q_text: str, entity2mention: dict) -> Dict:
        """
        :param q_text:
        :return: candidate_subject: { ent_name: {'mention':mention_txt,
                                                  'feature': [feature1, feature2, ...]
                                                  },
                                    ...
                                    }
        """
        candidate_subject = defaultdict(dict)  # {ent:[mention, feature1, feature2, ...]}
        #
        for entity, mention in entity2mention.items():
            similar_feature, popular_feature = self.get_entity_feature(q_text, entity)
            candidate_subject[entity]['mention'] = mention
            candidate_subject[entity]['feature'] = [sim for sim in similar_feature] + [popular_feature ** 0.5]

        # subject_score_topn打分做筛选
        top_subjects = self.subject_score_topn(candidate_subject)
        return top_subjects

    def subject_score_topn(self, candidate_entities: dict, top_k=10):
        '''
            :candidate_entities  {ent: }
        输入候选主语和对应的特征，使用训练好的模型进行打分，排序后返回前topn个候选主语
        '''
        top_k = len(candidate_entities)  # TODO 不做筛选，保留所有实体；目前筛选效果不好
        if top_k >= len(candidate_entities):
            return candidate_entities
        entities = []
        features = []
        for ent, feature_dic in candidate_entities.items():
            entities.append(ent)
            features.append(feature_dic['feature'])
        scores = self.entity_score.predict(features)
        # print(f'scores: {scores}')
        top_k_indexs = np.asarray(scores).argsort()[-top_k:][::-1]
        top_k_entities = {entities[i]: candidate_entities[entities[i]]
                          for i in top_k_indexs}
        return top_k_entities


@singleton
class EL(object):
    """实体链接"""

    def __init__(self):
        self.ceg = CEG()  # Candidate Entity Generation
        self.ed = ED()  # Entity Disambiguation

    def el(self, q_text: str) -> Dict:
        ent2mention = self.ceg.get_ent2mention(q_text)
        candidate_entities = self.ed.get_candidate_entities(q_text, ent2mention)
        return candidate_entities

# try
# except:
#     import traceback,ipdb
#     traceback.print_exc()
#     ipdb.set_trace()
