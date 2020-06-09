from collections import defaultdict
from typing import List

import jieba

from ckbqa.qa.algorithms import sequences_set_similar
from ckbqa.qa.cache import Memory
from ckbqa.qa.lac_tools import BaiduLac
from ckbqa.qa.neo4j_graph import GraphDB
from ckbqa.utils.decorators import singleton

jieba.enable_paddle()  # 启动paddle模式。 0.40版之后开始支持，早期版本不支持


@singleton
class Recognizer(object):
    def __init__(self):
        self.memory = Memory()
        self.graph_db = GraphDB()
        self.lac = BaiduLac(mode='lac', _load_customization=True)
        self.mention_stop_words = {'是什么', '在哪里', '哪里', '什么', '提出的', '有什么', '国家', '哪个', '所在',
                                   '培养出', '为什么', '什么时候', '人', '你知道', '都包括', '是谁', '告诉我',
                                   '又叫做'}
        self.entity_stop_words = {'有', '是', '《', '》', '率', '·'}

    def get_ner_words(self, text):
        words, tags = self.lac.run(text)
        mentions = []
        entities = []
        for word, tag in zip(words, tags):
            if len(word) < 2 or word in self.entity_stop_words:
                continue
            if tag == 'ENTITY' or f"<{word}>" in self.memory.entity2id:
                entities.append(word)
            elif tag == 'MENTION' or word in self.memory.mention2entity:
                if word not in self.mention_stop_words:
                    mentions.append(word)
        return mentions, entities

    def seg_text(self, text):
        return self.lac.run(text)[0]

    def get_attrs(self, text):
        words, tags = self.lac.run(text)
        jieba_words = list(jieba.cut_for_search(text))
        attrs = set(words + jieba_words) & self.memory.all_attrs
        return attrs

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
            rel_tokens.update(self.seg_text(rel_name[1:-1]))
        rel_chars = set([char for rel_name in relations for char in rel_name])
        #
        question_tokens = set(self.seg_text(question))
        question_chars = set(question)
        #
        entity_tokens = set(self.seg_text(entity[1:-1]))
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

    def get_candidate_entities(self, q_text):
        """
        :param q_text:
        :return: candidate_subject: { ent:[mention,feature1, feature2, ...]}
        """
        mentions, _entities = self.get_ner_words(q_text)
        candidate_subject = defaultdict(dict)  # {ent:[mention, feature1, feature2, ...]}

        def add_candidate_feature(mention, ent_name):
            # if len(ent_name) < 4:  # <北京>
            #     return
            similar_feature, popular_feature = self.get_entity_feature(q_text, ent_name)
            candidate_subject[ent_name]['mention'] = mention
            candidate_subject[ent_name]['feature'] = [sim for sim in similar_feature] + [popular_feature ** 0.5]

        for mention in mentions:
            for ent in self.memory.mention2entity[mention]:
                ent_name = f"<{ent}>"
                add_candidate_feature(mention, ent_name)
        for ent in _entities:
            ent_name = f"<{ent}>"
            add_candidate_feature(ent, ent_name)
        for attr_name in self.get_attrs(q_text):
            if attr_name not in candidate_subject:
                add_candidate_feature(attr_name, attr_name)
        return candidate_subject

    def subject_score_topn(self, candidate_entities: dict):
        '''
            :candidate_entities  {ent: }
        输入候选主语和对应的特征，使用训练好的模型进行打分，排序后返回前topn个候选主语
        '''
        entitys = []
        features = []
        for ent, feature_dic in candidate_entities.items():
            entitys.append(ent)
            features.append(feature_dic['feature'])
        # 打分做筛选
        return candidate_entities

# self.ent_tags = {'ORG', 'PER', 'LOC', 'TIME'}  # , 'nz', 'n', 'nw'}
# self.rel_tags = {'v'}
# self.legal_tags = {'ENTITY', 'MENTION'}
#
