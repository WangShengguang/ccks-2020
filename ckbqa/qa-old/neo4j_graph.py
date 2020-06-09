# coding: utf-8
import logging
import os

from py2neo import Graph

from ckbqa.qa.cache import Memory
from ckbqa.utils.decorators import singleton
from ckbqa.utils.tools import json_load, json_dump
from config import Config

graph = Graph('http://localhost:7474', username='neo4j', password='password')  # bolt://localhost:7687


@singleton
class GraphDB(object):
    def __init__(self):
        # self.node_selector = NodeMatcher(graph)
        self._one_hop_relNames_map = {}
        self._two_hop_relNames_map = {}
        self._one_hop_relNames_count = {}
        self.memory = Memory()
        self.load_cache()

    def __del__(self):
        self.cache()

    def load_cache(self):
        if os.path.isfile(Config.neo4j_query_cache):
            parms = json_load(Config.neo4j_query_cache)
            self._one_hop_relNames_map = parms['_one_hop_relNames_map']
            self._two_hop_relNames_map = parms['_two_hop_relNames_map']
            self._one_hop_relNames_count = parms['_one_hop_relNames_count']
            # for key, values in parms.items():
            #     setattr(self, key, values)

    def cache(self):
        parms = {'_one_hop_relNames_map': self._one_hop_relNames_map,
                 '_two_hop_relNames_map': self._two_hop_relNames_map,
                 '_one_hop_relNames_count': self._one_hop_relNames_count}
        json_dump(parms, Config.neo4j_query_cache)

    # def find_entity_by_id(self, entity_id: int):
    #     node = self.node_selector.match("Entity", id=entity_id).first()
    #     return node

    def get_onehop_relations_by_entName(self, ent_name: str):
        if ent_name not in self._one_hop_relNames_map:
            ent_id = self.memory.get_entity_id(ent_name)
            cpl = f"MATCH p=(ent:Entity)-[r1:Relation]->()  WHERE ent.id={ent_id} RETURN DISTINCT r1.name"
            rel_names = [rel.data()['r1.name'] for rel in graph.run(cpl)]
            # TODO 反向查询
            # cpl = f"MATCH p=(ent:Entity)-[r1:Relation]->()  WHERE ent.id={ent_id} RETURN DISTINCT r1.name"
            # rel_names = [rel.data()['r1.name'] for rel in graph.run(cpl)]
            self._one_hop_relNames_map[ent_name] = rel_names
            self._one_hop_relNames_count[ent_name] = len(rel_names)
        rel_names = self._one_hop_relNames_map[ent_name]
        return rel_names

    def get_twohop_relations_by_entName(self, ent_name: str):
        if ent_name not in self._two_hop_relNames_map:  # 2032001,23624760,25545439,
            ent_id = self.memory.get_entity_id(ent_name)
            cql = f"MATCH p=(ent:Entity)-[r1:Relation]->()-[r2:Relation]->()  WHERE ent.id={ent_id} RETURN DISTINCT r1.name,r2.name"
            rels = [rel.data() for rel in graph.run(cql)]
            rel_names = [[d['r1.name'], d['r2.name']] for d in rels]
            self._two_hop_relNames_map[ent_name] = rel_names
        rel_names = self._two_hop_relNames_map[ent_name]
        return rel_names

    def get_onehop_relCount_by_entName(self, ent_name: str):
        '''根据实体名，得到与之相连的关系数量，代表实体在知识库中的流行度'''
        if ent_name not in self._one_hop_relNames_count:
            ent_id = self.memory.get_entity_id(ent_name)
            cql = f"match p=(ent:Entity)-[r1:Relation]-() where ent.id={ent_id} return count(p) AS count"
            count = [ent['count'] for ent in graph.run(cql)][0]
            self._one_hop_relNames_count[ent_name] = count
        count = self._one_hop_relNames_count[ent_name]
        return count

    def search_by_2path(self, ent_name, rel_name):
        ent_id = self.memory.get_entity_id(ent_name)
        cql = f"match (ent:Entity)-[r1:Relation]-(target) where ent.id={ent_id} and r1.name='{rel_name}' return target.name"
        logging.info(cql)
        res = graph.run(cql)
        target_name = [ent['target.name'] for ent in res]
        return target_name

    def search_by_3path(self, ent_name, rel1_name, rel2_name):
        ent_id = self.memory.get_entity_id(ent_name)
        cql = (f"match (ent:Entity)-[r1:Relation]-()-[r2:Relation]-(target) where ent.id={ent_id} "
               f" and r1.name='{rel1_name}' and r2.name='{rel2_name}' return target.name")
        logging.info(cql)
        res = graph.run(cql)
        target_name = [ent['target.name'] for ent in res]
        return target_name

    def search_by_4path(self, ent1_name, rel1_name, rel2_name, ent2_name):
        ent1_id = self.get_entity_id(ent1_name)
        ent2_id = self.get_entity_id(ent2_name)
        # rel1_id = self.entity2id[rel1_name]
        # rel2_id = self.entity2id[rel2_name]
        cql = (f"match (ent:Entity)-[r1:Relation]-(target)-[r2:Relation]-(ent2:Entity) "
               f" where ent.id={ent1_id} and r1.name='{rel1_name}' and r2.name='{rel2_name}' and ent2.id={ent2_id}"
               f"return target.name")
        logging.info(cql)
        res = graph.run(cql)
        target_name = [ent['target.name'] for ent in res]
        return target_name
