# coding: utf-8
import logging
import os
from queue import Queue

from py2neo import Graph

from ckbqa.qa.cache import Memory
from ckbqa.utils.async_tools import apply_async
from ckbqa.utils.decorators import singleton
from ckbqa.utils.tools import json_load, json_dump
from config import Config

graph = Graph('http://localhost:7474', username='neo4j', password='password')  # bolt://localhost:7687


@singleton
class GraphDB(object):
    def __init__(self):
        self._one_hop_relNames_map = {'in': {}, 'out': {}}
        self._two_hop_relNames_map = {'in': {}, 'out': {}}
        self.all_directions = {'in', 'out'}
        self.memory = Memory()
        self.load_cache()
        apply_async(self.async_cache)

    def __del__(self):
        self.cache()

    def get_total_entity_count(self):
        count = (len(self._one_hop_relNames_map['out']) +
                 len(self._one_hop_relNames_map['in']) +
                 len(self._two_hop_relNames_map['out']) +
                 len(self._two_hop_relNames_map['in']))
        return count

    def load_cache(self):
        self.queue = Queue(1)
        if os.path.isfile(Config.neo4j_query_cache):
            data_dict = json_load(Config.neo4j_query_cache)
            self._one_hop_relNames_map = data_dict['_one_hop_relNames_map']
            self._two_hop_relNames_map = data_dict['_two_hop_relNames_map']
            self.total_count = self.get_total_entity_count()
            logging.info(f'load neo4j_query_cache, total: {self.total_count}')
        else:
            logging.info(f'not found neo4j_query_cache: {Config.neo4j_query_cache}')
            self.total_count = 0

    def update_total_queue(self):
        total = self.get_total_entity_count()
        self.queue.put(total)

    def async_cache(self):
        while True:
            total = self.queue.get()
            if total > self.total_count:
                self.cache()

    def cache(self):
        total = self.get_total_entity_count()
        if total > self.total_count:
            parms = {'_one_hop_relNames_map': self._one_hop_relNames_map,
                     '_two_hop_relNames_map': self._two_hop_relNames_map}
            json_dump(parms, Config.neo4j_query_cache)
            logging.info(f'neo4j_query_cache , increase: {total - self.total_count}, total: {total}')
            self.total_count = total

    def get_onehop_relations_by_entName(self, ent_name: str, direction='out'):
        assert direction in self.all_directions
        if ent_name not in self._one_hop_relNames_map[direction]:
            ent_id = self.memory.get_entity_id(ent_name)
            start, end = ('ent:Entity', '') if direction == 'out' else ('', 'ent:Entity')
            cpl = f"MATCH p=({start})-[r1:Relation]->({end})  WHERE ent.id={ent_id} RETURN DISTINCT r1.name"
            _one_hop_relNames = [rel.data()['r1.name'] for rel in graph.run(cpl)]
            self._one_hop_relNames_map[direction][ent_name] = _one_hop_relNames
        rel_names = self._one_hop_relNames_map[direction][ent_name]
        return rel_names

    def get_twohop_relations_by_entName(self, ent_name: str, direction='out'):
        assert direction in self.all_directions
        if ent_name not in self._two_hop_relNames_map[direction]:
            ent_id = self.memory.get_entity_id(ent_name)
            start, end = ('ent:Entity', '') if direction == 'out' else ('', 'ent:Entity')
            cql = f"MATCH p=({start})-[r1:Relation]->()-[r2:Relation]->({end})  WHERE ent.id={ent_id} RETURN DISTINCT r1.name,r2.name"
            rels = [rel.data() for rel in graph.run(cql)]
            rel_names = [[d['r1.name'], d['r2.name']] for d in rels]
            self._two_hop_relNames_map[direction][ent_name] = rel_names
        rel_names = self._two_hop_relNames_map[direction][ent_name]
        return rel_names

    def get_onehop_relCount_by_entName(self, ent_name: str):
        '''根据实体名，得到与之相连的关系数量，代表实体在知识库中的流行度'''
        if ent_name not in self._one_hop_relNames_map['out']:
            rel_names = self.get_onehop_relations_by_entName(ent_name, direction='out')
            self._one_hop_relNames_map['out'][ent_name] = rel_names
        if ent_name not in self._one_hop_relNames_map['in']:
            rel_names = self.get_onehop_relations_by_entName(ent_name, direction='in')
            self._one_hop_relNames_map['in'][ent_name] = rel_names
        count = len(self._one_hop_relNames_map['out'][ent_name]) + len(self._one_hop_relNames_map['in'][ent_name])
        return count

    def search_by_2path(self, ent_name, rel_name, direction='out'):
        ent_id = self.memory.get_entity_id(ent_name)
        assert direction in self.all_directions
        start, end = ('ent:Entity', 'target') if direction == 'out' else ('target', 'ent:Entity')
        cql = (f"match ({start})-[r1:Relation]-({end}) where ent.id={ent_id} "
               f" and r1.name='{rel_name}' return DISTINCT target.name")
        logging.info(f"{cql}; {ent_name}")
        res = graph.run(cql)
        target_name = [ent['target.name'] for ent in res]
        return target_name

    def search_by_3path(self, ent_name, rel1_name, rel2_name, direction='out'):
        ent_id = self.memory.get_entity_id(ent_name)
        assert direction in self.all_directions
        start, end = ('ent:Entity', 'target') if direction == 'out' else ('target', 'ent:Entity')
        cql = (f"match ({start})-[r1:Relation]-()-[r2:Relation]-({end}) where ent.id={ent_id} "
               f" and r1.name='{rel1_name}' and r2.name='{rel2_name}' return DISTINCT target.name")
        logging.info(f"{cql}; {ent_name}")
        res = graph.run(cql)
        target_name = [ent['target.name'] for ent in res]
        return target_name

    def search_by_4path(self, ent1_name, rel1_name, rel2_name, ent2_name, direction='out'):
        ent1_id = self.get_entity_id(ent1_name)
        ent2_id = self.get_entity_id(ent2_name)
        assert direction in self.all_directions
        start, end = ('ent:Entity', 'target') if direction == 'out' else ('target', 'ent:Entity')
        cql = (f"match ({start})-[r1:Relation]-({end})-[r2:Relation]-(ent2:Entity) "
               f" where ent.id={ent1_id} and r1.name='{rel1_name}' and r2.name='{rel2_name}' and ent2.id={ent2_id}"
               f"return DISTINCT target.name")
        logging.info(f"{cql}; {ent1_name},{ent2_name}")
        res = graph.run(cql)
        target_name = [ent['target.name'] for ent in res]
        return target_name
