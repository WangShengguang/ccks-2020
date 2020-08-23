"""
与知识图谱相关的数据与处理
"""
import gc
import logging
import os
import re
from collections import defaultdict, Counter
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ckbqa.utils.tools import json_dump, get_file_linenums, pkl_dump, json_load, tqdm_iter_file
from config import Config, mention2ent_txt, kb_triples_txt


def iter_triples(tqdm_prefix=''):
    """迭代图谱的三元组；"""
    rdf_patten = re.compile(r'(["<].*?[>"])')
    ent_partten = re.compile('<(.*?)>')
    attr_partten = re.compile('"(.*?)"')

    def parse_entities(string):
        ents = ent_partten.findall(string)
        if ents:  # 加上括号，使用上面会去掉括号
            ents = [f'<{_ent.strip()}>' for _ent in ents if _ent]
        else:  # 去掉引号
            ents = [_ent.strip() for _ent in attr_partten.findall(string) if _ent]
        return ents

    line_num = get_file_linenums(kb_triples_txt)
    # f_record = open('fail.txt', 'w', encoding='utf-8')
    with open(kb_triples_txt, 'r', encoding='utf-8') as f:
        desc = Path(kb_triples_txt).name + '-' + 'iter_triples'
        for line in tqdm(f, total=line_num, desc=desc):
            # for idx, line in enumerate(f):
            #     if idx < 6618182:
            #         continue
            # 有些数据不以\t分割；有些数据连续两个<><>作为头实体；
            triples = line.rstrip('.\n').split('\t')
            if len(triples) != 3:
                triples = line.rstrip('.\n').split()
            if len(triples) != 3:
                triples = rdf_patten.findall(line)
            if len(triples) == 6:
                head_ent, rel, tail_ent = triples[:3]
                yield parse_entities(head_ent)[0], parse_entities(rel)[0], parse_entities(tail_ent)[0]
                head_ent, rel, tail_ent = triples[3:]
                yield parse_entities(head_ent)[0], parse_entities(rel)[0], parse_entities(tail_ent)[0]
                continue
            head_ent, rel, tail_ent = triples
            #
            head_ents = parse_entities(head_ent)
            tail_ents = parse_entities(tail_ent)
            rels = parse_entities(rel)
            for head_ent in head_ents:
                for rel in rels:
                    for tail_ent in tail_ents:
                        yield head_ent, rel, tail_ent


def fit_triples():
    """entity,relation map to id；
        实体到id的映射词典
    """
    logging.info('fit_triples start ...')
    entities = set()
    relations = set()
    for head_ent, rel, tail_ent in iter_triples(tqdm_prefix='fit_triples '):
        entities.add(head_ent)
        entities.add(tail_ent)
        relations.add(rel)
    logging.info('entity2id start ...')
    entity2id = {ent: idx + 1 for idx, ent in enumerate(sorted(entities))}
    json_dump(entity2id, Config.entity2id)
    id2entity = {id: ent for ent, id in entity2id.items()}
    pkl_dump(id2entity, Config.id2entity_pkl)
    logging.info('relation2id start ...')
    relation2id = {rel: idx + 1 for idx, rel in enumerate(sorted(relations))}
    json_dump(relation2id, Config.relation2id)
    id2relation = {id: rel for rel, id in relation2id.items()}
    pkl_dump(id2relation, Config.id2relation_pkl)


def map_mention_entity():
    """mention2ent； 4 min； mention到实体的映射"""
    logging.info('map_mention_entity start ...')
    ent2mention = defaultdict(set)
    mention2ent = defaultdict(set)
    # count = 0
    for line in tqdm_iter_file(mention2ent_txt, prefix='iter mention2ent.txt '):
        mention, ent, rank = line.split('\t')  # 有部分数据有问题，mention为空字符串
        mention2ent[mention].add(ent)
        ent2mention[ent].add(mention)
        # if count == 13930117:
        #     print(line)
        #     import ipdb
        #     ipdb.set_trace()
        # count += 1
    #
    mention2ent = {k: sorted(v) for k, v in mention2ent.items()}
    json_dump(mention2ent, Config.mention2ent_json)
    ent2mention = {k: sorted(v) for k, v in ent2mention.items()}
    json_dump(ent2mention, Config.ent2mention_json)


def candidate_words():
    """实体的属性和类型映射字典；作为尾实体候选; 7min"""
    logging.info('candidate_words gen start ...')
    ent2types_dict = defaultdict(set)
    ent2attrs_dict = defaultdict(set)
    all_attrs = set()
    for head_ent, rel, tail_ent in iter_triples(tqdm_prefix='candidate_words '):
        if rel == '<类型>':
            ent2types_dict[head_ent].add(tail_ent)
        elif not head_ent.startswith('<'):
            ent2attrs_dict[tail_ent].add(head_ent)
            all_attrs.add(head_ent)
        elif not tail_ent.startswith('<'):
            ent2attrs_dict[head_ent].add(tail_ent)
            all_attrs.add(tail_ent)
    json_dump({ent: sorted(_types) for ent, _types in ent2types_dict.items()},
              Config.entity2types_json)
    json_dump({ent: sorted(attrs) for ent, attrs in ent2attrs_dict.items()},
              Config.entity2attrs_json)
    json_dump(list(all_attrs), Config.all_attrs_json)


def _get_top_counter():
    """26G，高频实体和mention，作为后期筛选和lac字典；
        统计实体和mention出现的次数； 方便取top作为最终自定义分词的词典；
    """
    logging.info('kb_count_top_dict start ...')
    if not (os.path.isfile(Config.entity2count_json) and
            os.path.isfile(Config.relation2count_json)):
        entities = []
        relations = []
        for head_ent, rel, tail_ent in iter_triples(tqdm_prefix='kb_top_count_dict '):
            entities.extend([head_ent, tail_ent])
            relations.append(rel)
        ent_counter = Counter(entities)
        json_dump(dict(ent_counter), Config.entity2count_json)
        del entities
        rel_counter = Counter(relations)
        del relations
        json_dump(dict(rel_counter), Config.relation2count_json)
    else:
        ent_counter = Counter(json_load(Config.entity2count_json))
        rel_counter = Counter(json_load(Config.relation2count_json))
    #
    if not os.path.isfile(Config.mention2count_json):
        mentions = []
        for line in tqdm_iter_file(mention2ent_txt, prefix='count_top_dict iter mention2ent.txt '):
            mention, ent, rank = line.split('\t')  # 有部分数据有问题，mention为空字符串
            mentions.append(mention)
        mention_counter = Counter(mentions)
        del mentions
        json_dump(dict(mention_counter), Config.mention2count_json)
    else:
        mention_counter = Counter(json_load(Config.mention2count_json))
    return ent_counter, rel_counter, mention_counter


def create_lac_custom_dict():
    """生成自定义分词词典"""
    logging.info('create_lac_custom_dict start...')
    ent_counter, rel_counter, mention_counter = _get_top_counter()
    mention_count = mention_counter.most_common(50 * 10000)  #
    customization_dict = {mention: 'MENTION' for mention, count in mention_count
                          if 2 <= len(mention) <= 8}
    del mention_count, mention_counter
    logging.info('create ent&rel customization_dict ...')
    _ent_pattrn = re.compile(r'["<](.*?)[>"]')
    # customization_dict.update({' '.join(_ent_pattrn.findall(rel)): 'REL'
    #                            for rel, count in rel_counter.most_common(10000)  # 10万
    #                            if 2 <= len(rel) <= 8})
    # del rel_counter
    customization_dict.update({' '.join(_ent_pattrn.findall(ent)): 'ENT'
                               for ent, count in ent_counter.most_common(50 * 10000)  # 100万
                               if 2 <= len(ent) <= 8})
    q_entity2id = json_load(Config.q_entity2id_json)
    q_entity2id.update(json_load(Config.a_entity2id_json))
    customization_dict.update({' '.join(_ent_pattrn.findall(ent)): 'ENT'
                               for ent, _id in q_entity2id.items()})
    del ent_counter
    with open(Config.lac_custom_dict_txt, 'w') as f:
        for e, t in customization_dict.items():
            if len(e) >= 3:
                f.write(f"{e}/{t}\n")
    logging.info('attr_custom_dict gen start ...')
    entity2attrs = json_load(Config.entity2attrs_json)
    all_attrs = set()
    for attrs in entity2attrs.values():
        all_attrs.update(attrs)
    name_patten = re.compile('"(.*?)"')
    with open(Config.lac_attr_custom_dict_txt, 'w') as f:
        for _attr in all_attrs:
            attr = ' '.join(name_patten.findall(_attr))
            if len(attr) >= 2:
                f.write(f"{attr}/ATTR\n")


def create_graph_csv():
    """  生成数据库导入文件
    cd /home/wangshengguang/neo4j-community-3.4.5/bin/
    ./neo4j-admin import --database=graph.db --nodes /home/wangshengguang/ccks-2020/data/graph_entity.csv  --relationships /home/wangshengguang/ccks-2020/data/graph_relation2.csv --ignore-duplicate-nodes=true --id-type INTEGER --ignore-missing-nodes=true
    CREATE CONSTRAINT ON (ent:Entity) ASSERT ent.id IS UNIQUE;
    CREATE CONSTRAINT ON (ent:Entity) ASSERT ent.name IS UNIQUE;
    CREATE CONSTRAINT ON (r:Relation) ASSERT r.name IS UNIQUE;
    """
    logging.info('start load Config.id2entity ..')
    entity2id = json_load(Config.entity2id)
    pd.DataFrame.from_records(
        [(id, ent, 'Entity') for ent, id in entity2id.items()],
        columns=["id:ID(Entity)", "name:String", ":LABEL"]).to_csv(
        Config.graph_entity_csv, index=False, encoding='utf_8_sig')
    #
    records = [[entity2id[head_name], entity2id[tail_name], 'Relation', rel_name]
               for head_name, rel_name, tail_name in iter_triples(tqdm_prefix='gen relation csv')]
    del entity2id
    gc.collect()
    pd.DataFrame.from_records(
        records, columns=[":START_ID(Entity)", ":END_ID(Entity)", ":TYPE", "name:String"]).to_csv(
        Config.graph_relation_csv, index=False, encoding='utf_8_sig')

# def kge_data():
#     """17.3G,生成KGE训练数据"""
#     logging.info('kge_data start ...')
#     kge_data_dir = Path(Config.entity2id).parent.joinpath('ccks-2020')
#     kge_data_dir.mkdir(exist_ok=True)
#
#     def data_df2file(data_df, save_path):
#         with open(save_path, 'w') as f:
#             f.write(f'{data_df.shape[0]}\n')
#         data_df.to_csv(save_path, mode='a', header=False, index=False, sep='\t')
#
#     #
#     ent_most = set([ent for ent, count in Counter(json_load(Config.entity2count_json)).most_common(1000000)])
#     entity2id = {ent: id + 1 for id, ent in enumerate(sorted(ent_most))}
#     data_df2file(data_df=pd.DataFrame([(ent, idx) for ent, idx in entity2id.items()]),
#                  save_path=kge_data_dir.joinpath('entity2id.txt'))
#     #
#     rel_most = set([rel for rel, _count in Counter(json_load(Config.relation2count_json)).most_common(1 * 200000)])
#     relation2id = {rel: id + 1 for id, rel in enumerate(sorted(rel_most))}
#     data_df2file(data_df=pd.DataFrame([(rel, idx) for rel, idx in relation2id.items()]),
#                  save_path=kge_data_dir.joinpath('relation2id.txt'))
#     # 训练集、验证集、测试集
#     logging.info('kge train data start ...')
#     # head,rel,tail -> head,tail,rel #HACK 交换了顺序，为了配合KGE模型的训练数据需要
#     data_df = pd.DataFrame([(entity2id.get(head_ent, 0), entity2id.get(tail_ent, 0),
#                              relation2id.get(rel, 0))
#                             for head_ent, rel, tail_ent in iter_triples()])
#     del entity2id, relation2id
#     train_df, test_df = train_test_split(data_df, test_size=0.01)
#     data_df2file(test_df, save_path=kge_data_dir.joinpath('test2id.txt'))
#     del test_df
#     train_df, valid_df = train_test_split(train_df, test_size=0.01)
#     data_df2file(valid_df, save_path=kge_data_dir.joinpath('valid2id.txt'))
#     del valid_df
#     data_df2file(train_df, save_path=kge_data_dir.joinpath('train2id.txt'))
#     del train_df
#     #
#     src_dir = str(kge_data_dir)
#     dst_dir = str(
#         Path(Config.entity2id).parent.parent.parent.joinpath('KE').joinpath('benchmarks').joinpath('ccks-2020'))
#     logging.info(f'move file: {src_dir} to {dst_dir}')
#     shutil.copytree(src=src_dir, dst=dst_dir)  # directory
