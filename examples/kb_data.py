def add_root_path():
    import sys
    from pathlib import Path
    cur_dir = Path(__file__).absolute().parent
    times = 10
    while not str(cur_dir).endswith('ccks-2020') and times:
        cur_dir = cur_dir.parent
        times -= 1
    print(cur_dir)
    sys.path.append(str(cur_dir))


add_root_path()

import re
from collections import Counter

import pandas as pd

from ckbqa.dataset.data_prepare import load_data
from ckbqa.dataset.kb_data_prepare import iter_triples
from ckbqa.utils.tools import json_load, json_dump
from config import mention2ent_txt


def triples2csv():
    """26G"""
    data_df = pd.DataFrame([(head_ent, rel, tail_ent) for head_ent, rel, tail_ent in iter_triples()],
                           columns=['head', 'rel', 'tail'])
    data_df.to_csv('./triples.csv', encoding='utf_8_sig', index=False)


def most_ents():
    """26G"""

    entities = []
    for head_ent, rel, tail_ent in iter_triples():
        entities.extend([head_ent, rel, tail_ent])
    print(f' len(entities): {len(entities)}')
    ent_counter = Counter(entities)
    del entities
    ent_count = ent_counter.most_common(1000000)  # 百万
    print(ent_count[:10])
    print(ent_count[-10:])
    json_dump(dict(ent_counter), 'ent2count.json')
    #
    print('-----' * 10)
    mentions = []
    with open(mention2ent_txt, 'r', encoding='utf-8') as f:
        for line in f:
            mention, ent, rank = line.split('\t')  # 有部分数据有问题，mention为空字符串
            mentions.append(mention)
    print(f"len(mentions): {len(mentions)}")
    mention_counter = Counter(mentions)
    del mentions
    mention_count = mention_counter.most_common(1000000)  # 百万
    print(mention_count[:10])
    print(mention_count[-10:])
    json_dump(dict(mention_counter), 'mention2count.json')

    import ipdb
    ipdb.set_trace()


def lac_test():
    from LAC import LAC
    print('start ...')
    mention_count = Counter(json_load('mention2count.json')).most_common(10000)
    customization_dict = {mention: 'MENTION' for mention, count in mention_count
                          if len(mention) >= 2}
    print('second ...')
    ent_count = Counter(json_load('ent2count.json')).most_common(300000)
    ent_pattrn = re.compile(r'["<](.*?)[>"]')
    customization_dict.update({' '.join(ent_pattrn.findall(ent)): 'ENT' for ent, count in ent_count})
    with open('./customization_dict.txt', 'w') as f:
        f.write('\n'.join([f"{e}/{t}" for e, t in customization_dict.items()
                           if len(e) >= 3]))
    import time
    before = time.time()
    print(f'before ...{before}')
    lac = LAC(mode='lac')
    lac.load_customization('./customization_dict.txt')  # 20万21s;30万47s
    lac_raw = LAC(mode='lac')
    print(f'after ...{time.time() - before}')
    ##
    test_count = 10
    for q, sparql, a in load_data():
        q_text = q.split(':')[1]
        print('---' * 10)
        print(q_text)
        print(sparql)
        print(a)
        words, tags = lac_raw.run(q_text)
        print(list(zip(words, tags)))
        words, tags = lac.run(q_text)
        print(list(zip(words, tags)))
        if not test_count:
            break
        test_count -= 1
    import ipdb
    ipdb.set_trace()


class Node(object):
    def __init__(self, data):
        self.data = data
        self.ins = set()  # 入度节点，关系和实体统一处理
        self.outs = set()  # 出度节点，关系实体统一处理


def create_graph():
    nodes = {}
    for head_ent, rel, tail_ent in iter_triples():
        if head_ent not in nodes:
            head_node = Node(head_ent)
        else:
            head_node = nodes[head_ent]
        head_node.outs.add(rel)
        #
        if rel not in nodes:
            rel_node = Node(rel)
        else:
            rel_node = nodes[rel]
        rel_node.ins.add(head_ent)
        rel_node.outs.add(tail_ent)
        #
        if tail_ent not in nodes:
            tail_node = Node(tail_ent)
        else:
            tail_node = nodes[tail_ent]
        tail_node.ins.add(rel)


if __name__ == '__main__':
    # triples2csv()
    # most_ents()
    lac_test()
