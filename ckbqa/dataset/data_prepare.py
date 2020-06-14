import re

import pandas as pd
from tqdm import tqdm

from ckbqa.utils.tools import json_dump
from config import DataConfig, raw_train_txt

entity_pattern = re.compile(r'(<.*?>)')  # 保留<>
attr_pattern = re.compile(r'"(.*?)"')  # 不保留"", "在json key中不便保存
question_patten = re.compile(r'q\d{1,4}:(.*)')

PAD = 0
UNK = 1


def load_data(tqdm_prefix=''):
    with open(raw_train_txt, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    for q, sparql, a in tqdm([lines[i:i + 3] for i in range(0, len(lines), 3)],
                             desc=tqdm_prefix):
        yield q, sparql, a


def fit_on_texts():
    q_entities_set = set()
    a_entities_set = set()
    words_set = set()
    for q, sparql, a in load_data():
        a_entities = entity_pattern.findall(a)
        q_entities = entity_pattern.findall(sparql)
        q_text = question_patten.findall(q)
        entities = a_entities + q_entities
        words = list(q_text[0]) + [e for ent in entities for e in ent]

        q_entities_set.update(q_entities)
        a_entities_set.update(a_entities)
        words_set.update(words)
    word2id = {"PAD": PAD, "UNK": UNK}
    for w in sorted(words_set):
        word2id[w] = len(word2id)
    json_dump(word2id, DataConfig.word2id_json)
    # question entity
    q_entity2id = {"PAD": PAD, "UNK": UNK}
    for e in sorted(q_entities_set):
        q_entity2id[e] = len(q_entity2id)
    json_dump(q_entity2id, DataConfig.q_entity2id_json)
    # answer entity
    a_entity2id = {"PAD": PAD, "UNK": UNK}
    for e in sorted(a_entities_set):
        a_entity2id[e] = len(a_entity2id)
    json_dump(a_entity2id, DataConfig.a_entity2id_json)


def data_convert():
    """转化原始数据格式, 方便采样entity"""
    data = {'question': [], 'q_entities': [], 'q_strs': [],
            'a_entities': [], 'a_strs': []}
    for q, sparql, a in load_data():
        q_text = question_patten.findall(q)[0]
        q_entities = entity_pattern.findall(sparql)
        q_strs = attr_pattern.findall(sparql)
        a_entities = entity_pattern.findall(a)
        a_strs = attr_pattern.findall(a)
        data['question'].append(q_text)
        data['q_entities'].append(q_entities)
        data['q_strs'].append(q_strs)
        data['a_entities'].append(a_entities)
        data['a_strs'].append(a_strs)
    pd.DataFrame(data).to_csv(DataConfig.data_csv, encoding='utf_8_sig', index=False)
