import random
import re

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from ckbqa.utils.tools import json_dump, json_load
from config import DataConfig, raw_train_txt

entity_patten = re.compile(r'<(.*?)>')
string_patten = re.compile(r'"(.*?)"')
question_patten = re.compile('q\d{1,4}:(.*)')
# subject_patten = re.compile(r'["<](.*?)[>"]')  # 不要过多预处理

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
        a_entities = entity_patten.findall(a)
        q_entities = entity_patten.findall(sparql)
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
    """转化原始数据格式，方便采样entity"""
    data = {'question': [], 'q_entities': [], 'q_strs': [],
            'a_entities': [], 'a_strs': []}
    for q, sparql, a in load_data():
        q_text = question_patten.findall(q)[0]
        q_entities = entity_patten.findall(sparql)
        q_strs = string_patten.findall(sparql)
        a_entities = entity_patten.findall(a)
        a_strs = string_patten.findall(a)
        data['question'].append(q_text)
        data['q_entities'].append(q_entities)
        data['q_strs'].append(q_strs)
        data['a_entities'].append(a_entities)
        data['a_strs'].append(a_strs)
    pd.DataFrame(data).to_csv(DataConfig.data_csv, encoding='utf_8_sig', index=False)


def data2samples(neg_rate=3, test_size=0.1):
    questions = []
    sim_questions = []
    labels = []
    all_relations = list(json_load(DataConfig.relation2id))
    _entity_patten = re.compile(r'["<](.*?)[>"]')
    for q, sparql, a in load_data(tqdm_prefix='data2samples '):
        q_text = question_patten.findall(q)[0]
        q_entities = _entity_patten.findall(sparql)
        questions.append(q_text)
        sim_questions.append('的'.join(q_entities))
        labels.append(1)
        #
        for neg_relation in random.sample(all_relations, neg_rate):
            questions.append(q_text)
            neg_question = '的'.join(q_entities[:-1] + [neg_relation])  # 随机替换 <关系>
            sim_questions.append(neg_question)
            labels.append(0)
    data_df = pd.DataFrame({'question': questions, 'sim_question': sim_questions, 'label': labels})
    data_df.to_csv(DataConfig.sample_csv, encoding='utf_8_sig', index=False)
    train_df, test_df = train_test_split(data_df, test_size=test_size)
    test_df.to_csv(DataConfig.get_sample_csv_path('test', neg_rate), encoding='utf_8_sig', index=False)
    train_df.to_csv(DataConfig.get_sample_csv_path('train', neg_rate), encoding='utf_8_sig', index=False)
