import re

import hanlp
import jieba


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

from ckbqa.dataset.data_prepare import load_data, question_patten
from ckbqa.utils.logger import logging_config
from ckbqa.utils.tools import json_load

#
logging_config('test.log', stream_log=True)

from config import Config


def ner_train_data():
    ent2mention = json_load(Config.ent2mention_json)
    # recognizer = hanlp.load(hanlp.pretrained.ner.MSRA_NER_BERT_BASE_ZH)
    tokenizer = hanlp.load('PKU_NAME_MERGED_SIX_MONTHS_CONVSEG')
    recognizer = hanlp.load(hanlp.pretrained.ner.MSRA_NER_BERT_BASE_ZH)
    from LAC import LAC
    # 装载LAC模型
    lac = LAC(mode='lac')
    jieba.enable_paddle()  # 启动paddle模式。 0.40版之后开始支持，早期版本不支持
    _ent_patten = re.compile(r'["<](.*?)[>"]')
    for q, sparql, a in load_data():
        q_text = question_patten.findall(q)[0]
        hanlp_entities = recognizer([list(q_text)])
        hanlp_words = tokenizer(q_text)
        lac_results = lac.run(q_text)
        q_entities = _ent_patten.findall(sparql)
        jieba_results = list(jieba.cut_for_search(q_text))
        mentions = [ent2mention.get(ent) for ent in q_entities]
        print(f"q_text: {q_text}\nq_entities: {q_entities}, "
              f"\nlac_results:{lac_results}"
              f"\nhanlp_words: {hanlp_words}, "
              f"\njieba_results: {jieba_results}, "
              f"\nhanlp_entities: {hanlp_entities}, "
              f"\nmentions: {mentions}")
        import ipdb
        ipdb.set_trace()


if __name__ == '__main__':
    ner_train_data()
