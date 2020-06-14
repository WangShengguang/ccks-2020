import argparse
import re

import pandas as pd
from tqdm import tqdm

from ckbqa.utils.logger import logging_config
from config import ResultSaver


def train_answer_data():
    logging_config('train_evaluate.log', stream_log=True)
    from ckbqa.qa.evaluation_matrics import get_metrics
    #
    partten = re.compile(r'["<](.*?)[>"]')
    #
    _paths = ResultSaver(new_path=False).train_result_csv
    print(_paths)
    train_df = pd.read_csv(_paths[0])
    ceg_precisions, ceg_recalls, ceg_f1_scores = [], [], []
    answer_precisions, answer_recalls, answer_f1_scores = [], [], []
    for index, row in tqdm(train_df.iterrows(), total=train_df.shape[0], desc='evaluate '):
        subject_entities = partten.findall(row['subject_entities'])  # 匹配文字
        if not subject_entities:
            subject_entities = eval(row['subject_entities'])
        # 修复之前把实体<>去掉造成的问题；问题解析时去掉，但预测时未去掉；
        # 所以需要匹配文字，不匹配 <>, ""
        # CEG  Candidate Entity Generation
        candidate_entities = eval(row['candidate_entities']) + partten.findall(row['candidate_entities'])
        precision, recall, f1 = get_metrics(subject_entities, candidate_entities)
        ceg_precisions.append(precision)
        ceg_recalls.append(recall)
        ceg_f1_scores.append(f1)
        # Answer
        standard_entities = eval(row['answer_entities'])
        result_entities = eval(row['result_entities'])
        precision, recall, f1 = get_metrics(standard_entities, result_entities)
        answer_precisions.append(precision)
        answer_recalls.append(recall)
        answer_f1_scores.append(f1)
        #
        # print(f"question: {row['question']}\n"
        #       f"subject_entities: {subject_entities}, candidate_entities: {candidate_entities}"
        #       f"precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}\n\n")
        # import time
        # time.sleep(2)
    ave_ceg_precision = sum(ceg_precisions) / len(ceg_precisions)
    ave_ceg_recall = sum(ceg_recalls) / len(ceg_recalls)
    ave_ceg_f1_score = sum(ceg_f1_scores) / len(ceg_f1_scores)
    print(f"ave_ceg_precision: {ave_ceg_precision:.3f}, "
          f"ave_ceg_recall: {ave_ceg_recall:.3f}, "
          f"ave_ceg_f1_score:{ave_ceg_f1_score:.3f}")
    #
    ave_answer_precision = sum(answer_precisions) / len(answer_precisions)
    ave_answer_recall = sum(answer_recalls) / len(answer_recalls)
    ave_answer_f1_score = sum(answer_f1_scores) / len(answer_f1_scores)
    print(f"ave_answer_precision: {ave_answer_precision:.3f}, "
          f"ave_answer_recall: {ave_answer_recall:.3f}, "
          f"ave_answer_f1_score:{ave_answer_f1_score:.3f}")


def ceg():
    logging_config('train_evaluate.log', stream_log=True)
    from ckbqa.qa.evaluation_matrics import get_metrics
    from ckbqa.qa.el import CEG
    from ckbqa.dataset.data_prepare import load_data, question_patten, entity_pattern, attr_pattern  #
    ceg = CEG()  # Candidate Entity Generation
    ceg_precisions, ceg_recalls, ceg_f1_scores = [], [], []
    ceg_csv = "./ceg.csv"
    data = []
    for q, sparql, a in load_data(tqdm_prefix='ceg evaluate '):
        q_entities = entity_pattern.findall(sparql) + attr_pattern.findall(sparql)

        q_text = ''.join(question_patten.findall(q))
        # 修复之前把实体<>去掉造成的问题；问题解析时去掉，但预测时未去掉；
        # 所以需要匹配文字，不匹配 <>, ""
        ent2mention = ceg.get_ent2mention(q_text)
        # CEG  Candidate Entity Generation
        precision, recall, f1 = get_metrics(q_entities, ent2mention)
        ceg_precisions.append(precision)
        ceg_recalls.append(recall)
        ceg_f1_scores.append(f1)
        #

        #
        data.append([q, q_entities, list(ent2mention.keys())])
        if recall == 0:
            # ceg.memory.entity2id
            # ceg.memory.mention2entity
            print(f"question: {q}\n"
                  f"subject_entities: {q_entities}, candidate_entities: {ent2mention}"
                  f"precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}\n\n")
            import ipdb
            ipdb.set_trace()
            print('\n\n')
        # import time
        # time.sleep(2)
    pd.DataFrame(data, columns=['question', 'q_entities', 'ceg']).to_csv(
        ceg_csv, index=False, encoding='utf_8_sig')
    ave_precision = sum(ceg_precisions) / len(ceg_precisions)
    ave_recall = sum(ceg_recalls) / len(ceg_recalls)
    ave_f1_score = sum(ceg_f1_scores) / len(ceg_f1_scores)
    print(f"ave_precision: {ave_precision:.3f}, "
          f"ave_recall: {ave_recall:.3f}, "
          f"ave_f1_score:{ave_f1_score:.3f}")


def main():
    parser = argparse.ArgumentParser(description="基础，通用parser")
    # logging config 日志配置
    parser.add_argument('--stream_log', action="store_true", help="是否将日志信息输出到标准输出")  # log print到屏幕
    #
    group = parser.add_mutually_exclusive_group(required=True)  # 一组互斥参数,且至少需要互斥参数中的一个

    group.add_argument('--ceg', action="store_true", help="ceg Candidate Entity Generation评价")
    group.add_argument('--train_answer_data', action="store_true", help="train_answer_data评价")
    # parse args
    args = parser.parse_args()
    #
    # from ckbqa.utils.tools import ProcessManager
    # ProcessManager().run()
    if args.ceg:
        ceg()
    elif args.train_answer_data:
        train_answer_data()
    elif args.task:
        task()


def task():
    logging_config('ceg.log', stream_log=True)
    ceg()


if __name__ == '__main__':
    """
    example:
        nohup  python qa.py --train_evaluate_data &>train_evaluate_data.out & 
        nohup  python qa.py --ceg &>ceg.out & 
    """
    # from ckbqa.utils.tools import ProcessManager #实时查看内存占用情况
    # ProcessManager().run()
    main()
