import argparse
import logging
import re
import traceback

import pandas as pd
from tqdm import tqdm

from ckbqa.utils.logger import logging_config
from config import valid_question_txt, ResultSaver


def train_answer():
    logging_config('train_answer.log', stream_log=True)
    from ckbqa.qa.qa import QA
    from ckbqa.dataset.data_prepare import load_data, question_patten, entity_pattern, attr_pattern
    from ckbqa.qa.evaluation_matrics import get_metrics
    #
    logging.info('* start run ...')
    qa = QA()
    data = []
    for question, sparql, answer in load_data(tqdm_prefix='test qa'):
        print('\n' * 2)
        print('*****' * 20)
        print(f" question  : {question}")
        print(f" sparql    : {sparql}")
        print(f" standard answer   : {answer}")
        q_text = question_patten.findall(question)[0]
        subject_entities = entity_pattern.findall(sparql) + attr_pattern.findall(sparql)
        answer_entities = entity_pattern.findall(answer) + attr_pattern.findall(answer)
        try:
            (result_entities, candidate_entities,
             candidate_out_paths, candidate_in_paths) = qa.run(q_text, return_candidates=True)
        except:
            logging.info(traceback.format_exc())
            result_entities = []
            candidate_entities = []
        precision, recall, f1 = get_metrics(subject_entities, candidate_entities)
        if recall == 0 or len(set(answer_entities) & set(candidate_entities)) == 0:
            print(f"question: {question}\n"
                  f"subject_entities: {subject_entities}, candidate_entities: {candidate_entities}"
                  f"precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}\n\n")
        # import ipdb
        # ipdb.set_trace()
        data.append([question, subject_entities, list(candidate_entities), answer_entities, result_entities])
    data_df = pd.DataFrame(data, columns=['question', 'subject_entities', 'candidate_entities',
                                          'answer_entities', 'result_entities'])
    data_df.to_csv(ResultSaver().train_result_csv, index=False, encoding='utf_8_sig')


def train_evaluate():
    logging_config('train_evaluate.log', stream_log=True)
    from ckbqa.qa.evaluation_matrics import get_metrics
    #
    partten = re.compile(r'["<](.*?)[>"]')
    #
    _paths = ResultSaver(new_path=False).train_result_csv
    print(_paths)
    train_df = pd.read_csv(_paths[0])
    precisions, recalls, f1_scores = [], [], []
    for index, row in tqdm(train_df.iterrows(), total=train_df.shape[0], desc='evaluate '):
        subject_entities = partten.findall(row['subject_entities'])  # 匹配文字
        if not subject_entities:
            subject_entities = eval(row['subject_entities'])
        # 修复之前把实体<>去掉造成的问题；问题解析时去掉，但预测时未去掉；
        # 所以需要匹配文字，不匹配 <>, ""
        candidate_entities = eval(row['candidate_entities']) + partten.findall(row['candidate_entities'])
        precision, recall, f1 = get_metrics(subject_entities, candidate_entities)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        # print(f"question: {row['question']}\n"
        #       f"subject_entities: {subject_entities}, candidate_entities: {candidate_entities}"
        #       f"precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}\n\n")
        # import time
        # time.sleep(2)
    ave_precision = sum(precisions) / len(precisions)
    ave_recall = sum(recalls) / len(recalls)
    ave_f1_score = sum(f1_scores) / len(f1_scores)
    print(f"ave_precision: {ave_precision:.3f}, "
          f"ave_recall: {ave_recall:.3f}, "
          f"ave_f1_score:{ave_f1_score:.3f}")


def valid_answer():
    """验证数据答案"""
    logging_config('valid_answer.log', stream_log=True)
    from ckbqa.qa.qa import QA
    from ckbqa.dataset.data_prepare import question_patten
    #
    data_df = pd.DataFrame([question.strip() for question in open(valid_question_txt, 'r', encoding='utf-8')],
                           columns=['question'])
    logging.info(f"data_df.shape: {data_df.shape}")
    qa = QA()
    valid_datas = {'question': [], 'result': []}
    with open(ResultSaver().submmit_result_txt, 'w', encoding='utf-8') as f:
        for index, row in tqdm(data_df.iterrows(), total=data_df.shape[0], desc='qa answer'):
            question = row['question']
            q_text = question_patten.findall(question)[0]
            try:
                answer_entities = qa.run(q_text)
                answer_entities = [ent if ent.startswith('<') else f'"{ent}"' for ent in answer_entities]
                f.write('\t'.join(answer_entities) + '\n')
            except:
                answer_entities = []
                f.write('""\n')
                logging.info(traceback.format_exc())
            f.flush()
            valid_datas['question'].append(question)
            valid_datas['result'].append(answer_entities)
    pd.DataFrame(valid_datas).to_csv(ResultSaver().valid_result_csv, index=False, encoding='utf_8_sig')


def valid2submmit():
    '''
        验证数据输出转化为答案提交
    '''
    logging_config('valid2submmit.log', stream_log=True)
    data_df = pd.read_csv(ResultSaver().valid_result_csv)
    with open(ResultSaver().submmit_result_txt, 'w', encoding='utf-8') as f:
        for index, row in data_df.iterrows():
            ents = []
            for ent in eval(row['result']):
                if ent.startswith('<'):
                    ents.append(ent)
                elif ent.startswith('"'):
                    ent = ent.strip('"')
                    ents.append(f'"{ent}"')
            if ents:
                f.write('\t'.join(ents) + '\n')
            else:
                f.write('""\n')
            f.flush()


def main():
    parser = argparse.ArgumentParser(description="基础，通用parser")
    # logging config 日志配置
    parser.add_argument('--stream_log', action="store_true", help="是否将日志信息输出到标准输出")  # log print到屏幕
    #
    group = parser.add_mutually_exclusive_group(required=True)  # 一组互斥参数,且至少需要互斥参数中的一个

    group.add_argument('--train_answer', action="store_true", help="训练集答案")
    group.add_argument('--train_evaluate', action="store_true", help="验证数据指标评价")
    group.add_argument('--valid_answer', action="store_true", help="验证数据（待提交数据）答案")
    group.add_argument('--valid2submmit', action="store_true", help="验证数据处理后提交")
    group.add_argument('--task', action="store_true", help="临时组织的任务，调用多个函数")
    # parse args
    args = parser.parse_args()
    #
    # from ckbqa.utils.tools import ProcessManager
    # ProcessManager().run()
    if args.train_answer:
        train_answer()
    elif args.train_evaluate:
        train_evaluate()
    elif args.valid_answer:
        valid_answer()
    elif args.valid2submmit:
        valid2submmit()
    elif args.task:
        task()


def task():
    logging_config('qa_task.log', stream_log=True)
    train_answer()
    valid_answer()


if __name__ == '__main__':
    """
    example:
        nohup  python qa.py --train_answer &>train_answer.out & 
        nohup  python qa.py --train_evaluate &>train_evaluate.out & 
        nohup  python qa.py --valid_answer &>valid_answer.out & 
        nohup  python qa.py --valid2submmit &>valid2submmit.out & 
        nohup  python qa.py --task &>qa_task.out & 
    """
    main()
