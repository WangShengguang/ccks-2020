import logging
import traceback
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from ckbqa.utils.logger import logging_config

logging_config('qa.log', stream_log=True)

from ckbqa.dataset.data_prepare import load_data, question_patten, entity_patten, string_patten
from ckbqa.qa.qa import QA
from config import valid_question_txt


def run():
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
        subject_entities = entity_patten.findall(sparql) + string_patten.findall(sparql)
        answer_entities = entity_patten.findall(answer) + string_patten.findall(answer)
        try:
            result_entities = qa.run(q_text)
        except:
            logging.info(traceback.format_exc())
            result_entities = []
        data.append([question, subject_entities, answer_entities, result_entities])
    data_df = pd.DataFrame(data, columns=['question', 'subject_entities',
                                          'answer_entities', 'result_entities'])
    date_str = str(datetime.today())[:10]
    data_df.to_csv(f'./{date_str}-train_answer_result.csv', index=False, encoding='utf_8_sig')


def get_answer():
    data_df = pd.DataFrame([question.strip() for question in open(valid_question_txt, 'r', encoding='utf-8')],
                           columns=['question'])
    logging.info(f"data_df.shape: {data_df.shape}")
    qa = QA()
    date_str = str(datetime.today())[:10]
    valid_datas = {'question': [], 'result': []}
    with open(f'./{date_str}-result.txt', 'w', encoding='utf-8') as f:
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
            # logging.info(f'q_text: {row["q_text"]}, \nanswer: {answer_entities}') #内部已有
    pd.DataFrame(valid_datas).to_csv('./valid_result.csv', index=False, encoding='utf_8_sig')


def main():
    print('* start run ...')
    get_answer()
    run()


if __name__ == '__main__':
    """
    example:
        nohup  python qa.py &>qa.out & 
    """
    main()
