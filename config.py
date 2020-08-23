import logging
import os
from pathlib import Path

import arrow
import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = cur_dir
data_dir = os.path.join(root_dir, "data")
raw_data_dir = os.path.join(data_dir, 'raw_data')

output_dir = os.path.join(root_dir, "output")
ckpt_dir = os.path.join(output_dir, "ckpt")
result_dir = os.path.join(output_dir, 'result')

# 原始数据
mention2ent_txt = os.path.join(raw_data_dir, 'PKUBASE', 'pkubase-mention2ent.txt')
kb_triples_txt = os.path.join(raw_data_dir, 'PKUBASE', 'pkubase-complete2.txt')
# 问答原始数据
raw_train_txt = os.path.join(raw_data_dir, 'ccks_2020_7_4_Data', 'task1-4_train_2020.txt')
valid_question_txt = os.path.join(raw_data_dir, 'ccks_2020_7_4_Data', 'task1-4_valid_2020.questions')


class DataConfig(object):
    """
    原始数据经过处理后生成的数据
    """
    word2id_json = os.path.join(data_dir, 'word2id.json')
    q_entity2id_json = os.path.join(data_dir, 'q_entity2id.json')
    a_entity2id_json = os.path.join(data_dir, 'a_entity2id.json')
    #
    data_csv = os.path.join(data_dir, 'data.csv')  # 训练数据做了一点格式转换
    #
    mention2ent_json = os.path.join(data_dir, 'mention2ent.json')
    ent2mention_json = os.path.join(data_dir, 'ent2mention.json')
    entity2id = os.path.join(data_dir, 'entity2id.json')
    id2entity_pkl = os.path.join(data_dir, 'id2entity.pkl')
    relation2id = os.path.join(data_dir, 'relation2id.json')
    id2relation_pkl = os.path.join(data_dir, 'id2relation.pkl')
    # count
    entity2count_json = os.path.join(data_dir, 'entity2count.json')
    relation2count_json = os.path.join(data_dir, 'relation2count.json')
    mention2count_json = os.path.join(data_dir, 'mention2count.json')
    #
    lac_custom_dict_txt = os.path.join(data_dir, 'lac_custom_dict.txt')
    lac_attr_custom_dict_txt = os.path.join(data_dir, 'lac_attr_custom_dict.txt')
    jieba_custom_dict = os.path.join(data_dir, 'jieba_custom_dict.json')
    # graph_pkl = os.path.join(data_dir, 'graph.pkl')
    graph_entity_csv = os.path.join(data_dir, 'graph_entity.csv')  # 图谱导入
    graph_relation_csv = os.path.join(data_dir, 'graph_relation.csv')  # 图谱导入
    entity2types_json = os.path.join(data_dir, 'entity2type.json')
    entity2attrs_json = os.path.join(data_dir, 'entity2attr.json')
    all_attrs_json = os.path.join(data_dir, 'all_attrs.json')  # 所有属性
    #
    lac_model_pkl = os.path.join(data_dir, 'lac_model.pkl')
    # EntityScore model
    entity_score_model_pkl = os.path.join(data_dir, 'entity_score_model.pkl')
    entity_score_data_pkl = os.path.join(data_dir, 'entity_score_data.pkl')
    #
    neo4j_query_cache = os.path.join(data_dir, 'neo4j_query_cache.json')

    #
    relation_score_sample_csv = os.path.join(data_dir, 'sample.csv')

    @staticmethod
    def get_relation_score_sample_csv(data_type, neg_rate):
        if data_type == 'train':
            file_path = Path(DataConfig.relation_score_sample_csv).with_name(f'train.1_{neg_rate}.csv')
        else:
            file_path = Path(DataConfig.relation_score_sample_csv).with_name('test.csv')
        return str(file_path)


class TorchConfig(object):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    # device = "cpu"
    gpu_nums = torch.cuda.device_count()
    multi_gpu = True
    gradient_accumulation_steps = 1
    clip_grad = 2


class Parms(object):
    #
    learning_rate = 0.001
    # #
    # min_epoch_nums = 1
    # max_epoch_nums = 10
    # #
    # embedding_dim = 128  # entity enbedding dim, relation enbedding dim , word enbedding dim
    max_len = 50  # max sentence length
    # batch_size = 32
    # # subtask = 'general'
    # test_batch_size = 128


class Config(TorchConfig, DataConfig, Parms):
    pretrained_model_name_or_path = os.path.join(data_dir, 'bert-base-chinese-pytorch')  # 'bert-base-chinese'
    # load_pretrain = True
    # rand_seed = 1234
    # load_model_mode = "min_loss"
    # load_model_mode = "max_step"
    # load_model_mode = "max_acc"  # mrr
    #
    # train_count = 1000  # TODO for debug
    # test_count = 10  # 10*2*13589


class ResultSaver(object):
    """输出文件管理；自动生成新文件名；避免覆盖
        自动查找已存在的文件
    """

    def __init__(self, find_exist_path=False):
        os.makedirs(result_dir, exist_ok=True)
        self.find_exist_path = find_exist_path

    def _get_new_path(self, file_name):
        date_str = arrow.now().format("YYYYMMDD")
        # date_str = '20200609' #临时修改
        num = 1
        path = os.path.join(result_dir, f"{date_str}-{num}-{file_name}")
        while os.path.isfile(path):
            path = os.path.join(result_dir, f"{date_str}-{num}-{file_name}")
            num += 1
        return path

    def _find_paths(self, file_name):
        paths = [str(_path) for _path in
                 Path(result_dir).rglob(f'*{file_name}')]
        _paths = sorted(paths, reverse=True)
        return _paths

    def get_path(self, file_name):
        if self.find_exist_path:
            path = self._find_paths(file_name)
        else:
            path = self._get_new_path(file_name)
        logging.info(f'* get path: {path}')
        return path

    @property
    def train_result_csv(self):
        file_name = 'train_answer_result.csv'
        path = self.get_path(file_name)
        return path

    @property
    def valid_result_csv(self):
        file_name = 'valid_result.csv'
        path = self.get_path(file_name)
        return path

    @property
    def submit_result_txt(self):
        file_name = 'submit_result.txt'
        path = self.get_path(file_name)
        return path
