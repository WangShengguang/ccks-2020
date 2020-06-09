import os
from pathlib import Path

import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = cur_dir
data_dir = os.path.join(root_dir, "data")
raw_data_dir = os.path.join(data_dir, 'raw_data')

output_dir = os.path.join(root_dir, "output")
ckpt_dir = os.path.join(output_dir, "ckpt")

# 原始数据
mention2ent_txt = os.path.join(raw_data_dir, 'PKUBASE', 'pkubase-mention2ent.txt')
kb_triples_txt = os.path.join(raw_data_dir, 'PKUBASE', 'pkubase-complete.txt')
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
    sample_csv = os.path.join(data_dir, 'sample.csv')
    #
    data_csv = os.path.join(data_dir, 'data.csv')
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
    # graph_pkl = os.path.join(data_dir, 'graph.pkl')
    graph_entity_csv = os.path.join(data_dir, 'graph_entity.csv')
    graph_relation_csv = os.path.join(data_dir, 'graph_relation.csv')
    entity2types_json = os.path.join(data_dir, 'entity2type.json')
    entity2attrs_json = os.path.join(data_dir, 'entity2attr.json')
    all_attrs_json = os.path.join(data_dir, 'all_attrs.json')
    #
    lac_model_pkl = os.path.join(data_dir, 'lac_model.pkl')
    entity_score_model_pkl = os.path.join(data_dir, 'entity_score_model.pkl')
    #
    neo4j_query_cache = os.path.join(data_dir, 'neo4j_query_cache.json')

    @staticmethod
    def get_sample_csv_path(data_type, neg_rate):
        if data_type == 'train':
            file_path = Path(DataConfig.sample_csv).with_name(f'train.1_{neg_rate}.csv')
        else:
            file_path = Path(DataConfig.sample_csv).with_name('test.csv')
        return str(file_path)


class TorchConfig(object):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = "cpu"
    gpu_nums = torch.cuda.device_count()
    multi_gpu = True
    gradient_accumulation_steps = 1
    clip_grad = 2


class Parms(object):
    #
    learning_rate = 0.001
    #
    min_epoch_nums = 50
    max_epoch_nums = 200
    #
    embedding_dim = 128  # entity enbedding dim, relation enbedding dim , word enbedding dim
    max_len = 50  # max sentence length
    batch_size = 32
    # subtask = 'general'
    test_batch_size = 128
    lr = 0.0001


class Config(TorchConfig, DataConfig, Parms):
    pretrained_model_name_or_path = os.path.join(data_dir, 'bert-base-chinese-pytorch')  # 'bert-base-chinese'
    load_pretrain = True
    rand_seed = 1234
    # load_model_mode = "min_loss"
    # load_model_mode = "max_step"
    load_model_mode = "max_acc"  # mrr
    #
    # train_count = 1000  # TODO for debug
    test_count = 10  # 10*2*13589


__all__ = ['Config']
