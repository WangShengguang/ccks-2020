import argparse

from ckbqa.utils.logger import logging_config


def create_db_tabels():
    from sqlalchemy_utils import database_exists, create_database
    from ckbqa.dao.db import sqlite_db_engine
    if not database_exists(sqlite_db_engine.url):
        create_database(sqlite_db_engine.url)
    from ckbqa.dao.sqlite_models import BaseModel
    BaseModel.metadata.create_all(sqlite_db_engine)  # 创建表


def data_prepare():
    logging_config('data_prepare.log', stream_log=True)
    from ckbqa.dataset.data_prepare import data2samples, fit_on_texts, data_convert
    # map_mention_entity()
    data2samples(neg_rate=3)
    data_convert()
    fit_on_texts()
    create_db_tabels()


def kb_data_prepare():
    logging_config('kb_data_prepare.log', stream_log=True)
    from ckbqa.dataset.kb_data_prepare import (create_lac_custom_dict, candidate_words, fit_triples)
    from ckbqa.dataset.kb_data_prepare import create_graph_csv
    fit_triples()  # 生成字典
    candidate_words()  # 属性
    create_lac_custom_dict()  # 自定义分词词典

    create_graph_csv()  # 生成数据库导入文件
    from examples.lac_test import lac_model
    lac_model()


def main():
    parser = argparse.ArgumentParser(description="基础，通用parser")
    # logging config 日志配置
    parser.add_argument('--stream_log', action="store_true", help="是否将日志信息输出到标准输出")  # log print到屏幕
    #
    group = parser.add_mutually_exclusive_group(required=True)  # 一组互斥参数,且至少需要互斥参数中的一个

    group.add_argument('--data_prepare', action="store_true", help="训练集数据预处理")
    group.add_argument('--kb_data_prepare', action="store_true", help="知识库数据预处理")
    group.add_argument('--task', action="store_true", help="临时组织的任务，调用多个函数")
    # parse args
    args = parser.parse_args()
    #
    # from ckbqa.utils.tools import ProcessManager
    # ProcessManager().run()
    if args.data_prepare:
        data_prepare()
    elif args.kb_data_prepare:
        kb_data_prepare()
    elif args.task:
        task()


def task():
    pass


if __name__ == '__main__':
    """ 代码执行入口
    examples:
        python manage.py --data_prepare
        nohup python manage.py --kb_data_prepare &>kb_data_prepare.out &
    """
    # from ckbqa.utils.tools import ProcessManager #实时查看内存占用情况
    # ProcessManager().run()
    main()
