import argparse

from ckbqa.utils.logger import logging_config

logging_config('./data.log', stream_log=True)


def create_db_tabels():
    from sqlalchemy_utils import database_exists, create_database
    from ckbqa.dao.db import sqlite_db_engine
    if not database_exists(sqlite_db_engine.url):
        create_database(sqlite_db_engine.url)
    from ckbqa.dao.sqlite_models import BaseModel
    BaseModel.metadata.create_all(sqlite_db_engine)  # 创建表


def create_neo4j_graph():
    logging_config()
    from ckbqa.dataset.neo4j_graph import create_graph_csv
    create_graph_csv()


def data_prepare():
    from ckbqa.dataset.kb_data_prepare import (create_lac_custom_dict, candidate_words, fit_triples,
                                               map_mention_entity,create_graph_csv)
    from ckbqa.dataset.data_prepare import data2samples,fit_on_texts
    fit_triples()  # 生成字典
    # map_mention_entity()
    candidate_words()  # 属性
    # data2samples(neg_rate=3)
    # data_convert()
    # fit_on_texts()
    # create_lac_custom_dict()  #
    # create_db_tabels()
    # from ckbqa.dataset.neo4j_graph import create_graph_csv
    create_graph_csv()
    #
    from examples.lac_test import lac_model
    lac_model()


def main():
    ''' Parse command line arguments and execute the code
        --stream_log, --relative_path, --log_level
        --allow_gpus, --cpu_only
    '''
    parser = argparse.ArgumentParser(description="基础，通用parser")
    # logging config 日志配置
    parser.add_argument('--stream_log', action="store_true", help="是否将日志信息输出到标准输出")  # log print到屏幕
    #
    group = parser.add_mutually_exclusive_group(required=True)  # 一组互斥参数,且至少需要互斥参数中的一个

    group.add_argument('--data_prepare', action="store_true", help="数据预处理")
    group.add_argument('--create_graph', action="store_true", help="数据预处理")
    # parse args
    args = parser.parse_args()
    #
    # from ckbqa.utils.tools import ProcessManager
    # ProcessManager().run()
    if args.data_prepare:
        data_prepare()
    elif args.create_graph:
        create_neo4j_graph()


if __name__ == '__main__':
    """ 代码执行入口
    examples:
        python manage.py --data_prepare
    """

    main()
