import argparse

from ckbqa.utils.logger import logging_config


def set_envs(cpu_only, allow_gpus):
    import os
    import random
    rand_seed = 1234
    random.seed(rand_seed)
    if cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("CPU only ...")
    elif allow_gpus:
        from ckbqa.utils.gpu_selector import get_available_gpu
        available_gpu = get_available_gpu(num_gpu=1, allow_gpus=allow_gpus)  # default allow_gpus 0,1,2,3
        os.environ["CUDA_VISIBLE_DEVICES"] = available_gpu
        print("* using GPU: {} ".format(available_gpu))  # config前不可logging，否则config失效
    # 进程内存管理
    # from ckbqa.utils.tools import ProcessManager
    # ProcessManager().run()


def run(model_name, mode):
    logging_config(f'{model_name}_{mode}.log', stream_log=True)
    if model_name in ['bert_match', 'bert_match2']:
        from ckbqa.models.relation_score.trainer import RelationScoreTrainer
        RelationScoreTrainer(model_name).train_match_model()
    elif model_name == 'entity_score':
        from ckbqa.models.entity_score.model import EntityScore
        EntityScore().train()


def main():
    ''' Parse command line arguments and execute the code
        --stream_log, --relative_path, --log_level
        --allow_gpus, --cpu_only
    '''
    parser = argparse.ArgumentParser(description="基础，通用parser")
    # logging config 日志配置
    parser.add_argument('--stream_log', action="store_true", help="是否将日志信息输出到标准输出")  # log print到屏幕
    parser.add_argument('--allow_gpus', default="0,1,2,3", type=str,
                        help="指定GPU编号，0 or 0,1,2 or 0,1,2...7  | nvidia-smi 查看GPU占用情况")
    parser.add_argument('--cpu_only', action="store_true", help="CPU only, not to use GPU ")
    #
    group = parser.add_mutually_exclusive_group(required=True)  # 一组互斥参数,且至少需要互斥参数中的一个
    #
    all_models = ['bert_match', 'bert_match2', 'entity_score']
    group.add_argument('--train', type=str, choices=all_models, help="训练")
    group.add_argument('--test', type=str, choices=all_models, help="测试")
    # parse args
    args = parser.parse_args()
    #
    set_envs(args.cpu_only, args.allow_gpus)  # 设置环境变量等
    #
    if args.train:
        model_name = args.train
        mode = 'train'
    elif args.test:
        model_name = args.test
        mode = 'test'
    else:
        raise ValueError('must set model name ')
    run(model_name, mode)


if __name__ == '__main__':
    """ 代码执行入口
    examples:
        nohup python manage.py --train bert_match &>bert_match.out&
        nohup python manage.py --train entity_score &>entity_score.out&
    """
    # from ckbqa.utils.tools import ProcessManager #实时查看内存占用情况
    # ProcessManager().run()
    main()
