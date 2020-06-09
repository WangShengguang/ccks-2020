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

import logging

from ckbqa.qa.lac_tools import BaiduLac
from ckbqa.utils.logger import logging_config
from config import Config

logging_config('lac_test.log', stream_log=True)


def lac_model():
    # self.lac_seg = LAC(mode='seg')
    logging.info(f' load lac_custom_dict from  {Config.lac_custom_dict_txt} start ...')
    baidu_lac = BaiduLac(mode='lac', _load_customization=True, reload=True)  # 装载LAC模型
    save_path = baidu_lac._save_customization()  # Config.lac_custom_dict_txt
    logging.info(f'load lac_custom_dict done, save to {save_path}...')


def lac_test():
    logging.info(f' load lac_custom_dict from  {Config.lac_custom_dict_txt} start ...')
    save_path = './test.pkl'
    baidu_lac = BaiduLac(mode='lac', _load_customization=True, reload=True)  # 装载LAC模型
    logging.info(f'load lac_custom_dict done, save to {save_path}...')


if __name__ == '__main__':
    lac_model()
