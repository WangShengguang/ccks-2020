import gc
import json
import logging
import os
import pickle
import platform
import re
import resource
import sys
import threading
import time
import traceback

import psutil
from tqdm import tqdm


def pkl_load(file_path: str):
    with open(file_path, 'rb') as f:
        gc.disable()
        obj = pickle.load(f)
        gc.enable()
    return obj


def pkl_dump(obj: object, file_path: str, recursionlimit='default'):
    limit = {'default': sys.getrecursionlimit(),
             'common': 10 * 10000,
             'max': resource.getrlimit(resource.RLIMIT_STACK)[0]
             }
    sys.setrecursionlimit(limit[recursionlimit])
    with open(file_path, 'wb') as f:
        gc.disable()
        pickle.dump(obj, f)
        gc.enable()
    sys.setrecursionlimit(limit['default'])


def json_dump(dict_data, save_path, override_exist=True):
    if override_exist or os.path.isfile(save_path):
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(dict_data, f, ensure_ascii=False, indent=4, sort_keys=True)


def json_load(path, object_hook=None):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f, object_hook=object_hook)


def get_file_linenums(file_name):
    if platform.system() in ['Linux', 'Darwin']:  # Linux,Mac
        num_str = os.popen(f'wc -l {file_name}').read()
        line_num = int(re.findall('\d+', num_str)[0])
    else:  # Windows
        line_num = sum([1 for _ in open(file_name)])
    return line_num


def tqdm_iter_file(file_path, prefix=''):
    line_num = get_file_linenums(file_path)
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=line_num, desc=prefix):
            yield line


def byte2human(bytes, unit='B', precision=2):
    unit_mount_map = {'B': 1, 'KB': 1024, 'MB': 1024 * 1024, 'GB': 1024 * 1024 * 1024}
    memo = bytes / unit_mount_map[unit]
    memo = round(memo, precision)
    return memo


def get_var_size(var, unit='B'):
    size = sys.getsizeof(var)
    readable_size = f"{byte2human(size):.2f} {unit}"
    return readable_size


class ShowTime(object):
    '''
    用上下文管理器计时
    '''

    def __init__(self, prefix=""):
        self.prefix = prefix

    def __enter__(self):
        self.start_timestamp = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.runtime = time.time() - self.start_timestamp
        print("{} take time: {:.2f} s".format(self.prefix, self.runtime))
        if exc_type is not None:
            print(exc_type, exc_val, exc_tb)
            print(traceback.format_exc())
            return self


class ProcessManager(object):
    def __init__(self, check_secends=20, memo_unit='GB', precision=2):
        self.pid = os.getpid()
        self.p = psutil.Process(self.pid)
        self.check_secends = check_secends
        self.memo_unit = memo_unit
        self.precision = precision
        self.start_time = time.time()

    def kill(self):
        child_poll = self.p.children(recursive=True)
        for p in child_poll:
            if not 'SYSTEM' in p.username:
                print(f'kill sub process: PID: {p.pid}  user: {p.username()} name: {p.name()}')
                p.kill()
        self.p.kill()
        print(f'kill {self.pid}')

    def get_memory_info(self):
        memo = byte2human(self.p.memory_info().rss, self.memo_unit)
        info = psutil.virtual_memory()
        total_memo = byte2human(info.total, self.memo_unit)
        used = byte2human(info.used, self.memo_unit)
        free = byte2human(info.free, self.memo_unit)
        available = byte2human(info.available, self.memo_unit)
        cur_pid_percent = info.percent
        return memo, used, free, available, total_memo, cur_pid_percent

    def task(self):
        while True:
            memo, used, free, available, total_memo, cur_pid_percent = self.get_memory_info()
            print('--' * 20)
            print(f'PID: {self.pid} name: {self.p.name()}')
            print(f'当前进程内存占用 :\t {memo:.2f} {self.memo_unit}')
            print(f'used           :\t {used:.2f} {self.memo_unit}')
            print(f'free           :\t {free:.2f} {self.memo_unit}')
            print(f'total          :\t {total_memo} {self.memo_unit}')
            print(f'内存占比        :\t {cur_pid_percent} %')
            print(f'运行时间        :\t {(time.time() - self.start_time) / 60:.2f} min')
            # print('cpu个数：', psutil.cpu_count())
            if cur_pid_percent > 95:
                logging.info(f'内存占比过高: {cur_pid_percent}%， kill {self.pid}')
                self.kill()  # 停止进程
            time.sleep(self.check_secends)

    def run(self):
        thr = threading.Thread(target=self.task)
        thr.setDaemon(True)  # 跟随主线程结束
        thr.start()
