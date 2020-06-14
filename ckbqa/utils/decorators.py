import gc
import logging
import threading
import traceback
from functools import wraps


def synchronized(func):
    func.__lock__ = threading.Lock()

    @wraps(func)
    def lock_func(*args, **kwargs):
        with func.__lock__:
            res = func(*args, **kwargs)
        return res

    return lock_func


# singleton_classes = set()


def singleton(cls):
    """
        只初始化一次；new和init都只调用一次
    """
    instances = {}  # 当类创建完成之后才有内容
    # singleton_classes.add(cls)

    @synchronized
    @wraps(cls)
    def get_instance(*args, **kw):
        if cls not in instances:  # 保证只初始化一次
            logging.info(f"{cls.__name__} init start ...")
            instances[cls] = cls(*args, **kw)
            logging.info(f"{cls.__name__} init done ...")
            gc.collect()
        return instances[cls]

    return get_instance


class Singleton(object):
    """
        保证只new一次；但会初始化多次，每个子类的init方法都会被调用
        会造成对象虽然是同一个，但因为会不断地调用init方法，对象属性被不断的修改
    """
    instance = None

    @synchronized
    def __new__(cls, *args, **kwargs):
        """
        :type kwargs: object
        """
        if cls.instance is None:  # 保证只new一次；但会初始化多次，每个子类的init方法都会被调用，造成对象虽然是同一个但会不断地被修改
            cls.instance = super().__new__(cls)
        return cls.instance


def try_catch_with_logging(default_response=None):
    def out_wrapper(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                res = func(*args, **kwargs)
            except Exception:
                res = default_response
                logging.error(traceback.format_exc())
            return res

        return wrapper

    return out_wrapper
