import logging
import threading
from functools import wraps


def synchronized(func):
    func.__lock__ = threading.Lock()

    def lock_func(*args, **kwargs):
        with func.__lock__:
            return func(*args, **kwargs)

    return lock_func


def singleton(cls):
    instances = {}
    print(f"*********************************** instances: {instances}")

    @synchronized
    @wraps(cls)
    def get_instance(*args, **kw):
        if cls not in instances:
            logging.info(f"{cls.__name__} async init start ...")
            instances[cls] = cls(*args, **kw)
            logging.info(f"{cls.__name__} async init done ...")
        return instances[cls]

    return get_instance


class Singleton(object):
    instance = None

    @synchronized
    def __new__(cls, *args, **kwargs):
        """
        :type kwargs: object
        """
        if cls.instance is None:
            cls.instance = super().__new__(cls)
        return cls.instance

    def _init__(self, *args, **kwargs):
        pass

    # def __init__(self, num):  # 每次初始化后重新设置，重新初始化实例的属性值
    #     self.a = num + 5
    # #
    # def printf(self):
    #     print(self.a)


@singleton
class A(object):
    def __init__(self, data: int):
        super().__init__()
        self.data = data


class B(Singleton):
    def __init__(self, data: int):
        super().__init__()
        self.data = data ** data


def test():
    a = Singleton(3)
    print(id(a), a, a.a)
    b = Singleton(4)
    print(id(b), b, b.a)


def main():
    a = A(2)
    print(f"a.data: {a.data}")
    a2 = A(4)
    print(f'a2.data: {a2.data}')
    b = B(3)
    print(f'b.data: {b.data}')
    b2 = B(6)
    print(f'b2.data: {b2.data}')
    print(singleton)


if __name__ == '__main__':
    main()
