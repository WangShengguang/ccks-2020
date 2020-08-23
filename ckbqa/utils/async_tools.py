import logging
import threading

from ckbqa.utils.decorators import synchronized


def apply_async(func, *args, daemon=False, **kwargs):
    thr = threading.Thread(target=func, args=args, kwargs=kwargs)
    if daemon:
        thr.setDaemon(True)  # 随主线程一起结束
    thr.start()
    if not daemon:
        thr.join()
    return thr


# for sc in A.__subclasses__():#所有子类
#     print(sc.__name__)
@synchronized
def async_init_singleton_class(classes=()):
    logging.info('async_init_singleton_class start ...')
    thrs = [threading.Thread(target=_singleton_class)
            for _singleton_class in classes]
    for t in thrs:
        t.start()
    for t in thrs:
        t.join()  # 等待所有线程结束再往下继续
    logging.info('async_init_singleton_class done ...')
