import threading


def apply_async(func, *args, **kwargs):
    thr = threading.Thread(target=func, args=args, kwargs=kwargs)
    # thr.setDaemon(True) #随主线程一起结束
    thr.start()
    return thr
