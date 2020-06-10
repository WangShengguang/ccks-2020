import threading


def apply_async(func, *args, daemon=False, **kwargs):
    thr = threading.Thread(target=func, args=args, kwargs=kwargs)
    if daemon:
        thr.setDaemon(True)  # 随主线程一起结束
    thr.start()
    if not daemon:
        thr.join()
    return thr
