import logging.handlers


def logging_config(logging_name='./run.log', stream_log=False, log_level="info"):
    """
    :param logging_name:  log名
    :param stream_log: 是否把log信息输出到屏幕,标准输出
    :param level: fatal,error,warn,info,debug
    :return: None
    """
    log_handles = [logging.handlers.RotatingFileHandler(
        logging_name, maxBytes=20 * 1024 * 1024, backupCount=5, encoding='utf-8')]
    if stream_log:
        log_handles.append(logging.StreamHandler())
    logging_level = {"fatal": logging.FATAL, "error": logging.ERROR, "warn": logging.WARN,
                     "info": logging.INFO, "debug": logging.DEBUG}[log_level]
    logging.basicConfig(
        handlers=log_handles,
        level=logging_level,
        format="%(asctime)s - %(levelname)s %(filename)s %(funcName)s %(lineno)s - %(message)s"
    )

# if __name__ == "__main__":
#     logging_config("./test.log", stream_log=True)  # ../../log/test.log
#     logging.info("标准输出 log ...")
#     logging.debug("hello")
