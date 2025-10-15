import logging
import os
import sys

def setup_logger(name, log_dir, filename="training.log"):
    """配置一个 logger，使其同时输出到控制台和文件。"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 控制台处理器
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    if not logger.hasHandlers():
        logger.addHandler(stream_handler)

    # 文件处理器
    file_handler = logging.FileHandler(os.path.join(log_dir, filename))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger