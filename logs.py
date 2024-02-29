#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    : logs.py
"""

import sys

from loguru import logger as _logger

from const import PROJECT_ROOT


def define_log_level(print_level="INFO", logfile_level="DEBUG"):
    """调整日志级别到level之上
       Adjust the log level to above level
    """
    _logger.remove()
    _logger.add(sys.stderr, level=print_level)
    _logger.add(PROJECT_ROOT / 'logs/log.txt', level=logfile_level)
    return _logger


logger = define_log_level()