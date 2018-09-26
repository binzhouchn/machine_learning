# -*- coding: utf-8 -*-
"""
__title__ = 'os_utils'
__author__ = 'JieYuan'
__mtime__ = '2018/7/27'
"""

import os
import shutil


def _makedirs(dir, force=False):
    if os.path.exists(dir):
        if force:
            shutil.rmtree(dir)
            os.makedirs(dir)
    else:
        os.makedirs(dir)


