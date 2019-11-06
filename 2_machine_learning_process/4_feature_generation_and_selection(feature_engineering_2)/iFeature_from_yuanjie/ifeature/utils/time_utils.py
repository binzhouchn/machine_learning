# -*- coding: utf-8 -*-
"""
__title__ = 'time_utils'
__author__ = 'JieYuan'
__mtime__ = '2018/7/27'
"""


import pandas as pd
import datetime


def _timestamp():
    now = datetime.datetime.now()
    now_str = now.strftime("%Y%m%d%H%M")
    return now_str


from datetime import timedelta, date
pd.date_range(date(2017, 6, 14) - timedelta(days=10), periods=3, freq='D')