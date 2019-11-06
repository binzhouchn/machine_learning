# -*- coding: utf-8 -*-
"""
__title__ = 'plot'
__author__ = 'JieYuan'
__mtime__ = '2018/8/15'
"""

import matplotlib.pyplot as plt


class Matplotlib(object):
    def __init__(self):
        """
        https://www.programcreek.com/python/example/4890/matplotlib.rcParams
        """
        plt.style.use('ggplot')
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 中文乱码的处理
        plt.rcParams['axes.unicode_minus'] = False  # 负号
        plt.rcParams["text.usetex"] = False
        plt.rcParams["legend.numpoints"] = 1
        plt.rcParams["figure.figsize"] = (10, 5)
        plt.rcParams["figure.dpi"] = 100
        plt.rcParams["savefig.dpi"] = plt.rcParams["figure.dpi"]
        plt.rcParams["font.size"] = 10
        plt.rcParams["pdf.fonttype"] = 42

    def reset(self):
        plt.close('all')
        plt.rcdefaults()
