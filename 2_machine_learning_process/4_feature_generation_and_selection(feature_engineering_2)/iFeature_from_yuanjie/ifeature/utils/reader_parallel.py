# -*- coding: utf-8 -*-
"""
__title__ = 'parallel_reader'
__author__ = 'JieYuan'
__mtime__ = '2018/8/14'
"""

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
from tqdm import tqdm

import pandas as pd
import numpy as np

class ParallelReader(object):

    def __init__(self, path=None, n_jobs=32, read_function=None):
        """单文件夹下并行读取多文件
        :param path:
        :param read_function: 默认简单concat
            def reader(group):
                df = pd.DataFrame()
                for p in tqdm_notebook(group, 'Files'):
                    df = pd.concat([df, pd.read_csv(p).assign(file_name=p.name)])
                return df
        """
        self.n_jobs = n_jobs
        self.paths = list(Path(path).iterdir())
        self.groups = np.split(self.paths, n_jobs * np.arange(1, len(self.paths) // n_jobs)) # 分组

        if read_function:
            self.read_function = read_function
        else:
            self.read_function = self.__read_function

    def load_data(self):
        with ThreadPoolExecutor(max_workers=self.n_jobs) as pool:
            df = pd.concat(pool.map(self.read_function, tqdm(self.groups, 'Groups ...')))
        return df

    @staticmethod
    def __read_function(group):
        df = pd.DataFrame()
        for p in tqdm(group, 'Files'):
            df = pd.concat([df, pd.read_csv(p)])
        return df


