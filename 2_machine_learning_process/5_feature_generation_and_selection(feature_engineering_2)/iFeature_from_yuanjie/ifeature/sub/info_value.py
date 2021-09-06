# coding=utf-8
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from tqdm import tqdm


class InformationValue(object):
    def __init__(self, df: pd.DataFrame, label: str):
        """
        :param df:
        :param label: target name
        """
        assert label in df.columns

        self.label = label
        self.df = df.assign(_label=1 - df[label])
        self.feats = [col for col in df.columns if col != label]

        self.y1 = self.df[label].values.sum()
        self.y0 = self.df['_label'].values.sum()

    def iv(self, order=True, n_jobs=16):
        with ThreadPoolExecutor(max_workers=n_jobs) as pool:
            ivs = pool.map(self.__iv, tqdm(self.feats, 'Calculating ...'), chunksize=1)
        z = zip(ivs, self.feats)
        return pd.DataFrame(sorted(z, reverse=True) if order else list(z), columns=['iv', 'feats'])

    def __iv(self, feat):
        gr = self.df.groupby(feat)
        gr1, gr0 = gr[self.label].sum().values + 1e-8, gr['_label'].sum().values + 1e-8
        good, bad = gr1 / self.y1, gr0 / self.y0
        woe = np.log(good / bad)
        iv = (good - bad) * woe
        return iv.sum()
