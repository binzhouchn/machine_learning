# -*- coding: utf-8 -*-
"""
__title__ = 'reader_pandas'
__author__ = 'JieYuan'
__mtime__ = '2018/8/18'
"""
import json
import pandas as pd
from tqdm import tqdm


def reader_pandas(file, chunksize=100000, patitions=10 ** 4):
    reader = pd.read_csv(file, chunksize=chunksize)  # pd.read_json(file_path, 'records', lines=True, chunksize=chunkSize)
    chunks = []
    with tqdm(range(patitions), 'Reading ...') as t:
        for _ in t:
            try:
                chunks.append(reader.__next__())
            except StopIteration:
                break
    return pd.concat(chunks, ignore_index=True)

def load_json(file_path):
    with open(file_path) as f:
        with tqdm(f, 'Reading ...') as t:
            return pd.DataFrame([json.loads(line) for line in t])

def reader_open(file):
    l = []
    with open(file, 'r') as file:
        for line in tqdm(file, 'Reading ...'):
            l.append(line.strip().split(','))
    return pd.DataFrame(l[1:], columns=l[0])
