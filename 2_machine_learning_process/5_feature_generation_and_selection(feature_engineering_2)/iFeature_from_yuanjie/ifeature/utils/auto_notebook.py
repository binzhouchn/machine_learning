# -*- coding: utf-8 -*-
"""
__title__ = 'auto_notebook'
__author__ = 'JieYuan'
__mtime__ = '2018/8/18'
"""
import functools
from collections import Iterable

try:
    from IPython import get_ipython

    if 'IPKernelApp' not in get_ipython().config:
        raise ImportError("console")
except:
    from tqdm import tqdm

else:
    from tqdm import tqdm_notebook as tqdm


def mytqdm(desc=None):
    def wrapper(func):
        """
	pip install ipywidgets
	jupyter nbextension enable --py widgetsnbextension

        @mytqdm('Example ...')
        def func(x):
            for i in x:
                pass
        """
        @functools.wraps(func)
        def _wrapper(iter_obj, *args, **kwargs):
            """保证第一个参数为可迭代参数即可"""
            assert isinstance(iter_obj, Iterable)
            with tqdm(iter_obj, desc) as t:
                temp = func(t, *args, **kwargs)
            return temp

        return _wrapper

    return wrapper


