# -*- coding: utf-8 -*-
"""
__title__ = 'setup'
__author__ = 'JieYuan'
__mtime__ = '2018/7/23'
"""

from setuptools import find_packages, setup

with open("README.md", encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='ifeature',
    version='0.0.1',
    url='https://github.com/Jie-Yuan/iFeature',
    keywords=["DeepLearning", "313303303@qq.com"],
    description=('description'),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='JieYuan',
    author_email='313303303@qq.com',
    maintainer='JieYuan',
    maintainer_email='313303303@qq.com',
    license='MIT',
    packages=find_packages(),
    include_package_data=True,
    package_data={'': ['*.*']},
    platforms=["all"],
    python_requires='>=3, <4',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: Implementation',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries'
    ]
)
