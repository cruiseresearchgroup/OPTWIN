import os
from os import listdir
import pathlib
from os.path import isfile, isdir, join

import csv
import numpy as np
import json

import time

from windowing.optimal_window import CsvOptimalWindowing
from mobiquitous2016_proxy_func import MobiQuitous2016ProxyFunc
from windowing.sliding_window import DStreamManager, CsvDsStream

import sys

def runmobiquitousexperiment():
    if sys.version_info >= (3, 5):
        print('Python version is at least 3.5')
    else:
        print('Please use python >= 3.5')

    print('testing for optimal window code')

    streamdir = './processed/'

    filestoinspect = [
        'S1-ADL1.csv',
        'S1-Drill.csv',
        'S2-ADL2.csv',
        'S2-Drill.csv',
        'S3-ADL3.csv',
        'S3-Drill.csv',
        'S4-ADL2.csv',
        'S4-Drill.csv',
    ]

    func_datasource_arr = []
    for ind_file in filestoinspect:
        func_datasource_arr.append(MobiQuitous2016ProxyFunc(streamdir, ind_file))

    optimal_processor = CsvOptimalWindowing()
    clear_cache = False
    the_sum = optimal_processor.extract_features(600, 5000, 100,
                                                 func_datasource_arr=func_datasource_arr,
                                                 cache_folder='./caches',
                                                 overlap_percentage=0, clear_cache=clear_cache)
    print('Total file processed: {}'.format(the_sum))

    recommended_window_size = optimal_processor.infer_best_window_size(600, 5000, 100,
                                                                       func_datasource_arr=func_datasource_arr,
                                                                       cache_folder='./caches',
                                                                       overlap_percentage=0, clear_cache=clear_cache)

    print('recommended window: {} seconds'.format(recommended_window_size))