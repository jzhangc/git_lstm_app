#!/usr/bin/env python3
"""
Current objectives:
1. tensorflow 2.0: tf.keras
2. DNN class-based modelling
3. Python3 commandline application for LSTM analysis
"""

# ------ system variables -------
__version__ = '0.1.0'


# ------ import modules ------
import math
import os
import argparse

import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import History  # for input argument type check
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
# StratifiedKFold should be used for classification problems
# StratifiedKFold makes sure the fold has an equal representation of the classes
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from custom_functions.custom_exceptions import (NpArrayShapeError,
                                                PdDataFrameTypeError)
from custom_functions.cv_functions import (idx_func, longitudinal_cv_xy_array,
                                           lstm_cv_train, lstm_ensemble_eval,
                                           lstm_ensemble_predict)
from custom_functions.data_processing import training_test_spliter_final
from custom_functions.plot_functions import y_yhat_plot
from custom_functions.util_functions import logging_func


# ------ augment definition ------
parser = argparse.ArgumentParser(description='LSTM regression modelling using multiple-timpoint MEG connectome.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
add_arg = parser.add_argument
add_arg('file', nargs='*', default=[])
