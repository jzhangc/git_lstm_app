#!/usr/bin/env python3
"""
Current objectives:
[x] 1. Test argparse
[x] 2. Test output directory creation
[ ] 3. Test file reading
[ ] 4. Test file processing
[ ] 5. Test training
"""

# ------ import modules ------
# import math
import os
import argparse
from datetime import datetime
# import numpy as np
import pandas as pd
# from tensorflow.keras.callbacks import History  # for input argument type check
# from matplotlib import pyplot as plt
# from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
# # StratifiedKFold should be used for classification problems
# # StratifiedKFold makes sure the fold has an equal representation of the classes
# from sklearn.model_selection import KFold
# from sklearn.preprocessing import MinMaxScaler, StandardScaler

# from custom_functions.custom_exceptions import (NpArrayShapeError,
#                                                 PdDataFrameTypeError)
# from custom_functions.cv_functions import (idx_func, longitudinal_cv_xy_array,
#                                            lstm_cv_train, lstm_ensemble_eval,
#                                            lstm_ensemble_predict)
# from custom_functions.data_processing import training_test_spliter_final
# from custom_functions.plot_functions import y_yhat_plot
# from custom_functions.util_functions import logging_func


# ------ custom functions ------
def add_bool_arg(parser, name, help, input_type, default=False):
    """
    Purpose\n
            autmatically add a pair of mutually exclusive boolean arguments to the
            argparser

    Arguments\n
            parser: a parser object
            name: str. the argument name
            help: str. the help message
            input_type: str. the value type for the argument
            default: the default value of the argument if not set
    """
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=name,
                       action='store_true', help=input_type + '. ' + help)
    group.add_argument('--no-' + name, dest=name,
                       action='store_false', help=input_type + '. ''(Not to) ' + help)
    parser.set_defaults(**{name: default})


# ------ GLOBAL variables -------
__version__ = '0.1.0'
AUTHOR = 'Jing Zhang, PhD'
DESCRIPITON = """
---------------------------- Description ---------------------------
LSTM regression modelling using multiple-timpoint MEG connectome.
Currently, the program only accepts same feature size per timepoint.
--------------------------------------------------------------------
"""


# ------ augment definition ------
parser = argparse.ArgumentParser(description=DESCRIPITON,
                                 epilog='Written by: {}. Current version: {}\n\r'.format(
                                     AUTHOR, __version__),
                                 formatter_class=argparse.RawDescriptionHelpFormatter)

add_arg = parser.add_argument
add_arg('file', nargs='*', default=[])
add_arg('-mf', '--meta_file', type=str, default=False,
        help='str. Meta data for input data files')
add_arg('-mn', '--meta_file-file_name', type=str, default=False,
        help='str. Column name for  in the meta data file')
add_arg('-mt', '--meta_file-n_timepoints', type=str, default=False,
        help='str. Column name for the number of timepoints')
add_arg('-ms', '--meta_file-test_subjects', type=str, default=False,
        help='str. Column name for test subjects ID')
add_arg("-nt", '--n_timepoints', type=int, default=2,
        help='int. Number of timepoints. NOTE: only needed with single file processing')
add_bool_arg(parser=parser, name='man_split', input_type='flag',
             help='Manually split data into training and test sets', default=False)

args = parser.parse_args()
print(args)
print('\n')
print(len(args.file))
if args.meta_file and (not args.meta_file_file_name or not args.meta_file_n_timepoints):
    parser.error(
        'Specify both -mn/--meta_file-file_name, -mt/--meta_file-n_timepoints and -ms/--metawhen -mf/--meta_file is set')
if (len(args.file) > 1 and args.man_split) and not args.meta_file_test_subjects:
    parser.error(
        'Set -ms/--meta_file-test_subjects if multiple input files are provided and -ms/--man_split is on')
if args.man_split and (len(args.holdout_samples) == 0 or not args.meta_file_test_subjects):
    parser.error(
        'set -hs/--holdout_samples or -mts/--meta_file-test_subjects when -ms/--man_split is on.')


# ------ local variables ------
# ------ loacl classes ------
# class InputData(object):
#     def __init__(self, file):
#         self.input = pd.read_csv(file)
#         self.__n_samples__ = self.input.shape[0]  # pd.shape[0]: nrow
#         self.__n_annot_col__ = len(args.annotation_variables)

#         self.__n_timepoints__ = args.n_timepoints
#         self.n_features = int((
#             self.input.shape[1] - self.__n_annot_col__)/self.__n_timepoints__)  # pd.shape[1]: ncol

#         if args.cross_validation_type == 'kfold':
#             self.__cv_fold__ = args.cv_fold
#         else:
#             self.__cv_fold__ = self.__n_samples__


# MyData = InputData(file=args.file[0])
# print('\n')
# print("MyData shape: {}".format(MyData.input.shape))
# print('\n')
# print("MyData.__n_samples__: {}; MyData.__n_annot_col__: {}".format(
#     MyData.__n_samples__, MyData.__n_annot_col__))
# print('\n')
# print("MyData.__cv_fold__: {}; MyData.n_features: {}".format(
#     MyData.__cv_fold__, MyData.n_features))

# ------ setup output folders ------

# ------ training pipeline ------
# -- read data --


# -- file processing --

# -- training and export --

# -- model evaluation and plotting --


# ------ __main__ statement ------
# if __name__ == '__main__':
#     pass
