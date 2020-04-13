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
import sys
import os
# import glob
import threading
import argparse
# import queue
from datetime import datetime
import numpy as np
import pandas as pd
from custom_functions.cv_functions import (idx_func, longitudinal_cv_xy_array,
                                           lstm_cv_train, lstm_ensemble_eval,
                                           lstm_ensemble_predict)
from custom_functions.data_processing import training_test_spliter_final
# from tensorflow.keras.callbacks import History  # for input argument type check
# from matplotlib import pyplot as plt
# from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
# # StratifiedKFold should be used for classification problems
# # StratifiedKFold makes sure the fold has an equal representation of the classes
# from sklearn.model_selection import KFold
# from sklearn.preprocessing import MinMaxScaler, StandardScaler

# from custom_functions.custom_exceptions import (NpArrayShapeError,
#                                                 PdDataFrameTypeError)
# from custom_functions.data_processing import training_test_spliter_final
# from custom_functions.plot_functions import y_yhat_plot
# from custom_functions.util_functions import logging_func


# ------ system classes ------
class colr:
    WHITE = '\033[0;97m'
    WHITE_B = '\033[1;97m'
    YELLOW = '\033[0;33m'
    YELLOW_B = '\033[1;33m'
    RED = '\033[0;31m'
    RED_B = '\033[1;31m'
    BLUE = '\033[0;94m'
    BLUE_B = '\033[1;94m'
    CYAN = '\033[0;36m'
    CYAN_B = '\033[1;36m'
    ENDC = '\033[0m'  # end colour


class AppArgParser(argparse.ArgumentParser):
    """
    This is a sub class to argparse.ArgumentParser.

    Purpose
                    The help page will display when (1) no argumment was provided, or (2) there is an error
    """

    def error(self, message, *lines):
        string = "\n{}ERROR: " + message + "{}\n" + \
            "\n".join(lines) + ("{}\n" if lines else "{}")
        print(string.format(colr.RED_B, colr.RED, colr.ENDC))
        self.print_help()
        sys.exit(2)


# ------ custom functions ------
# below: a lambda funciton to flatten the nested list into a single list
def flatten(x): return [item for sublist in x for item in sublist]


def error(message, *lines):
    """
    stole from: https://github.com/alexjc/neural-enhance
    """
    string = "\n{}ERROR: " + message + "{}\n" + \
        "\n".join(lines) + ("{}\n" if lines else "{}")
    print(string.format(colr.RED_B, colr.RED, colr.ENDC))
    sys.exit(2)


def warn(message, *lines):
    """
    stole from: https://github.com/alexjc/neural-enhance
    """
    string = "\n{}WARNING: " + message + "{}\n" + "\n".join(lines) + "{}\n"
    print(string.format(colr.YELLOW_B, colr.YELLOW, colr.ENDC))


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
parser = AppArgParser(description=DESCRIPITON,
                      epilog='Written by: {}. Current version: {}\n\r'.format(
                          AUTHOR, __version__),
                      formatter_class=argparse.RawDescriptionHelpFormatter)

add_arg = parser.add_argument
add_arg('file', nargs=1, default=[],
        help='Input CSV file. Currently only one file is accepable.')
add_arg('-w', "--working_dir", type=str, default=False,
        help='str. Working directory if not the current one')

add_arg('-s', '--sample_id', type=str, default=False,
        help='str. Vairable name for sample ID. NOTE: only needed with single file processing')
add_arg('-a', '--annotation_variables', type=str, nargs="+", default=[],
        help='names of the annotation columns in the input data. NOTE: only needed with single file processing')
add_arg("-n", '--n_timepoints', type=int, default=False,
        help='int. Number of timepoints. NOTE: only needed with single file processing')
add_arg('-y', '--outcome_variable', type=str, default=False,
        help='str. Vairable name for outcome. NOTE: only needed with single file processing')

add_bool_arg(parser=parser, name='man_split', input_type='flag',
             help='Manually split data into training and test sets. When set, the split is on -s/--sample_id.', default=False)
add_arg('-t', '--holdout_samples', nargs='+', type=str, default=[],
        help='str. Sample IDs selected as holdout test group when --man_split was set')
add_arg('-p', '--training_percentage', type=float, default=0.8,
        help='num, range: 0~1. Split percentage for training set when --no-man_split is set')

add_arg('-v', '--cross_validation-type', type=str,
        choices=['kfold', 'LOO', 'boot'], default='kfold', help='str. Cross validation type')
add_arg('-f', '--cv_fold', type=int, default=10,
        help='int. Number fo cross validation fold when --cross_validation-type=\'kfold\'')
add_bool_arg(parser=parser, name='cv_only', input_type='flag',
             help='Explort a scatter plot', default=False)

add_arg('-m', '--model_type', type=str, choices=['regression', 'classification'],
        default='classifciation', help='str. Model type. Options: \'regression\' and \'classification\'')
add_arg('-l', '--lstm_type', type=str, choices=['simple', 'stacked', 'bidirectional'],
        default='simple',
        help='str. LSTM model type. Options: \'simple\', \'stacked\', and \'bidirectional\'')
add_arg('-u', '--hidden_unit', type=int, default=50,
        help='int. Number of hidden unit for the LSTM netework')
add_arg('-e', '--epoches', type=int, default=500,
        help='int. Number of epoches for LSTM modelling')
add_arg('-b', '--batch_size', type=int, default=32,
        help='int. The batch size for LSTM modeling')
add_arg('-o', '--output_dir', type=str,
        default='.', help='str. Output directory')
add_arg('-r', '--random_state', type=int, default=1, help='int. Random state')
add_bool_arg(parser=parser, name='plot', input_type='flag',
             help='Explort a scatter plot', default=False)
add_arg('-j', '--plot-type', type=str,
        choices=['scatter', 'bar'], default='scatter', help='str. Plot type')

args = parser.parse_args()

# ------ loacl classes ------
# class FileProcesser(threading.Thread):
#     """
#         To do:
#         [ ] process individual files from the DataLoader class
#         [ ] see if multi-threading or multi-processing is needed to use i
#     """

#     def __init__(self, work_queue, work_dir='.'):
#         # make the thread a daemon object
#         super(FileProcesser, self).__init__(daemon=True)
#         # working directory
#         self.cwd = work_dir


class DataLoader(object):
    """
    Data loading class

    To do:
        [ ] add length check for outcome variable values
        [ ] set up X and Y
    """

    def __init__(self):
        # setup working director
        if args.working_dir:
            self.cwd = args.working_dir
        else:
            self.cwd = os.getcwd()

        # random state
        self._rand = args.random_state

        # check and load files
        if not args.sample_id:
            error('-s/--sample_id flag is mandatory.')
        if not args.n_timepoints:
            error('-n/--n_timepoints flag is mandatory.')
        if not args.outcome_variable:
            error('-y/--outcome_variable flag is mandatory.')
        if len(args.annotation_variables) < 1:
            error('-a/--annotation_variables flag is mandatory.')

        # args.file is a list. so use [0] to grab the string
        self.file = os.path.join(self.cwd, args.file[0])
        self._basename = os.path.basename(args.file[0])
        self.filename,  self._name_ext = os.path.splitext(self._basename)[
            0], os.path.splitext(self._basename)[1]

        if self._name_ext != ".csv":
            error('The input file should be in csv format.',
                  'Please check.')
        elif not os.path.exists(self.file):
            error('The input file or directory does not exist.',
                  'Please check.')
        else:
            self.raw = pd.read_csv(self.file, engine='python')

            self.annot_vars = args.annotation_variables
            self._n_annot_col = len(self.annot_vars)
            self.n_timepoints = args.n_timepoints
            self.n_features = int(
                (self.raw.shape[1] - self._n_annot_col) // self.n_timepoints)  # pd.shape[1]: ncol

    def data_split(self, percentage, random_state):
        if args.cv_only:
            self.test_x, self.test_y = None, None

            self.training_x = None
            self.training_y = None
        else:
            if args.man_split:
                self.training, self.test, _, _ = training_test_spliter_final(data=self.raw, random_state=self._rand,
                                                                             man_split=args.man_split, man_split_colname=args.sample_id)
            else:
                self.training, self.test = None, None

    def processing(self):
        if args.cv_only:
            self.training_x = None
        else:
            self.training_x = None


# class model_LSTM(object):
#     """
#     Modelling
#     """

#     def __init__(self):
#         self.lstm = None
#         self.hidden_unit = args.hidden_unit
#         self.epoches = args.epoches
#         self.model_type = args.model_type
#         self.data = DataLoader()
#         self.n_timepoint = self.data.n_timepoint

#     def simple_model(self):
#         None

#     def stacked_model(self):
#         None

#     def bidirectional_model(self):
#         None


# ------ local variables ------

# ------ setup output folders ------
# ------ training pipeline ------
# -- read data --
print(args)
print('\n')
print(len(args.file))
print(args.file[0])

print(os.path.exists(args.file[0]))

mydata = DataLoader()
print(mydata.raw)
print("\n")
print("input file path: {}".format(mydata.file))
print("\n")
print("input file name: {}. input file extension: {}".format(
    mydata.filename,  mydata.name_ext))
print("\n")
print("number of timepoints in the input file: {}".format(mydata.n_timepoints))
print("\n")
print("number of features in the inpout file: {}".format(mydata.n_features))


# print('mydata.cwd: {}'.format(mydata.cwd))
# print('self._n_timepoints: {}, self._holdout:{}, self._annot_var:{}, self._sample_id_var:{}'.format(
#     mydata._n_timepoints, mydata._holdout, mydata._annot_var, mydata._sample_id_var))

# print('self._n_timepoints_dict: {}'.format(mydata._n_timepoints_dict))
# print('\n')
# print('self._holdout_dict:{}'.format(mydata._holdout_dict))
# print('\n')
# print('self._outcome_var_dict'.format(mydata._outcome_var_dict))

# -- file processing --


# -- training and export --

# -- model evaluation and plotting --

# ------ process/__main__ statement ------
# if __name__ == '__main__':
#     mydata = DataLoader()
