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
import glob
import threading
import argparse
import queue
from datetime import datetime
import numpy as np
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
# below: a lambda funciton to flatten the nested list  into list
def flatten(x): return [item for sublist in x for item in sublist]


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


# ------- custom functions ------
def error(message, *lines):
    string = "\n{}ERROR: " + message + "{}\n" + \
        "\n".join(lines) + ("{}\n" if lines else "{}")
    print(string.format(colr.RED_B, colr.RED, colr.ENDC))
    sys.exit(-1)


def warn(message, *lines):
    string = "\n{}WARNING: " + message + "{}\n" + "\n".join(lines) + "{}\n"
    print(string.format(colr.YELLOW_B, colr.YELLOW, colr.ENDC))


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
add_arg('file', nargs='*', default=[])
add_arg('-wd', "--working_dir", type=str, default=False,
        help='str. Working directory if not the current one')

add_arg('-si', '--sample_id', type=str, default=[],
        help='str. Vairable name for sample ID. NOTE: only needed with single file processing')
add_arg('-av', '--annotation_variables', type=str, nargs="+", default=[],
        help='names of the annotation columns in the input data. NOTE: only needed with single file processing')
add_arg("-nt", '--n_timepoints', type=int, default=2,
        help='int. Number of timepoints. NOTE: only needed with single file processing')
add_arg('-ov', '--outcome_variable', type=str, default=[],
        help='str. Vairable name for outcome. NOTE: only needed with single file processing')

add_arg('-fp', '--file_pattern', type=str, default=False,
        help='str. Input file pattern for batch processing')
add_arg('-mf', '--meta_file', type=str, default=False,
        help='str. Meta data for input data files')
add_arg('-mn', '--meta_file-file_name', type=str, default=False,
        help='str. Column name for  in the meta data file')
add_arg('-mt', '--meta_file-n_timepoints', type=str, default=False,
        help='str. Column name for the number of timepoints')
add_arg('-ma', '--meta_file-annotation', type=str, default=False,
        help='str. Column name for annotation columns')
add_arg('-mi', '--meta_file-sample_id', type=str, default=False,
        help='str. Column name for sample subjects ID')
add_arg('-mo', '--meta_file-outcome_var', type=str, default=False,
        help='str. Column name for outcome variable.')

add_bool_arg(parser=parser, name='man_split', input_type='flag',
             help='Manually split data into training and test sets', default=False)
add_arg('-hs', '--holdout_samples', nargs='+', type=str, default=[],
        help='str. Sample IDs selected as holdout test group when --man_split was set')
add_arg('-mh', '--meta_file-holdout_samples', type=str, default=False,
        help='str. Column name for test subjects ID')

add_arg('-ct', '--cross_validation-type', type=str,
        choices=['kfold', 'LOO'], default='kfold', help='str. Cross validation type')
add_arg('-cf', '--cv_fold', type=int, default=10,
        help='int. Number fo cross validation fold when --cross_validation-type=\'kfold\'')
add_arg('-m', '--model_type', type=str, choices=['simple', 'stacked', 'bidirectional'],
        default='simple',
        help='str. LSTM model type. Options: \'simple\', \'stacked\', and \'bidirectional\'')
add_arg('-hu', '--hidden_unit', type=int, default=50,
        help='int. Number of hidden unit for the LSTM netework')
add_arg('-e', '--epoches', type=int, default=500,
        help='int. Number of epoches for LSTM modelling')
add_arg('-b', '--batch_size', type=int, default=32,
        help='int. The batch size for LSTM modeling')
add_arg('--tp', '--training_percentage', type=float, default=0.8,
        help='num, range: 0~1. Split percentage for training set when --no-man_split is set')
add_arg('-o', '--output_dir', type=str,
        default='.', help='str. Output directory')
add_arg('-rs', '--random_state', type=int, default=1, help='int. Random state')
add_bool_arg(parser=parser, name='plot', input_type='flag',
             help='Explort a scatter plot', default=False)
add_arg('-pt', '--plot-type', type=str,
        choices=['scatter', 'bar'], default='scatter', help='str. Plot type')

args = parser.parse_args()
print(args)
print('\n')
print(len(args.file))


# ------ loacl classes ------
class FileProcesser(threading.Thread):
    """
    sub-classing a threading.Thread class to process the data files

        To do:
                [ ] move the "logical checks" no files out of the class
                [ ] make the file processer class sole for processing a single file
    """

    def __init__(self, work_queue, work_dir='.'):
        # make the thread a daemon object
        super(FileProcesser, self).__init__(daemon=True)

        # set up working queue
        self.work_queue = work_queue

        # working directory
        self.cwd = work_dir

    def run(self):
        while True:
            try:
                file = self.work_queue.get()
                self.file_processing(file)
            finally:
                self.work_queue.task_done()

    def file_processing(self, file):
        # load file
        filename = os.path.join(self.cwd, file)
        file_basename = os.path.basename(file)

        try:
            dat = pd.read_csv(filename, engine='python')
            # pd.shape[1]: ncol
            n_feature = int(
                (dat.shape[1] - self._n_annot_col) // self._n_timepoints)
        except Exception as e:
            warn('Could not load the file {}'.format(file_basename))

        # initial settings
        if args.cross_validation_type == 'kfold':
            self._cv_fold = args.cv_fold
        else:
            self._cv_fold = self._n_samples


class DataLoader(object):
    """
    Data loading module

    To do:
        [ ] add length check for outcome variable values
    """

    def __init__(self):
        # load file names strings
        if args.file_pattern:
            self.files = glob.glob(args.file_pattern)
            if len(self.files) == 0:
                error("No files matching the specified file pattern: {}".format(args.file_pattern),
                      'Put all the files in the folder first.')
        else:
            self.files = args.file

        # load meta data
        if len(self.files) > 1:  # load meta data file
            self.meta_file = pd.read_csv(args.meta_file)
            self._fn = [i.split(',') for i in np.array(
                self.meta_file[args.meta_file_file_name])]
            self._fn = flatten(self._fn)
            self._n_timepoints_list, self._holdout_samples_list, self._annotation_var_list, self._sample_id_var_list, self._outcome_var_list = flatten([
                np.array(self.meta_file[args.meta_file_n_timepoints])]), [i.split(',') for i in np.array(
                    self.meta_file[args.meta_file_holdout_samples])], [i.split(',') for i in np.array(
                        self.meta_file[args.meta_file_annotation])], [np.array(self.meta_file[args.meta_file_sample_id])], [i.split(',') for i in np.array(
                            self.meta_file[args.meta_file_outcome_var])]
            self._n_timepoints_dict, self._holdout_dict, self._annot_var_dict, self._sample_id_var_dict, self._outcome_var_dict = dict(
                zip(self._fn, self._n_timepoints_list)), dict(zip(self._fn, self._holdout_samples_list)), dict(
                zip(self._fn, self._annotation_var_list)), dict(zip(self._fn, self._sample_id_var_list)), dict(
                zip(self._fn, self._outcome_var_list))

            self._n_timepoints, self._holdout, self._annot_var, self._sample_id_var, self._outcome_var = None, None, None, None, None
        else:
            self._n_timepoints_dict, self._holdout_samples_dict, self._anntation_var_dict, self._sample_id_var_dict, self._outcome_var_dict = None, None, None, None, None
            self._n_timepoints, self._holdout, self._annot_var = args.n_timepoints, args.holdout_samples, args.annotation_variables
            self._sample_id_var, self._outcome_var = args.sample_id, args.outcome_variable

        # setup working director
        if args.working_dir:
            self.cwd = args.working_dir
        else:
            self.cwd = os.getcwd()


# class MyLSTM(object):
#     def __init__(self, model_type):
#         self.model_type = model_type

#     def modelling(self):
#         if self.model_type == 'simple':
#             print('LSTM modeling')
#         elif self.model_type == 'stacked':
#             print('stacked modelling')


# ------ local variables ------


# ------ setup output folders ------

# ------ training pipeline ------
# -- read data --
mydata = DataLoader()

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
