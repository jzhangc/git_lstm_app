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
add_arg('-si', '--sample_id', type=str, default=[],
        help='str. Vairable name for sample ID. NOTE: only needed with single file processing')
add_arg('-av', '--annotation_variable', type=str, nargs="+", default=[],
        help='names of the annotation columns in the input data. NOTE: only needed with single file processing')
add_arg("-nt", '--n_timepoints', type=int, default=2,
        help='int. Number of timepoints. NOTE: only needed with single file processing')
add_arg('-ov', '--outcome_variable', type=str, default=[],
        help='str. Vairable name for outcome. NOTE: only needed with single file processing')


add_arg('-fp', '--file_pattern', type=str, default=False,
        help='str. Input file pattern for batch processing')
add_arg('-wd', "--working_dir", type=str, default=False,
        help='str. Working directory if not the current one')
add_arg('-mf', '--meta_file', type=str, default=False,
        help='str. Meta data for input data files')
add_arg('-mn', '--meta_file-file_name', type=str, default=False,
        help='str. Column name for  in the meta data file')
add_arg('-mt', '--meta_file-n_timepoints', type=str, default=False,
        help='str. Column name for the number of timepoints')
add_arg('-ms', '--meta_file-test_subjects', type=str, default=False,
        help='str. Column name for test subjects ID')
add_bool_arg(parser=parser, name='man_split', input_type='flag',
             help='Manually split data into training and test sets', default=False)
add_arg('-ma', '--meta_file-annotation', type=str, default=False,
        help='str. Column name for annotation columns')
add_arg('-mi', '--meta_file-sample_id', type=str, default=False,
        help='str. Column name for sample subjects ID')

args = parser.parse_args()
print(args)
print('\n')
print(len(args.file))


# ------ loacl classes ------
class FileProcesser(threading.Thread):
    """
    sub-classing a threading.Thread class to process the data files
    """

    def __init__(self, work_queue):
        # make the thread a daemon object
        super(FileProcesser, self).__init__(daemon=True)

        # set up working queue
        self.work_queue = work_queue

        # initial settings
        if args.cross_validation_type == 'kfold':
            self._cv_fold = args.cv_fold
        else:
            self._cv_fold = self._n_samples

    def run(self):
        while True:
            try:
                file = self.work_queue.get()
            finally:
                self.work_queue.task_done()

    def file_processing(self, file):
        filename = os.path.join(self.cwd, file)
        file_basename = os.path.basename(file)

        try:
            dat = pd.read_csv(filename, engine='python')
            # pd.shape[1]: ncol
            n_feature = int(
                (dat.shape[1] - self._n_annot_col) // self._n_timepoints)


class DataLoader(object):
    def __init__(self, file):
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
            self._file_basename = [os.path.basename(n) for n in df]
            self._n_timepoints_list, self._test_subjects_list, self._anntation_var_list, self._sample_id_var_list = [
                np.array(self.meta_file[args.meta_file_n_timepoints])], [i.split(",") for i in np.array(
                    self.meta_file[args.meta_file_test_subjects])], [i.split(",") for i in np.array(
                        self.meta_file[args.meta_file_annotation])], [np.array(self.meta_file[args.meta_file_sample_id])]
            self._timepoints_dict, self._test_subj_dict, self._annot_var_dict, self._sample_id_var_dict = dict(
                zip(self._file_basename, self._n_timepoints_list)),
            dict(zip(self._file_basename, self._test_subjects_list)), dict(
                zip(self._file_basename, self._annotation_var_list)), dict(zip(self._file_basename, self._sample_id_var_list))
        else:
            self._n_timepoints_dict, self._test_subjects_dict, self._anntation_var_dict, self._sample_id_var_dict = None, None, None, None

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

# -- file processing --

# -- training and export --

# -- model evaluation and plotting --

# ------ process/__main__ statement ------
# MyData = InputData(file=args.file[0])
# print('\n')
# print("MyData shape: {}".format(MyData.input.shape))
# print('\n')
# print("MyData.__n_samples__: {}; MyData.__n_annot_col__: {}".format(
#     MyData.__n_samples__, MyData.__n_annot_col__))
# print('\n')
# print("MyData.__cv_fold__: {}; MyData.n_features: {}".format(
#     MyData.__cv_fold__, MyData.n_features))

# if __name__ == '__main__':
#     pass
