#!/usr/bin/env python3
"""
Current objective:
Python3 commandline application for LSTM analysis
"""

# ------ import modules ------
# import math
# import os
import argparse

# import numpy as np
# import pandas as pd
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
                       action='store_false', help=input_type + '. ''not to ' + help)
    parser.set_defaults(**{name: default})


# ------ system variables -------
__version__ = '0.1.0'
author = 'Jing Zhang, PhD'
description = """
---------------------------- Description ---------------------------
LSTM regression modelling using multiple-timpoint MEG connectome.
Currently, the program only accepts same feature size per timepoint.
-------------------------------------------------------------------- 
"""

# ------ augment definition ------
parser = argparse.ArgumentParser(description=description,
                                 epilog='Written by: {}. Current version: {}'.format(
                                     author, __version__),
                                 formatter_class=argparse.RawDescriptionHelpFormatter)

add_arg = parser.add_argument
add_arg('file', nargs='*', default=[])
add_arg('--file_pattern', '-fp', type=str, default=False,
        help='str. Input file pattern for batch processing')
add_arg('--model_type', '-m', type=str, default='simple',
        help='str. LSTM model type. Options: \'simple\', \'stacked\', and \'bidirectional\''
        )
add_arg('--n_timepoints', "-nt", type=int, default=2,
        help='int. Number of timepoints')
add_arg('--output_dir', '-o', type=str,
        default='.', help='str. Output directory')
add_arg('--sample_variable', '-sv', type=str, default=[],
        help='str. Vairable name for samples')
add_bool_arg(parser=parser, name='man_split', input_type='bool',
             help='Manually split data into training and test sets', default=False)
add_arg('--split_percentage', '--sp', type=float, default=0.8,
        help='num. Training percentage of ')
add_arg('--selected_test_samples', '-st', nargs='+', type=str, default=[],
        help='str. Sample IDs selected as test group when --man_split was set')

add_req = parser.add_argument_group(title='required arguments').add_argument
add_req('--sample_annotation', '-sa', type=str, default=[],
        required=True, help='str. Sample annotation .csv file')
add_req('--n_features', '-nf', type=int, default=[],
        help='int. Number of features each timepoint', required=True)

args = parser.parse_args()

print(args)

# ------ __main__ statement ------
# if __name__ == '__main__':
#     pass
