#!/usr/bin/env python3
"""
Current objectives:
[x] 1. Test argparse
[ ] 2. Test output directory creation
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
                                 epilog='Written by: {}. Current version: {}'.format(
                                     AUTHOR, __version__),
                                 formatter_class=argparse.RawDescriptionHelpFormatter)

add_arg = parser.add_argument
add_arg('file', nargs='*', default=[])
add_arg('-o', '--output_dir', type=str,
        default='.', help='str. Output directory')

add_req = parser.add_argument_group(title='required arguments').add_argument

args = parser.parse_args()


# ------ local variables ------
res_dir = args.output_dir
input_filenames = list()
for i in args.file:
    basename = os.path.basename(i)
    filename = os.path.splitext(basename)[0]
    input_filenames.append(filename)

print(input_filenames)

# ------ setup output folders ------
try:
    os.makedirs(res_dir)
except FileExistsError:
    res_dir = res_dir+'_'+datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(res_dir)
    print('Output directory already exists. Use {} instread.'.format(res_dir))
except OSError:
    print('Creation of directory failed: {}'.format(res_dir))
else:
    print("Output directory created: {}".format(res_dir))

for i in input_filenames:
    sub_dir = os.path.join(res_dir, i)
    try:
        os.makedirs(sub_dir)
        os.makedirs(os.path.join(sub_dir, 'fit'))
        os.makedirs(os.path.join(sub_dir, 'cv_models'))
        os.makedirs(os.path.join(sub_dir, 'intermediate_data'))
    except FileExistsError:
        print('\tCreation of sub-directory failed (already exists): {}'.format(sub_dir))
        pass
    except OSError:
        print('\tCreation of sub-directory failed: {}'.format(sub_dir))
        pass
    else:
        print('\tSub-directory created in {} for file: {}'.format(res_dir, i))

# ------ training pipeline ------
# -- read data --

# -- file processing --

# -- training and export --

# -- model evaluation and plotting --


# ------ __main__ statement ------
# if __name__ == '__main__':
#     pass
