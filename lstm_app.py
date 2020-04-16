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
import argparse
import math
import os
import sys
import threading
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import (KFold, LeaveOneOut, ShuffleSplit,
                                     StratifiedKFold, StratifiedShuffleSplit)
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.layers import (LSTM, BatchNormalization, Bidirectional,
                                     Dense, Dropout)
from tensorflow.keras.models import Sequential

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
# set the arguments
parser = AppArgParser(description=DESCRIPITON,
                      epilog='Written by: {}. Current version: {}\n\r'.format(
                          AUTHOR, __version__),
                      formatter_class=argparse.RawDescriptionHelpFormatter)

add_arg = parser.add_argument
add_arg('file', nargs=1, default=[],
        help='Input CSV file. Currently only one file is accepable.')
add_arg('-w', "--working_dir", type=str, default=None,
        help='str. Working directory if not the current one')

add_arg('-s', '--sample_id_var', type=str, default=None,
        help='str. Vairable name for sample ID. NOTE: only needed with single file processing')
add_arg('-a', '--annotation_vars', type=str, nargs="+", default=[],
        help='names of the annotation columns in the input data. NOTE: only needed with single file processing')
add_arg("-n", '--n_timepoints', type=int, default=None,
        help='int. Number of timepoints. NOTE: only needed with single file processing')
add_arg('-y', '--outcome_variable', type=str, default=None,
        help='str. Vairable name for outcome. NOTE: only needed with single file processing')

add_arg('-v', '--cv_type', type=str,
        choices=['kfold', 'LOO', 'monte'], default='kfold', help='str. Cross validation type')
add_arg('-f', '--cv_fold', type=int, default=10,
        help='int. Number of cross validation fold when --cv_type=\'kfold\'')
add_arg('-mn', '--n_monte', type=int, default=10,
        help='int. Number of Monte Carlo cross validation iterations when --cv_type=\'monte\'')
add_arg('-mt', '--monte_test_rate', type=float, default=0.2,
        help='float. Ratio for cv test data split when --cv_type=\'monte\'')
add_bool_arg(parser=parser, name='cv_only', input_type='flag',
             help='Explort a scatter plot', default=False)
add_bool_arg(parser=parser, name='man_split', input_type='flag',
             help='Manually split data into training and test sets. When set, the split is on -s/--sample_id_var.', default=False)
add_arg('-t', '--holdout_samples', nargs='+', type=str, default=[],
        help='str. Sample IDs selected as holdout test group when --man_split was set')
add_arg('-p', '--training_percentage', type=float, default=0.8,
        help='num, range: 0~1. Split percentage for training set when --no-man_split is set')
add_arg('-r', '--random_state', type=int, default=1, help='int. Random state')

add_arg('-m', '--model_type', type=str, choices=['regression', 'classification'],
        default='classifciation', help='str. Model type. Options: \'regression\' and \'classification\'')
add_arg('-l', '--lstm_type', type=str, choices=['simple', 'bidirectional'],
        default='simple',
        help='str. LSTM model type. \'simple\' also contains stacked strcuture.')
add_arg('-ns', '--n_stack', type=int, default=1,
        help='int. Number of LSTM stacks. 1 means no stack.')
add_arg('-e', '--epochs', type=int, default=500,
        help='int. Number of epochs for LSTM modelling')
add_arg('-b', '--batch_size', type=int, default=32,
        help='int. The batch size for LSTM modeling')
add_arg('-d', '--dense_activation', type=str, choices=['relu', 'linear', 'sigmoid', 'softmax'],
        default='linear', help='str. Acivitation function for the dense layer of the LSTM model.')
add_arg('-c', '--loss', type=str,
        choices=['mean_squared_error', 'binary_crossentropy',
                 'categorical_crossentropy', 'sparse_categorical_crossentropy', 'hinge'],
        default='mean_squared_error', help='str. Loss function for LSTM models.')
add_arg('-g', '--optimizer', type=str,
        choices=['adam', 'sgd'], default='adam', help='str. Model optimizer.')
add_arg('-u', '--hidden_units', type=int, default=50,
        help='int. Number of hidden unit for the LSTM network')
add_arg('-x', '--dropout_rate', type=float, default=0.0,
        help='float, 0.0~1.0. Dropout rate for LSTM models . 0.0 means no dropout.')
add_bool_arg(parser=parser, name='stateful', input_type='flag', default=False,
             help="Use stateful LSTM for modelling.")

add_arg('-o', '--output_dir', type=str,
        default='.', help='str. Output directory')


add_bool_arg(parser=parser, name='plot', input_type='flag',
             help='Explort a scatter plot', default=False)
add_arg('-j', '--plot-type', type=str,
        choices=['scatter', 'bar'], default='scatter', help='str. Plot type')

args = parser.parse_args()
# check the arguments. did not use parser.error as error() has fancy colours
if not args.sample_id_var:
    error('-s/--sample_id_var missing.',
          'Be sure to set the following: -s/--sample_id_var, -n/--n_timepoints, -y/--outcome_variable, -a/--annotation_vars')
if not args.n_timepoints:
    error('-n/--n_timepoints flag missing.',
          'Be sure to set the following: -s/--sample_id_var, -n/--n_timepoints, -y/--outcome_variable, -a/--annotation_vars')
if not args.outcome_variable:
    error('-y/--outcome_variable flag missing.',
          'Be sure to set the following: -s/--sample_id_var, -n/--n_timepoints, -y/--outcome_variable, -a/--annotation_vars')
if len(args.annotation_vars) < 1:
    error('-a/--annotation_vars missing.',
          'Be sure to set the following: -s/--sample_id_var, -n/--n_timepoints, -y/--outcome_variable, -a/--annotation_vars')

if args.man_split and len(args.holdout_samples) < 1:
    error('Set -t/--holdout_samples when --man_split was set.')

if args.dropout_rate < 0.0 or args.dropout_rate > 1.0:
    error('-x/--dropout_rate should be between 0.0 and 1.0.')

if args.n_stack < 1:
    error('-ns/--n_stack should be equal to or greater than 1.')

if args.cv_type == 'monte':
    if args.monte_test_rate < 0.0 or args.monte_test_rate > 1.0:
        error('-mt/--monte_test_rate should be between 0.0 and 1.0.')

# ------ loacl classes ------


class DataLoader(object):
    """
    # Purpose
        Data loading class.

    # Details
        This class is designed to load the data and set up data for training LSTM models.

    # Methods
        __init__: load data and other information from argparser
        data_split: set up data for model training. No data splitting for the "CV only" mode.

    # Public class attributes
        cwd: str. working directory
        model_type: str. model type, classification or regression
        y_var: str. variable nanme for outcome
        file: str. complete input file path
        filename: str. input file name without extension
        raw: pandas dataframe. input data
        annot_vars: list of strings. column names for the annotation variables in the input dataframe
        n_timepints: int. number of timepoints
        n_features: int. number of features

    # Private class attributes (excluding class property)
        _rand: int. random state
        _basename: str. complete file name (with extension), no path
        _n_annot_col: int. number of annotation columns

    # Class property
        modelling_data: set up the data for model training. data is split if necessary.
            returns a dict object with 'training' and 'test' items

            _m_data: dict. output dictionary
            _training: pandas dataframe. data for model training.
            _test: pandas dataframe. holdout test data. Only available without the "--cv_only" flag
    """

    def __init__(self):
        # setup working director
        if args.working_dir:
            self.cwd = args.working_dir
        else:
            self.cwd = os.getcwd()

        # random state
        self._rand = args.random_state

        # load files
        self.model_type = args.model_type
        # convert to a list for training_test_spliter_final() to use
        self.y_var = [args.outcome_variable]

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
            self.annot_vars = args.annotation_vars
            self._n_annot_col = len(self.annot_vars)
            self.n_timepoints = args.n_timepoints
            self.n_features = int(
                (self.raw.shape[1] - self._n_annot_col) // self.n_timepoints)  # pd.shape[1]: ncol

        self.modelling_data = args.man_split  # call setter here

    @property
    def modelling_data(self):
        # print("called getter") # for debugging
        return self._modelling_data

    @modelling_data.setter
    def modelling_data(self, man_split):
        # print("called setter") # for debugging
        if args.cv_only:  # only training is stored
            self._training, self._test = self.raw, None
        else:
            # training and holdout test data split
            if args.man_split:
                # manual data split: the checks happen in the training_test_spliter_final() function
                self._training, self._test, _, _ = training_test_spliter_final(data=self.raw, random_state=self._rand,
                                                                               man_split=man_split, man_split_colname=args.sample_id_var,
                                                                               man_split_testset_value=args.holdout_samples,
                                                                               x_standardization=False, y_min_max_scaling=False)
            else:
                self._training, self._test, _, _ = training_test_spliter_final(
                    data=self.raw, random_state=self._rand, man_split=man_split, training_percent=args.training_percentage,
                    x_standardization=False, y_min_max_scaling=False)
        self._modelling_data = {
            'training': self._training, 'test': self._test}


class lstmModel(object):
    """
    # Purpose
        Simple or stacked LSTM modelling class

    # Methods
        __init__: load data and other information from DataLoader class and argparser
        simple_lstm_m: setup simple or stacked LSTM model and compile
        bidir_lstm_m: setup bidirectional LSTM model and compile
        lstm_fit: LSTM model fitting
        lstm_eval: additional LSTM model evaluation
    """

    def __init__(self,  model_type, n_timepoints, n_features):
        """
        # Behaviour
            The initilizer loads model configs from arg parser 

        # Public class attributes
            model_type: str. model type
            n_stack: int. number of LSTM stacks
            hidden_units: int. number of hidden units
            epochs: int. number of epochs
            batch_size: int. batch size
            n_timepoints: int. number of timeopints (steps)
            n_features: int. number of features per timepoint
            stateful: bool. if to use stateful LSTM
            dropout: float. dropout rate for LSTM
            dense_activation: str. activation function for the MLP (decision making/output DNN)
            loss: str. loss function
            optimizer: str. Optimizer type
        """
        self.model_type = model_type
        self.n_stack = args.n_stack
        self.hidden_units = args.hidden_units
        self.epochs = args.epochs
        self.batch_size = args.batch
        self.n_timepoints = n_timepoints
        self.n_features = n_features
        self.stateful = args.stateful
        self.dropout = args.dropout_rate
        self.dense_activation = args.dense_activation
        self.loss = args.loss
        self.optimizer = args.optimizer

    def simple_lstm_m(self, n_output=1):
        """
        # Behaviour
            This method uses dropout and batch normalization

        # Public class attributes
            simple_m: simple or stacked LSTM model
            m: the final LSTM model
        """
        # model setup
        self.simple_m = Sequential()
        if self.n_stack > 1:  # if to use stacked LSTM or not
            self.simple_m.add(LSTM(units=self.hidden_units, return_sequences=True,
                                   input_shape=(
                                       self.n_timepoints, self.n_features), stateful=self.stateful, dropout=self.dropout))
            self.simple_m.add(BatchNormalization())
            for _ in range(self.n_stack):
                self.simple_m.add(LSTM(units=self.hidden_units))
                self.simple_m.add(BatchNormalization())
        else:
            self.simple_m.add(LSTM(units=self.hidden_units, input_shape=(
                self.n_timepoints, self.n_features), stateful=self.stateful, dropout=self.dropout))
            self.simple_m.add(BatchNormalization())
        self.simple_m.add(
            Dense(units=n_output, activation=self.dense_activation))

        # model compiling
        self.simple_m.compile(
            loss=self.loss, optimizer=self.optimizer, metrics=['mse', 'accuracy'])
        self.m = self.simple_m

    def bidir_lstm_m(self, n_output=1):
        """
        # Behaviour
            This method uses dropout and batch normalization

        # Public class attributes
            bidir_m: bidirectional LSTM model
            m: the final LSTM model
            m_history: model history with metrices etc
        """
        # model setup
        self.bidir_m = Sequential()
        self.bidir_m.add(Bidirectional(LSTM(units=self.hidden_units, return_sequences=True,
                                            input_shape=(
                                                self.n_timepoints, self.n_features), stateful=self.stateful, dropout=self.dropout)))
        self.bidir_m.add(BatchNormalization())
        self.bidir_m.add(
            Dense(units=n_output, activation=self.dense_activation))
        self.bidir_m.compile(loss=self.loss, optimizer=self.optimizer, metrics=[
                             'mse', 'accuracy'])
        self.m = self.bidir_m

    def lstm_fit(self, trainX, trainY, testX, testY, log_dir=None):
        """
        # Arguments
            trainX: numpy ndarray for training X. shape requirment: n_samples x n_timepoints x n_features
            trainY: numpy ndarray for training Y. shape requirement: n_samples
            testX: numpy ndarray for test X. shape requirment: n_samples x n_timepoints x n_features
            testY: numpy ndarray for test Y. shape requirement: n_samples
            log_dir: str. path to output tensorboard results. It is opitonal

        # Public class attributes
            trainX: numpy ndarray for training X. shape requirment: n_samples x n_timepoints x n_features
            trainY: numpy ndarray for training Y. shape requirement: n_samples
            testX: numpy ndarray for test X. shape requirment: n_samples x n_timepoints x n_features
            testY: numpy ndarray for test Y. shape requirement: n_samples

        # Private class attributes (excluding class property)
            _earlystop_callback: early stop callback
            _tfboard_callback: tensorboard callback
            _callbacks: list. a list of callbacks for model fitting
        """
        # data
        self.trainX = trainX
        self.trainY = trainY
        self.testX = testX
        self.testY = testY

        # callbakcs
        self._earlystop_callback = EarlyStopping(
            monitor='val_loss', patience=5)
        if log_dir:
            self._tfboard_callback = TensorBoard(log_dir=log_dir)
            self._callbacks = [
                self._earlystop_callback, self._tfboard_callback]
        else:
            self._callbacks = [self._earlystop_callback]

        # fitting
        self.m_history = self.m.fit(x=self.trainX, y=self.trainY, epochs=self.epochs,
                                    batch_size=self.batch_size, callbacks=self._callbacks,
                                    validation_data=(self.testX, self.testY),
                                    verbose=True)

    def lstm_eval(self, newX=None, newY=None):
        """
        # Purpose
            Evalutate model performance with new data

        # Arguments
            newX: numpy ndarray for new data X. shape requirment: n_samples x n_timepoints x n_features
            newY: numpy ndarray for new data Y. shape requirement: n_samples
        """
        # evaluate
        if self.model_type == 'regression':
            self._mse = self.m.evaluate(newX, newY, verbose=True)
            self.accuracy = None
        else:
            self._mse, self.accuracy = self.m.evaluate(
                newX, newY, verbose=True)
        self.rmse = math.sqrt(self._mse)


class cvTraining(object):
    """
    # Purpose
        Use cross-validation to train models.

    # Behaviours
        This class uses the LSTM model classes

    # Methods
        __init__: load the CV configuration from arg parser
        cvSplit: calculate sub sample indices for cv according to the cv_type
        cvRun: run the CV modelling process according to the LSTM type
    """

    def __init__(self, training):
        """
        # argument 
            training: pandas dataframe. input data: row is sample.

        # Public class attributes
            cv_type: str. cross validation type
            n_iter: int. number of cv iterations according to cv_type

        # Private class attributes (excluding class property)

        # Class property
        """
        self.cv_type = args.cv_type
        if self.cv_type == 'kfold':
            self.n_iter = args.cv_fold
        elif self.cv_type == 'LOO':
            self.n_iter = training.shape[0]  # number of rows/samples
        else:
            self.n_iter = args.n_monte

        self.monte_test_rate = args.monte_test_rate
        self._rand = args.random_state
        self._model_type = args.model_type
        self._y_var = args.outcome_variable

    def cvSplit(self, training):
        """
        # Public class attributes
            cv_training_idx: list of int array. sample (row) index for cv training data folds
            cv_training_idx: list of int array. sample (row) index for cv test data folds

        # Private class attributes (excluding class property)
            _training: pandas dataframe. input training data. wit X and Y 
            _kfold: sklearn.KFold/sklearn.StratifiedKFold object if cv_type='kfold', according to the model type
            _loo: sklearn.LeaveOneOut object if cv_type='LOO'
            _monte: sklearn.ShuffleSplit/sklearn.StratifiedShuffleSplit object if cv_type='monte', according to the model type
            _train_index: int array. sample (row) index for one cv training data fold
            _test_index: int array. sample (row) index for one cv test data fold
        """
        # atrributes
        self._training = training

        # spliting
        self.cv_training_idx, self.cv_training_idx = list(), list()

        if self.cv_type == 'LOO':  # leave one out, same for both regression and classification models
            self._loo = LeaveOneOut()
            for _train_index, _test_index in self._loo.split(training):
                self.cv_training_idx.append(_train_index)
                self.cv_training_idx.append(_test_index)
        else:
            if self._model_type == 'regression':
                if self.cv_type == 'kfold':
                    self._kfold = KFold(n_splits=self.n_iter,
                                        shuffle=True, random_state=self._rand)
                    for _train_index, _test_index in self._kfold.split(training):
                        self.cv_training_idx.append(_train_index)
                        self.cv_training_idx.append(_test_index)
                else:
                    self._monte = ShuffleSplit(
                        n_splits=self.n_iter, test_size=self.monte_test_rate, random_state=self._rand)
                    for _train_index, _test_index in self._monte.split(training):
                        self.cv_training_idx.append(_train_index)
                        self.cv_training_idx.append(_test_index)
            else:  # classification
                if self.cv_type == 'kfold':  # stratified
                    self._kold = StratifiedKFold(n_splits=self.n_iter,
                                                 shuffle=True, random_state=self._rand)
                    for _train_index, _test_index in self._kold.split(training, training[self._y_var]):
                        self.cv_training_idx.append(_train_index)
                        self.cv_training_idx.append(_train_index)
                else:  # stratified
                    self._monte = StratifiedShuffleSplit(
                        n_splits=self.n_iter, test_size=self.monte_test_rate, random_state=self._rand)
                    for _train_index, _test_index in self._monte.split(training, training[self._y_var]):
                        self.cv_training_idx.append(_train_index)
                        self.cv_training_idx.append(_train_index)

    def cvRun(self):
        """
        # Purpose
            Run the CV training modelling

        # Private class attributes (excluding class property)
            _cv_training: a fold of cv training data
            _cv_test: a fold of cv test data

        """
        # set up data
        for _ in range(self.n_iter):
            self._cv_training, self._cv_test = None, None
            # TBC


# ------ test ------
print(args)
print('\n')
print(os.path.exists(args.file[0]))

mydata = DataLoader()
print(mydata.raw)
print("\n")
print("input file path: {}".format(mydata.file))
print("\n")
print("input file name: {}".format(mydata.filename))
print("\n")
print("number of timepoints in the input file: {}".format(mydata.n_timepoints))
print("\n")
print("number of features in the inpout file: {}".format(mydata.n_features))

print(mydata.modelling_data['training'])

# ------ process/__main__ statement ------
# ------ setup output folders ------
# if __name__ == '__main__':
#     mydata = DataLoader()
