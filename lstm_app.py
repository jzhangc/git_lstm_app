#!/usr/bin/env python3
"""
Current objectives:
[x] 1. Test argparse
[x] 2. Test output directory creation
[X] 3. Test file reading
[X] 4. Test file processing
[X] 5. Test training
[X] 6. Folder setup
[ ] 7. Save models and data
[ ] 8. Diplay messages
[X] 9. Code cleanup, generalization and optimization

NOTE
All the argparser inputs are loaded from method arguments, making the class more portable, i.e. not tied to
the application.

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
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.layers import (LSTM, BatchNormalization, Bidirectional,
                                     Dense, Dropout)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam

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
---------------------------------- Description ---------------------------------
LSTM regression/classification modelling using multiple-timepoint MEG connectome.
Currently, the program only accepts same feature size per timepoint.
--------------------------------------------------------------------------------
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
        help='list of str. names of the annotation columns in the input data, excluding the outcome variable.')
add_arg("-n", '--n_timepoints', type=int, default=None,
        help='int. Number of timepoints. NOTE: only needed with single file processing')
add_arg('-y', '--outcome_var', type=str, default=None,
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
add_arg('-lr', '--learning_rate', type=float, default=0.001,
        help='foalt. Learning rate for the optimizer. Note: use 0.01 for sgd.')
add_arg('-u', '--hidden_units', type=int, default=50,
        help='int. Number of hidden unit for the LSTM network')
add_arg('-x', '--dropout_rate', type=float, default=0.0,
        help='float, 0.0~1.0. Dropout rate for LSTM models . 0.0 means no dropout.')
add_bool_arg(parser=parser, name='stateful', input_type='flag', default=False,
             help="Use stateful LSTM for modelling.")

add_arg('-o', '--output_dir', type=str,
        default='.', help='str. Output directory. NOTE: not an absolute path, only relative to working directory -w/--working_dir')

add_bool_arg(parser=parser, name='verbose', input_type='flag', default=False,
             help='Verbose or not')


add_bool_arg(parser=parser, name='plot', input_type='flag',
             help='Explort a scatter plot', default=False)
add_arg('-j', '--plot-type', type=str,
        choices=['scatter', 'bar'], default='scatter', help='str. Plot type')

args = parser.parse_args()
# check the arguments. did not use parser.error as error() has fancy colours
if not args.sample_id_var:
    error('-s/--sample_id_var missing.',
          'Be sure to set the following: -s/--sample_id_var, -n/--n_timepoints, -y/--outcome_var, -a/--annotation_vars')
if not args.n_timepoints:
    error('-n/--n_timepoints flag missing.',
          'Be sure to set the following: -s/--sample_id_var, -n/--n_timepoints, -y/--outcome_var, -a/--annotation_vars')
if not args.outcome_var:
    error('-y/--outcome_var flag missing.',
          'Be sure to set the following: -s/--sample_id_var, -n/--n_timepoints, -y/--outcome_var, -a/--annotation_vars')
if len(args.annotation_vars) < 1:
    error('-a/--annotation_vars missing.',
          'Be sure to set the following: -s/--sample_id_var, -n/--n_timepoints, -y/--outcome_var, -a/--annotation_vars')

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
        This class uses the custom error() function. So be sure to load it. 

    # Methods
        __init__: load data and other information from argparser, as well as class label encoding for classification study
        data_split: set up data for model training. No data splitting for the "CV only" mode.

    # Class property
        modelling_data: set up the data for model training. data is split if necessary.
            returns a dict object with 'training' and 'test' items

            _m_data: dict. output dictionary
            _training: pandas dataframe. data for model training.
            _test: pandas dataframe. holdout test data. Only available without the "--cv_only" flag
    """

    def __init__(self, cwd, file, outcome_var, annotation_vars, n_timepoints, sample_id_var,
                 model_type,
                 cv_only,
                 man_split, holdout_samples, training_percentage, random_state, verbose):
        """
        # Arguments
            cwd: str. working directory
            file: str. complete input file path. "args.file[0]" from argparser
            outcome_var: str. variable nanme for outcome. Only one is accepted for this version. "args.outcome_var" from argparser
            annotation_vars: list of strings. Column names for the annotation variables in the input dataframe, EXCLUDING outcome variable.
                "args.annotation_vars" from argparser
            n_timepints: int. number of timepoints. "args.n_timepoints" from argparser
            sample_id_var: str. variable used to identify samples. "args.sample_id_var" from argparser
            model_type: str. model type, classification or regression
            cv_only: bool. If to split data into training and holdout test sets. "args.cv_only" from argparser
            man_split: bool. If to use manual split or not. "args.man_split" from argparser
            holdout_samples: list of strings. sample IDs for holdout sample, when man_split=True. "args.holdout_samples" from argparser
            training_percentage: float, betwen 0 and 1. percentage for training data, when man_split=False. "args.training_percentage" from argparser
            random_state: int. random state
            verbose: bool. verbose. "args.verbose" from argparser

        # Public class attributes
            Below are attributes read from arguments
                self.cwd
                self.model_type 
                self.file
                self.outcome_var
                self.annotation_vars
                self.n_timepints
                self.cv_only
                self.holdout_samples
                self.training_percentage
                self.rand: int. random state

            self.y_var: single str list. variable nanme for outcome
            self.filename: str. input file name without extension
            self.raw: pandas dataframe. input data
            self.complete_annot_vars: list of strings. column names for the annotation variables in the input dataframe, INDCLUDING outcome varaible
            self.n_features: int. number of features    
            self.le: sklearn LabelEncoder for classification study  

        # Private class attributes (excluding class properties)
            self._basename: str. complete file name (with extension), no path
            self._n_annot_col: int. number of annotation columns 
        """
        # setup working director
        self.cwd = cwd

        # random state
        self.rand = random_state
        self.verbose = verbose

        # load files
        self.model_type = model_type
        # convert to a list for training_test_spliter_final() to use
        self.outcome_var = outcome_var
        self.annotation_vars = annotation_vars
        self.y_var = [self.outcome_var]

        # args.file is a list. so use [0] to grab the string
        self.file = os.path.join(self.cwd, file)
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
            self.complete_annot_vars = self.annotation_vars + self.y_var
            self._n_annot_col = len(self.complete_annot_vars)
            self.n_timepoints = n_timepoints
            self.n_features = int(
                (self.raw.shape[1] - self._n_annot_col) // self.n_timepoints)  # pd.shape[1]: ncol

            self.cv_only = cv_only
            self.sample_id_var = sample_id_var
            self.holdout_samples = holdout_samples
            self.training_percentage = training_percentage

        if self.model_type == 'classification':
            self.le = LabelEncoder()
            self.le.fit(self.raw[self.y_var])
            self.raw[self.y_var] = self.le.transform(self.raw[self.y_var])

        # call setter here
        self.modelling_data = man_split

    @property
    def modelling_data(self):
        # print("called getter") # for debugging
        return self._modelling_data

    @modelling_data.setter
    def modelling_data(self, man_split):
        # print("called setter") # for debugging
        if self.cv_only:  # only training is stored
            self._training, self._test = self.raw, None
        else:
            # training and holdout test data split
            if args.man_split:
                # manual data split: the checks happen in the training_test_spliter_final() function
                self._training, self._test, _, _ = training_test_spliter_final(data=self.raw, random_state=self.rand,
                                                                               man_split=man_split, man_split_colname=self.sample_id_var,
                                                                               man_split_testset_value=self.holdout_samples,
                                                                               x_standardization=False, y_min_max_scaling=False)
            else:
                self._training, self._test, _, _ = training_test_spliter_final(
                    data=self.raw, random_state=self.rand, man_split=man_split, training_percent=self.training_percentage,
                    x_standardization=False, y_min_max_scaling=False)
        self._modelling_data = {
            'training': self._training, 'test': self._test}


class lstmModel(object):
    """
    # Purpose
        Simple or stacked LSTM modelling class

    # Details
        This class uses the custom error() function. So be sure to load it.

    # Methods
        __init__: load data and other information from DataLoader class and argparser
        simple_lstm_m: setup simple or stacked LSTM model and compile
        bidir_lstm_m: setup bidirectional LSTM model and compile
        lstm_fit: LSTM model fitting
        lstm_eval: additional LSTM model evaluation
    """

    def __init__(self, model_type, n_timepoints, n_features,
                 n_stack, hidden_units, epochs, batch_size, stateful, dropout, dense_activation,
                 loss, optimizer, learning_rate, verbose):
        """
        # Behaviour
            The initilizer loads model configs

        # Arguments
            model_type: str. model type, "classification" or "regression". "args.model_type" from argparser, or DataLoader.model_type
            n_timepoints: int. number of timeopints (steps). "n_timepoint" from argparser, or DataLoader.n_timepoint
            n_features: int. number of features per timepoint. could be from the DataLoader class attribute DataLoader.n_features
            n_stack: int. number of (simple) LSTM stacks. "args.n_stack" from argparser
            hidden_units: int. number of hidden units. "args.hidden_units" from argparser
            epochs: int. number of epochs. "args.epochs" from argparser
            batch_size: int. batch size. "args.batch" from argparser
            stateful: bool. if to use stateful LSTM. "args.stateful" from argparser
            dropout: float. dropout rate for LSTM. "args.dropout_rate" from argparser
            dense_activation: str. activation function for the MLP (decision making/output DNN). "args.dense_activation" from argparser
            loss: str. loss function. "args.loss" from argparser
            optimizer: str. optimizer. "args.optimizer" from argparser
            learning_rate: float. leanring rate for optimizer . "args.learning_rate" from argparser
            verbose: str. Optimizer type. "args.verbose" from argparser, or DataLoader.verbose. But it is recommneded to set it separately

        # Public class attributes
            Below: attributes read from arguments
                self.model_type
                self.n_timepoints
                self.n_features
                self.n_stack
                self.hidden_units
                self.epochs
                self.batch_size
                self.stateful
                self.dropout
                self.dense_activation
                self.loss
                self.optimizer
                self.lr: learning rate 

        # Private class attributes (excluding class propterties)
            Below: private attributes read from arguments 
                self._verbose

            self._opt: working optimizer with custom learning rate
        """
        self.model_type = model_type
        self.n_timepoints = n_timepoints
        self.n_features = n_features

        self.n_stack = n_stack
        self.hidden_units = hidden_units
        self.epochs = epochs
        self.batch_size = batch_size
        self.stateful = stateful
        self.dropout = dropout
        self.dense_activation = dense_activation
        self.loss = loss
        self._verbose = verbose
        self.lr = learning_rate

        # setup optimizer
        self.optimizer = optimizer
        if self.optimizer == 'adam':
            self._opt = Adam(lr=self.lr)
        else:
            self._opt = SGD(lr=self.lr)

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
            loss=self.loss, optimizer=self._opt, metrics=['mse', 'accuracy'])
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
        self.bidir_m.compile(loss=self.loss, optimizer=self._opt, metrics=[
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
                                    verbose=self._verbose)

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

    def __init__(self, training, n_features, lstm_type,
                 cv_type, cv_fold, n_monte, monte_test_rate,
                 model_type, outcome_var, annotation_vars,
                 random_state, verbose):
        """
        # argument
            training: pandas dataframe. input data: row is sample
            n_features: int. number of features per timepoint. could be from the DataLoader class attribute DataLoader.n_features attribute
            lstm_type: str. lstm type. "args.lstm_type" from argparser
            cv_type: str. cross validation type. "args.cv_type" from argparser
            cv_fold: int. number of fold when cv_type="LOO" or "kfold". "args.cv_fold" from argparser
            n_monte: int. number of Monte Carlo iteratins when cv_type="monte". "args.n_monte" from argparser
            monte_test_rate: float, between 0 and 1. resampling percentage for test set when cv_type="monte"
            model_type: str. model type, "classification" or "regression". "args.model_type" from argparser, or DataLoader.model_type attribute
            outcome_var: str. variable nanme for outcome. Only one is accepted for this version. "args.outcome_var" from argparser, or DataLoader.outcome_var
            annotation_vars: list of strings. Column names for the annotation variables in the input dataframe, EXCLUDING outcome variable.
                "args.annotation_vars" from argparser, or DataLoader.annotation_vars attribute
            random_state: int. random state. "args.random_state" from argparser, or DataLoader.rand attribute
            verbose: bool. verbose. "args.verbose", or DataLoader.verbose


        # Public class attributes
            Below are private attribute(s) read from arguments
                self.cv_type
                self.lstm_type

            self.n_iter: int. number of cv iterations according to cv_type

        # Private class attributes (excluding class property)
            Below are private attribute(s) read from arguments
                self._outcome_var
                self._annoation_vars
                self._n_features
                self._rand
                self._model_type
                self._verbose

            self._y_var: single str list. name of the outcome variable
            self._complete_annot_vars: list of strings. column names for the annotation variables in the input dataframe, INDCLUDING outcome varaible. 
            self._verbose
        """
        self.training = training
        self.cv_type = cv_type
        self.lstm_type = lstm_type

        if self.cv_type == 'kfold':
            self.n_iter = cv_fold
        elif self.cv_type == 'LOO':
            self.n_iter = training.shape[0]  # number of rows/samples
        else:
            self.n_iter = n_monte
            self.monte_test_rate = monte_test_rate

        self._n_features = n_features
        self._outcome_var = outcome_var
        self._annotation_vars = annotation_vars  # list of strings
        self._y_var = [self._outcome_var]
        self._complete_annot_vars = self._annotation_vars + self._y_var

        self._rand = random_state
        self._model_type = model_type
        self._verbose = verbose

    def cvSplit(self):
        """
        # Public class attributes
            cv_training_idx: list of int array. sample (row) index for cv training data folds
            cv_test_idx: list of int array. sample (row) index for cv test data folds

        # Private class attributes (excluding class property)
            _training: pandas dataframe. input training data. wit X and Y
            _kfold: sklearn.KFold/sklearn.StratifiedKFold object if cv_type='kfold', according to the model type
            _loo: sklearn.LeaveOneOut object if cv_type='LOO'
            _monte: sklearn.ShuffleSplit/sklearn.StratifiedShuffleSplit object if cv_type='monte', according to the model type
            _train_index: int array. sample (row) index for one cv training data fold
            _test_index: int array. sample (row) index for one cv test data fold
        """
        # spliting
        self.cv_training_idx, self.cv_training_idx = list(), list()

        if self.cv_type == 'LOO':  # leave one out, same for both regression and classification models
            self._loo = LeaveOneOut()
            for _train_index, _test_index in self._loo.split(self.training):
                self.cv_training_idx.append(_train_index)
                self.cv_training_idx.append(_test_index)
        else:
            if self._model_type == 'regression':
                if self.cv_type == 'kfold':
                    self._kfold = KFold(n_splits=self.n_iter,
                                        shuffle=True, random_state=self._rand)
                    for _train_index, _test_index in self._kfold.split(self.training):
                        self.cv_training_idx.append(_train_index)
                        self.cv_training_idx.append(_test_index)
                else:
                    self._monte = ShuffleSplit(
                        n_splits=self.n_iter, test_size=self.monte_test_rate, random_state=self._rand)
                    for _train_index, _test_index in self._monte.split(self.training):
                        self.cv_training_idx.append(_train_index)
                        self.cv_training_idx.append(_test_index)
            else:  # classification
                if self.cv_type == 'kfold':  # stratified
                    self._kold = StratifiedKFold(n_splits=self.n_iter,
                                                 shuffle=True, random_state=self._rand)
                    for _train_index, _test_index in self._kold.split(self.training, self.training[self._y_var]):
                        self.cv_training_idx.append(_train_index)
                        self.cv_training_idx.append(_train_index)
                else:  # stratified
                    self._monte = StratifiedShuffleSplit(
                        n_splits=self.n_iter, test_size=self.monte_test_rate, random_state=self._rand)
                    for _train_index, _test_index in self._monte.split(self.training, self.training[self._y_var]):
                        self.cv_training_idx.append(_train_index)
                        self.cv_training_idx.append(_train_index)

    def cvRun(self, working_dir, output_dir, *args, **kwargs):
        """
        # Purpose
            Run the CV training modelling. This class is less portable, as it is tied to the lstmModel class.

        # Arguments
            working_dir: str. working directory. "args.working_dir" from argparser, or DataLoader.cwd attribute
            output_dir: str. output directory. "args.output_dir" from argparser

            Below:

        # Public class attributes
            self.cv_m_ensemble
            self.cv_m_history_ensemble
            self.cv_test_accuracy_ensemble
            self.cv_test_rmse_ensemble

        # Private class attributes (excluding class property)
            below: private attributes read from arguments
                self._working_dir
                self._output_dir

            self._res_dir: str. working_dir + output_dir
            self._cv_training: a fold of cv training data
            self._cv_test: a fold of cv test data

        """
        # check and set up output path
        if self._verbose:
            print("Set up results directory...", end=' ')
        self._wd = working_dir
        self._output_dir = output_dir
        self._res_dir = os.path.join(self._wd, self._output_dir)

        if not os.path.exists(self._res_dir):  # set up out path
            os.mkdir(self._res_dir)
        else:
            self._res_dir = self._res_dir + "_" + datetime.now().strftime("%Y%m%d-%H%M%S")
            os.mkdir(self._res_dir)

        self._tfborad_dir = os.path.join(
            self._res_dir, 'tensorboard_res')  # set up tf board path
        # below: no need to check as self._res_dir is new for sure
        os.mkdir(self._tfborad_dir)
        if self._verbose:
            print('Done!')

        # set up data
        self.cv_m_ensemble, self.cv_m_history_ensemble = list(), list()
        self.cv_test_accuracy_ensemble, self.cv_test_rmse_ensemble = list(), list()
        for i in range(self.n_iter):
            iter_id = str(i+1)
            if self._verbose:
                print('cv iteration: ', iter_id)
            # below: .copy for pd dataframe makes an explicit copy, avoiding Pandas SettingWithCopyWarning
            self._cv_training, self._cv_test = self.training.iloc[self.cv_training_idx[i],
                                                                  :].copy(), self.training.iloc[self.cv_training_idx[i], :].copy()

            # x standardization
            self._cv_train_scaler_X = StandardScaler()
            self._cv_training[self._cv_training.columns[~self._cv_training.columns.isin(self._complete_annot_vars)]] = self._cv_train_scaler_X.fit_transform(
                self._cv_training[self._cv_training.columns[~self._cv_training.columns.isin(self._complete_annot_vars)]])
            self._cv_test[self._cv_test.columns[~self._cv_test.columns.isin(self._complete_annot_vars)]] = self._cv_train_scaler_X.transform(
                self._cv_test[self._cv_test.columns[~self._cv_test.columns.isin(self._complete_annot_vars)]])

            # process outcome variable
            if self._model_type == 'regression':
                self._cv_train_scaler_Y = MinMaxScaler(feature_range=(0, 1))
                self._cv_training[self._cv_training.columns[self._cv_training.columns.isin(self._y_var)]] = self._cv_train_scaler_Y.fit_transform(
                    self._cv_training[self._cv_training.columns[self._cv_training.columns.isin(self._y_var)]])
                self._cv_test[self._cv_test.columns[self._cv_test.columns.isin(self._y_var)]] = self._cv_train_scaler_Y.fit_transform(
                    self._cv_test[self._cv_test.columns[self._cv_test.columns.isin(self._y_var)]])

            # convert data to np arrays
            self._cv_train_x, self._cv_train_y = longitudinal_cv_xy_array(input=self._cv_training, Y_colnames=self._y_var,
                                                                          remove_colnames=self._annotation_vars, n_features=self._n_features)
            self._cv_test_x, self._cv_test_y = longitudinal_cv_xy_array(input=self._cv_test, Y_colnames=self._y_var,
                                                                        remove_colnames=self._annotation_vars, n_features=self._n_features)

            # training
            # below: make sure to have all the arguments
            cv_lstm = lstmModel(*args, **kwargs)

            if self.lstm_type == "simple":
                cv_lstm.simple_lstm_m()
            else:  # stacked
                cv_lstm.bidir_lstm_m()

            cv_lstm.lstm_fit(trainX=self._cv_training, trainY=self._cv_train_y,
                             testX=self._cv_test_x, testY=self._cv_test_y, log_dir=os.path.join(self._tfborad_dir, 'cv_iter_'+iter_id))
            cv_lstm.lstm_eval(newX=self._cv_test_x, newY=self._cv_test_y)

            # saving and exporting
            cv_lstm.m.save(os.path.join(
                self._res_dir, 'lstm_cv_model_'+'iter_'+str(i+1)+'.h5'))
            self.cv_m_ensemble.append(cv_lstm.m)
            self.cv_m_history_ensemble.append(cv_lstm.m_history)
            self.cv_test_accuracy_ensemble.append(cv_lstm.accuracy)
            self.cv_test_rmse_ensemble.append(cv_lstm.rmse)


# ------ local variables ------
if args.working_dir:
    cwd = args.working_dir
else:
    cwd = os.getcwd()

# ------ test ------
print(args)
print('\n')
print(os.path.exists(args.file[0]))

mydata = DataLoader(
    cwd=cwd, file=args.file[0],
    outcome_var=args.outcome_var, annotation_vars=args.annotation_vars,
    sample_id_var=args.sample_id_var, n_timepoints=args.n_timepoints,
    model_type=args.model_type, cv_only=args.cv_only,
    man_split=args.man_split, holdout_samples=args.holdout_samples, training_percentage=args.training_percentage,
    random_state=args.random_state, verbose=args.verbose)
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
