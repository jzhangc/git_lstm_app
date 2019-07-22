"""
Utility functions
"""
# ------ libraries ------
import logging
import pandas as pd
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout  # fully connnected layer
from sklearn.metrics import roc_curve, auc  # calculate ROC-AUC
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt  # to plot ROC-AUC


# ------ functions ------
def logging_func(filepath, formatter='%(message)s'):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(message)s')
    file_handler = logging.FileHandler(filepath)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def split_sequence_univar(sequence, n_steps):
    """
    Purpose: 
        This function is specific for univariate time series data splitting

    Arguments
        sequence: input data array
        n_setps: input subsetting unit size (i.e. number of time points)

    Details:
        split an array into a matrix (X) and outcome (y)

    >>> raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    >>> n_steps = 3
    >>> X, y = split_sequence_univar(raw_seq, n_steps)
    X,				y
    10, 20, 30		40
    20, 30, 40		50
    30, 40, 50		60
    ...
    """
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of the pattern
        end_ix = i + n_steps
        # check if it exceeds the range
        if end_ix > len(sequence) - 1:
            break
        # gather input and output
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


# split a univariate sequence into samples
def split_sequence_univar_multiY(sequence, n_steps_in, n_steps_out):
    """
    This is function is similar to the one above, but produces two values for y

    sequence: input data array
    n_setps_in: input subsetting unit size (i.e. number of time points)
    n_setps_out: output values

    >>> raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    >>> X, Y = split_sequence(raw_seq, n_steps_in, n_steps_out)
    >>> for i in range(len(X)):
        ..      print(X[i], y[i])
    [10 20 30] [40 50]
    [20 30 40] [50 60]
    [30 40 50] [60 70]
    [40 50 60] [70 80]
    [50 60 70] [80 90]
    """
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


# split a multivariate sequence into samples
def split_sequences_multivar(sequences, n_steps):
    """
    This function is specific for multivariate time series data splitting
    sequence: input data array
    n_setps: input subsetting unit size (i.e. number of time points)

    >>> import numpy as np
    >>> in_seq1 = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
    >>> in_seq2 = np.array([15, 25, 35, 45, 55, 65, 75, 85, 95])
    >>> out_seq = np.array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])
    >>> in_seq1 = in_seq1.reshape(len(in_seq1), 1)
    >>> in_seq2 = in_seq2.reshape(len(in_seq2), 1)
    >>> out_seq = out_seq.reshape(len(out_seq), 1)
    >>> data = np.hstack((in_seq1, in_seq2, out_seq))
    >>> X, y = split_sequences_multivar(sequence=data, n_steps=3)
    >>> X
    [[[10 15]
      [20 25]
      [30 35]]

     [[20 25]
      [30 35]
      [40 45]]

     [[30 35]
      [40 45]
      [50 55]]

     [[40 45]
      [50 55]
      [60 65]]

     [[50 55]
      [60 65]
      [70 75]]

     [[60 65]
      [70 75]
      [80 85]]

     [[70 75]
      [80 85]
      [90 95]]]

    >>> y
    [ 65  85 105 125 145 165 185]

    """
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def split_sequences_multivar_multiY(sequences, n_steps_in, n_steps_out):
    """
    This function is similar to the previous one but with mulitple Y
    n_setps_in: input subsetting unit size (i.e. number of time points)
    n_steps_out: output unit size
    """
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out-1
        # check if we are beyond the dataset
        if out_end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-
                                 1], sequences[end_ix-1:out_end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


# split a multivariate sequence into samples
def split_sequences_multivar_para(sequences, n_steps):
    """
    This function is specific for multivariate and multi-output time series data splitting
    sequence: input data array
    n_setps: input subsetting unit size (i.e. number of time points)

    >>> import numpy as np
    >>> in_seq1 = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
    >>> in_seq2 = np.array([15, 25, 35, 45, 55, 65, 75, 85, 95])
    >>> out_seq = np.array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])
    >>> in_seq1 = in_seq1.reshape(len(in_seq1), 1)
    >>> in_seq2 = in_seq2.reshape(len(in_seq2), 1)
    >>> out_seq = out_seq.reshape(len(out_seq), 1)
    >>> data = np.hstack((in_seq1, in_seq2, out_seq))
    >>> X, y = split_sequences_multivar(sequence=data, n_steps=3)
    >>> print(X)
    [[[ 10  10  25]
      [ 20  20  45]
      [ 30  30  65]]

     [[ 20  20  45]
      [ 30  30  65]
      [ 40  40  85]]

     [[ 30  30  65]
      [ 40  40  85]
      [ 50  50 105]]

     [[ 40  40  85]
      [ 50  50 105]
      [ 60  60 125]]

     [[ 50  50 105]
      [ 60  60 125]
      [ 70  70 145]]

     [[ 60  60 125]
      [ 70  70 145]
      [ 80  80 165]]]

    >>> (y)
     [[ 40  40  85]
      [ 50  50 105]
      [ 60  60 125]
      [ 70  70 145]
      [ 80  80 165]
      [ 90  90 185]]
    """
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


# split a multivariate sequence into samples
def split_sequences_multivar_multiY_para(sequences, n_steps_in, n_steps_out):
    """
    This function is similar to the previous but with mulitple Y and outputs 
    per timepoint

    n_setps_in: input subsetting unit size (i.e. number of time points)
    n_steps_out: output unit size
    """
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the dataset
        if out_end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)
