"""
This is a collection of functions for LSTM RNN practices
"""
# ------ libraries ------
import logging
import os

import numpy as np
import pandas as pd  # this is not needed as Keras needs numpy inputs for LSTM
import tensorflow as tf
# from keras.layers import TimeDistributed  # for CNN and encoder-decoder models
from matplotlib import pyplot as plt  # to plot ROC-AUC
# for CNN-related models; for encoder-decoder models
from tensorflow.keras.layers import (LSTM, Bidirectional, Conv1D, ConvLSTM2D,
                                     Dense, Flatten, MaxPooling1D,
                                     RepeatVector, TimeDistributed)
from tensorflow.keras.models import Sequential


# ------ functions ------
def simple_lstm_m(n_steps, n_features, n_output=1, hidden_units=50,
                  output_activation="linear", loss='mean_squared_error', optimizer='adam'):
    """
    Purpose:
        LSTM model function for RNN with only one hidden layer (aka simple)

    Arguments: 
        n_steps: int. number of "time points"
        n_features: int. number of input features
        n_output: int. number of output
        hidden_unit: int. number of hidden unit
        output_activation: string. activation for the output layer, 
                            following the keras Dense() activation argument.
        loss: string. type of loss function
        optimizer: string. type of optimizer

    Details:
        The inpout data should have three dimensions: [sample, time points, features]
        The output_action function "sigmoid" can be used for the min-max scaled data to (0, 1)
    """
    m = Sequential()
    m.add(LSTM(units=hidden_units, activation='relu',
               input_shape=(n_steps, n_features)))
    m.add(Dense(units=n_output, activation=output_activation))
    m.compile(loss=loss, optimizer=optimizer)  # regression study
    return m


def bidirectional_lstm_m(n_steps, n_features, n_output=1, hidden_units=50,
                         output_activation='linear', loss='mean_squared_error', optimizer='adam'):
    """
    Purpose:
        LSTM modle function for RNN with bidirectional structure

    Arguments:
        n_steps: int. number of "time points"
        n_features: int. number of input features
        n_output: int. number of output
        hidden_unit: int. number of hidden unit
        output_activation: string. activation for the output layer, 
                    following the keras Dense() activation argument.
        loss: string. type of loss function
        optimizer: string. type of optimizer

    Details:
        This function requires importing of keras.layers.Bidirectional.
        The output_action function "sigmoid" can be used for the min-max scaled data to (0, 1)

    """
    m = Sequential()
    m.add(Bidirectional(LSTM(units=hidden_units, activation='relu',
                             input_shape=(n_steps, n_features))))
    m.add(Dense(units=n_output, activation=output_activation))
    m.compile(loss=loss, optimizer='adam')
    return m


def stacked_lstm_m(n_steps, n_features, n_output=1, hidden_units=50,
                   output_activation='linear', loss='mean_squared_error', optimizer='adam'):
    """
    Purpose:
        LSTM model function for RNN with stacked hidden layers

    Arguments:
        n_steps: int. number of "time points"
        n_features: int. number of input features
        n_output: int. number of output
        hidden_unit: int. number of hidden unit
        output_activation: string. activation for the output layer, 
            following the keras Dense() activation argument.
        loss: string. type of loss function
        optimizer: string. type of optimizer

    Details:
        Currently the function has two hidden layers
        The output_action function "sigmoid" can be used for the min-max scaled data to (0, 1)
    """
    m = Sequential()
    m.add(LSTM(units=hidden_units, activation='relu', return_sequences=True,
               input_shape=(n_steps, n_features)))
    m.add(LSTM(units=hidden_units, activation='relu'))
    m.add(Dense(units=n_output, activation=output_activation))
    m.compile(loss='mse', optimizer=optimizer)  # regression study
    return m


def encoder_decoder_lstm_m(n_steps, n_features, n_output=1,
                           n_dense_out=1, hidden_units=50,
                           output_avtivation='linear', loss='mean_squared_error', optimizer='adam'):
    """
    Purpose:
        LSTM model function for encoder-decoder models for time series

    Arguments:
        n_steps: int. number of "time points"
        n_features: int. number of input features
        n_output: int. number of output
        hidden_unit: int. number of hidden unit
        output_activation: string. activation for the output layer, 
            following the keras Dense() activation argument.
        loss: string. type of loss function
        optimizer: string. type of optimizer

    Details;
        The encoder is traditionally a Vanilla LSTM model, although other 
        encoder models can be used such as Stacked, Bidirectional, and CNN models.
        The output_action function "sigmoid" can be used for the min-max scaled data to (0, 1)

    """
    m = Sequential()
    m.add(LSTM(units=hidden_units, activation='relu',
               input_shape=(n_steps, n_features)))  # encoder
    m.add(RepeatVector(n=n_output))
    m.add(LSTM(units=hidden_units, activation='relu',
               return_sequences=True))  # decoder
    m.add(TimeDistributed(Dense(units=n_dense_out, activation=output_avtivation)))
    m.compile(loss=loss, optimizer='adam')
    return m


def cnn_lstm_m(n_steps, n_features, n_output=1, hidden_units=50,
               outpout_activiation='linear', loss='mean_squared_error', optimizer='adam'):
    """
    Purpose:
        LSTM model function for CNN-RNN hybrid model. 

    Arguments:
        n_steps: int. number of "time points"
        n_features: int. number of input features
        n_output: int. number of output
        hidden_unit: int. number of hidden unit
        output_activation: string. activation for the output layer, 
            following the keras Dense() activation argument.
        loss: string. type of loss function
        optimizer: string. type of optimizer


    Details:
        This function requires the CNN modules from keras

        CNN-LSTM VERY brief principles:
        The first step is to split the input sequences into subsequences that can be
        processed by the CNN model. For example, we can first split our univariate time
        series data into input/output samples with four steps as input and one as output.

        Each sample can then be split into two sub-samples, each with two time steps (filter).
        The CNN can interpret each subsequence of two time steps (convolution) and provide a time series
        of interpretations of the subsequences to the LSTM model to process as input.

        We want to reuse the same CNN model when reading in each sub-sequence of data separately.

        This can be achieved by wrapping the entire CNN model in a TimeDistributed wrapper that
        will apply the entire model once per input, in this case, once per input subsequence.

        The CNN model first has a convolutional layer for reading across the subsequence that
        requires a number of filters and a kernel size to be specified.
        The number of filters is the number of reads or interpretations of the input sequence.
        The kernel size is the number of time steps included of each ‘read’ operation of the input sequence.

        The convolution layer is followed by a max pooling layer that distills the filter maps
        down to 1/4 of their size that includes the most salient features. These structures are
        then flattened down to a single one-dimensional vector to be used as a single input time
        step to the LSTM layer.

        The input X needs to be reshaped into:
        (sample x filter size x timpoints per filter size x n_features)
        before fitting

        The output_action function "sigmoid" can be used for the min-max scaled data to (0, 1)
    """
    m = Sequential()
    m.add(TimeDistributed(Conv1D(filters=64, kernel_size=1,
                                 activation='relu'), input_shape=(None, n_steps, n_features)))
    m.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    m.add(TimeDistributed(Flatten()))
    m.add(LSTM(units=hidden_units, activation='relu'))
    m.add(Dense(units=n_output))
    m.compile(optimizer=optimizer, loss=loss)
    return m


# def conv_lstm_m(n_seq, n_steps, n_features, n_row=1, n_output=1, hidden_units=50, optimizer='adam'):
#     """
#     A type of LSTM related to the CNN-LSTM is the ConvLSTM, where the convolutional
#     reading of input is built directly into each LSTM unit.

#     The ConvLSTM was developed for reading two-dimensional spatial-temporal data,
#     but can be adapted for use with univariate time series forecasting.

#     We can define the ConvLSTM as a single layer in terms of the number of filters
#     and a two-dimensional kernel size in terms of (rows, columns). As we are working
#     with a one-dimensional series, the number of rows is always fixed to 1 in the kernel.

#     The input X needs to be reshaped into:
#     sample x filter size x rows x timpoints per filter size x n_features
#     before fitting
#     """
#     m = Sequential()
#     m.add(ConvLSTM2D(filiters=64, kernal_size=(1, 2), activation='relu',
#                      input_shape=(n_seq, n_row, n_steps, n_features)))
#     m.add(Dense(units=n_output))
#     m.compile(loss='mse', optimizer=optimizer)
#     return m
