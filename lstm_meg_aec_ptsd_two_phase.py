"""
LSTM test for MEG AEC connectivity PTSD longitudinal data
"""

import math
# ------ libraries ------
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.callbacks import Callback, ModelCheckpoint
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from custom_functions.data_processing import (inverse_norm_y,
                                              training_test_spliter)
from custom_functions.lstm_functions import (bidirectional_lstm_m,
                                             encoder_decoder_lstm_m,
                                             simple_lstm_m, stacked_lstm_m)
from custom_functions.plot_functions import epochs_loss_plot, y_yhat_plot
from custom_functions.util_functions import logging_func

# ------ housekeeping ------
# OMP issue
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# logging
tf.logging.set_verbosity(tf.logging.ERROR)

# set the RNG
np.random.seed(2)


# ------ custom functions ------
# ------ script ------
# ---- working directory
main_dir = os.path.abspath('./')
dat_dir = os.path.join(main_dir, 'data')
res_dir = os.path.join(main_dir, 'results')
log_dir = os.path.join(main_dir, 'log')

# ---- setup logger
logger = logging_func(filepath=os.path.join(
    log_dir, 'lstm_meg_ptsd_longitudinal.log'))

# ---- import data
raw = pd.read_csv(os.path.join(
    dat_dir, 'lstm_aec_phases_freq8.csv'), engine='python')
freq = 'freq8'
raw.shape
raw.iloc[0:5, 0:5]


# ---- generate training and test sets with min-max normalization
training, test, scaler_X, scaler_Y = training_test_spliter(
    data=raw, training_percent=0.9, random_state=9, min_max_scaling=True,
    scale_column_as_y=['PCL'],
    scale_column_to_exclude=['subject', 'PCL', 'group'])

training.shape
test.shape

# ---- set up X and y
# -- training
training_X = training[training.columns[~training.columns.isin(
    ['subject', 'PCL', 'group'])]]  # exclude the annotation columns using "not" operator "~"
n_features = int(training_X.shape[1]/2)  # used for test data too
print(training_X.columns)
training_X_phase1, training_X_phase2 = training_X.iloc[:,
                                                       0:n_features], training_X.iloc[:, n_features:]

lstm_training_X = list()
for i in range(len(training_X_phase1)):
    arr = np.array([training_X_phase1.values[i, ],
                    training_X_phase2.values[i, ]])
    lstm_training_X.append(arr)
lstm_training_X = np.array(lstm_training_X)
lstm_training_y = np.array(training.loc[:, "PCL"])

lstm_training_X.shape  # sample x timepoints x features
lstm_training_y.shape
for i in range(len(lstm_training_X)):
    print(lstm_training_X[i], lstm_training_y[i])

# -- test
test_X = test[test.columns[~test.columns.isin(
    ['subject', 'PCL', 'group'])]]
print(test_X.columns)
test_X_phase1, test_X_phase2 = test_X.iloc[:,
                                           0:n_features], test_X.iloc[:, n_features:]

lstm_test_X = list()
for i in range(len(test_X_phase1)):
    arr = np.array([test_X_phase1.values[i, ],
                    test_X_phase2.values[i, ]])
    lstm_test_X.append(arr)
lstm_test_X = np.array(lstm_test_X)
lstm_test_y = np.array(test.loc[:, "PCL"])

lstm_test_X.shape  # sample x timepoints x features
lstm_test_y.shape
for i in range(len(lstm_test_X)):
    print(lstm_test_X[i], lstm_test_y[i])

# ------ modelling
# simple LSTM
simple_model = simple_lstm_m(n_steps=2, n_features=n_features, hidden_units=6)
simple_model_history = simple_model.fit(
    x=lstm_training_X, y=lstm_training_y, epochs=400,
    batch_size=29, callbacks=None, verbose=True)

# stacked LSTM model
stacked_model = stacked_lstm_m(
    n_steps=2, n_features=n_features, hidden_units=6)
stacked_model_history = stacked_model.fit(
    x=lstm_training_X, y=lstm_training_y, epochs=400,
    batch_size=29, callbacks=None,
    verbose=True)

# bidrecitional LSTM model
bidir_model = bidirectional_lstm_m(
    n_steps=2, n_features=n_features, hidden_units=6)
bidir_model_history = bidir_model.fit(
    x=lstm_training_X, y=lstm_training_y, epochs=450,
    batch_size=29, callbacks=None,
    verbose=True)

# y needs to be in samples, timesteps, features: n_feature for the outpout is 1
edc_y = lstm_training_y.reshape(lstm_training_y.shape[0], 1, 1)
enc_dec_model = encoder_decoder_lstm_m(
    n_steps=2, n_features=n_features, n_output=1, hidden_units=6)
enc_dec_model_history = enc_dec_model.fit(
    x=lstm_training_X, y=edc_y, epochs=450, batch_size=29, verbose=True)

# ---- plotting
histories = [simple_model_history, stacked_model_history,
             bidir_model_history, enc_dec_model_history]

epochs_loss_plot(filepath=os.path.join(res_dir, freq+'_simple_loss.pdf'),
                 plot_title='Simple LSTM model',
                 model_history=simple_model_history)

epochs_loss_plot(filepath=os.path.join(res_dir, freq+'_stacked_loss.pdf'),
                 plot_title='Stacked LSTM model',
                 model_history=stacked_model_history)

epochs_loss_plot(filepath=os.path.join(res_dir, freq+'_bidir_loss.pdf'),
                 plot_title='Bidirectional LSTM model',
                 model_history=bidir_model_history)

epochs_loss_plot(filepath=os.path.join(res_dir, freq+'_enc-dec_loss.pdf'),
                 plot_title='Encoder-Decoder LSTM model',
                 model_history=enc_dec_model_history)

# ---- predict
# raw predict
training_y_hat = simple_model.predict(lstm_training_X)
test_y_hat = simple_model.predict(lstm_test_X)

training_y_hat = stacked_model.predict(lstm_training_X)
test_y_hat = stacked_model.predict(lstm_test_X)

training_y_hat = bidir_model.predict(lstm_training_X)
test_y_hat = bidir_model.predict(lstm_test_X)

training_y_hat = enc_dec_model.predict(lstm_training_X)
test_y_hat = enc_dec_model.predict(lstm_test_X)

# RMSE calculation
training_rmse = math.sqrt(mean_squared_error(
    lstm_training_y[0:], training_y_hat[:, 0]))
test_rmse = math.sqrt(mean_squared_error(lstm_test_y[0:], test_y_hat[:, 0]))

print('Training RMSE: {:.2f}'.format(training_rmse))
print('Test RMSE: {:.2f}'.format(test_rmse))

# -- we need to invert the predicted resutls to restore the raw data unit (passengers)
training_y_hat, test_y_hat = inverse_norm_y(
    training_y=training_y_hat, test_y=test_y_hat, scaler=scaler_Y)

# ---- plot
# -- data for plotting
# NOTE: run once!
y_plot = np.concatenate([lstm_training_y, lstm_test_y])
y_plot = scaler_Y.inverse_transform(y_plot.reshape(y_plot.shape[0], 1))
y_plot = y_plot.reshape(y_plot.shape[0], )

# -- plotting
y_yhat_plot(filepath=os.path.join(res_dir, freq+'_bidir.performance.pdf'),
            y=y_plot,
            training_yhat=training_y_hat, test_yhat=test_y_hat,
            plot_title='Bidirectional LSTM model prediction plot',
            ylabel='PCL', xlabel='Subjects', plot_style='classic')

# for enc-dec model
training_y_hat = training_y_hat.reshape(training_y_hat.shape[0], )
test_y_hat = test_y_hat.reshape(test_y_hat.shape[0], )
y_yhat_plot(filepath=os.path.join(res_dir, freq+'_enc-dec.performance.pdf'),
            y=y_plot,
            training_yhat=training_y_hat, test_yhat=test_y_hat,
            plot_title='Encoder-Decoder LSTM model prediction plot',
            ylabel='PCL', xlabel='Subjects', plot_type='bar', plot_style='classic')
