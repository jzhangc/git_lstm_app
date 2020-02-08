"""
to test small things
"""
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from custom_functions.custom_exceptions import (NpArrayShapeError,
                                                PdDataFrameTypeError)
from custom_functions.lstm_functions import (bidirectional_lstm_m,
                                             simple_lstm_m, stacked_lstm_m)
from custom_functions.plot_functions import epochs_plot

import math
import os
import datetime

import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import History  # for input argument type check
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
# StratifiedKFold should be used for classification problems
# StratifiedKFold makes sure the fold has an equal representation of the classes
from sklearn.model_selection import KFold
# from sklearn.preprocessing import MinMaxScaler, StandardScaler

from custom_functions.cv_functions import (idx_func, longitudinal_cv_xy_array,
                                           lstm_cv_train, lstm_ensemble_eval,
                                           lstm_ensemble_predict)
from custom_functions.data_processing import training_test_spliter_final
from custom_functions.plot_functions import y_yhat_plot
from custom_functions.util_functions import logging_func

# ------ test functions ------
# NOTE: this should be a custom class

# ------ data processing ------
# ---- working directory
main_dir = os.path.abspath('./')
dat_dir = os.path.join(main_dir, 'data')
res_dir = os.path.join(main_dir, 'results/new_lstm_phase1_base_results')
log_dir = os.path.join(main_dir, 'log')

# ---- setup logger
logger = logging_func(filepath=os.path.join(log_dir, 'test.log'))

# ---- import data
raw = pd.read_csv(os.path.join(
    dat_dir, 'new_lstm_aec_phase1_base_freq1.csv'), engine='python')
raw.iloc[0:5, 0:5]

# ---- key variable
n_features = 7
n_folds = 10
selected_features = [["PP12", "PN05", "PN11"],
                     ["PP19", "PP18", "PN23"],
                     ["PP15", "PN16", "PN18"],
                     ["PP15", "PP24", "PN11"],
                     ["PN17", "PN19", "PP13"],
                     ["PP05", "PN03", "PP20"],
                     ["PP20", "PP21", "PN11"]]  # order: freq 1~7


# ---- generate training and test sets with min-max normalization
# new spliting
training, test, training_scaler_X, training_scaler_Y, test_scaler_Y = training_test_spliter_final(
    data=raw, random_state=1,
    man_split=True, man_split_colname='subject', man_split_testset_value=selected_features[0],
    x_standardization=True,
    x_scale_column_to_exclude=['subject', 'PCL', 'group'],
    y_min_max_scaling=True,
    y_column_to_scale=['PCL'], y_scale_range=(0, 1))


# procesing
trainingX, trainingY = longitudinal_cv_xy_array(input=training, Y_colnames=['PCL'],
                                                remove_colnames=['subject', 'group'], n_features=n_features)
testX, testY = longitudinal_cv_xy_array(input=test, Y_colnames=['PCL'],
                                        remove_colnames=['subject', 'group'], n_features=n_features)


training_scaler_Y.inverse_transform(testY)

# below: as an example for converting the normalized Y back to measured values
# _, testY = inverse_norm_y(training_y=trainingY, test_y=testY, scaler=scaler_Y)
# ---- test k-fold data sampling
cv_train_idx, cv_test_idx = idx_func(input=training, n_features=n_features, Y_colnames=['PCL'],
                                     remove_colnames=['subject', 'group'], n_folds=n_folds, random_state=5)

trainingX.shape  # n_samples x n_timepoints x n_features
trainingY.shape  # 29x1
len(cv_train_idx[0])  # 26

# ------ cross-validation modelling ------
cv_m_ensemble, cv_m_history_ensemble, cv_m_test_rmse_ensemble, cv_m_test_rsq_ensemble = list(
), list(), list(), list()
for i in range(n_folds):
    fold_id = str(i+1)
    print('fold: ', fold_id)
    cv_train_X, cv_train_Y = trainingX[cv_train_idx[i]
                                       ], trainingY[cv_train_idx[i]]
    cv_test_X, cv_test_Y = trainingX[cv_test_idx[i]], trainingY[cv_test_idx[i]]
    cv_m, cv_m_history, cv_m_test_rmse, cv_m_test_rsq = lstm_cv_train(trainX=cv_train_X, trainY=cv_train_Y,
                                                                      testX=cv_test_X, testY=cv_test_Y,
                                                                      lstm_model='simple',
                                                                      hidden_units=25, epochs=150, batch_size=29,
                                                                      output_activation='sigmoid',
                                                                      log_dir=os.path.join(
                                                                          res_dir, 'fit', 'cv_fold_'+fold_id),
                                                                      plot=True,
                                                                      filepath=os.path.join(
                                                                          res_dir, 'cv_simple_loss_fold_'+fold_id+'.pdf'),
                                                                      plot_title=None,
                                                                      xlabel=None,
                                                                      ylabel=None,
                                                                      verbose=False)
    cv_m_ensemble.append(cv_m)
    cv_m_history_ensemble.append(cv_m_history)
    cv_m_test_rmse_ensemble.append(cv_m_test_rmse)
    cv_m_test_rsq_ensemble.append(cv_m_test_rsq)

cv_rmse_mean = np.mean(cv_m_test_rmse_ensemble)
cv_rmse_sem = np.std(cv_m_test_rmse_ensemble)/math.sqrt(n_folds)
cv_m_test_rmse_ensemble
cv_rmse_mean
cv_rmse_sem

cv_rsq_mean = np.mean(cv_m_test_rsq_ensemble)
cv_rsq_sem = np.std(cv_m_test_rsq_ensemble)/math.sqrt(n_folds)
cv_m_test_rsq_ensemble
cv_rsq_mean
cv_rsq_sem

# ------ prediction ------
# also features: inverse the predictions from normalized values to PCLs
# prediction for the test subjests
# yhats_testX = lstm_ensemble_predict(testX=testX,
#                                     models=cv_m_ensemble, model_index=[2, 5, 6, 8])
yhats_testX = lstm_ensemble_predict(testX=testX, models=cv_m_ensemble)
# below: anix=0 means "along the row, by column"
yhats_testX_mean = np.mean(yhats_testX, axis=0)
# NOTE: below: real world utility only uses training scalers
# yhats_testX_pred = training_scaler_Y.inverse_transform(
#     yhats_testX_mean)   # converted back
# yhats_testX_conv = [training_scaler_Y.inverse_transform(
#     ary) for ary in yhats_testX]
yhats_testX_pred = test_scaler_Y.inverse_transform(
    yhats_testX_mean)   # converted back

yhats_testX_conv = [test_scaler_Y.inverse_transform(
    ary) for ary in yhats_testX]
yhats_testX_conv = np.array(yhats_testX_conv)
yhats_testX_std = np.std(yhats_testX_conv, axis=0)
yhats_testX_sem = yhats_testX_std/math.sqrt(n_folds)

# prediction for 80% test subjests
# yhats_trainingX = lstm_ensemble_predict(
#     models=cv_m_ensemble, testX=trainingX, model_index=[2, 5, 6, 8])
yhats_trainingX = lstm_ensemble_predict(
    models=cv_m_ensemble, testX=trainingX)
yhats_trainingX_mean = np.mean(yhats_trainingX, axis=0)
# yhats_trainingX_pred = training_scaler_Y.inverse_transform(
#     yhats_trainingX_mean)   # converted back
# yhats_trainingX_conv = [training_scaler_Y.inverse_transform(
#     ary) for ary in yhats_trainingX]
yhats_trainingX_pred = training_scaler_Y.inverse_transform(
    yhats_trainingX_mean)   # converted back
yhats_trainingX_conv = [training_scaler_Y.inverse_transform(
    ary) for ary in yhats_trainingX]
yhats_trainingX_conv = np.array(yhats_trainingX_conv)
yhats_trainingX_std = np.std(yhats_trainingX_conv, axis=0)
yhats_trainingX_sem = yhats_trainingX_std/math.sqrt(n_folds)

testY_conv = test_scaler_Y.inverse_transform(testY)

# ------ eval ------
# below: calcuate RMSE and R2 (final, i.e. on the test data) using inversed y and yhat
# this RMSE is the score to report
rmse_yhats, rsq_yhats = list(), list()
for yhat_testX in yhats_testX_conv:  # NOTE: below: real world utility only uses training scalers
    rmse_yhat = math.sqrt(mean_squared_error(
        y_true=testY_conv, y_pred=yhat_testX))
    rsq_yhat = r2_score(y_true=testY_conv, y_pred=yhat_testX)
    rmse_yhats.append(rmse_yhat)
    rsq_yhats.append(rsq_yhat)


rmse_yhats = np.array(rmse_yhats)
rmse_yhats_mean = np.mean(rmse_yhats)
rmse_yhats_sem = np.std(rmse_yhats)/math.sqrt(n_folds)
rmse_yhats
rmse_yhats_mean
rmse_yhats_sem

rsq_yhats = np.array(rsq_yhats)
rsq_yhats_mean = np.mean(rsq_yhats)
rsq_yhats_sem = np.std(rsq_yhats)/math.sqrt(n_folds)
rsq_yhats
rsq_yhats_mean
rsq_yhats_sem

# ------ plot testing ------
y = np.concatenate([training_scaler_Y.inverse_transform(
    trainingY), test_scaler_Y.inverse_transform(testY)])
y_true = y.reshape(y.shape[0], )
yhats_trainingX_pred = yhats_trainingX_pred.reshape(
    yhats_trainingX_pred.shape[0], )
yhats_trainingX_std = yhats_trainingX_std.reshape(
    yhats_trainingX_std.shape[0], )
yhats_trainingX_sem = yhats_trainingX_sem.reshape(
    yhats_trainingX_sem.shape[0], )
yhats_testX_pred = yhats_testX_pred.reshape(yhats_testX_pred.shape[0], )
yhats_testX_std = yhats_testX_std.reshape(yhats_testX_std.shape[0], )
yhats_testX_sem = yhats_testX_sem.reshape(yhats_testX_sem.shape[0], )

y_yhat_plot(filepath=os.path.join(res_dir, 'new_freq1_cv_plot_scatter_test.pdf'),
            y_true=y_true,
            training_yhat=yhats_trainingX_pred,
            training_yhat_err=yhats_trainingX_std,
            test_yhat=yhats_testX_pred,
            test_yhat_err=yhats_testX_std,
            plot_title='Cross-validation prediction',
            ylabel='PCL', xlabel='Subjects', plot_type='scatter',
            bar_width=0.25)


# ------ true test realm ------
i = 0
fold_id = str(i+1)
print('fold: ', fold_id)
cv_train_X, cv_train_Y = trainingX[cv_train_idx[i]
                                   ], trainingY[cv_train_idx[i]]
cv_test_X, cv_test_Y = trainingX[cv_test_idx[i]], trainingY[cv_test_idx[i]]

trainX = cv_train_X
trainY = cv_train_Y
testX = cv_test_X
testY = cv_test_Y
lstm_model = 'simple'
hidden_units = 25
epochs = 500
batch_size = 29
output_activation = 'sigmoid'
log_dir = os.path.join(res_dir, 'fit', 'cv_fold_'+fold_id)
plot = True
filepath = os.path.join(res_dir, 'cv_simple_loss_fold_'+fold_id+'.pdf')
plot_title = None
xlabel = None
ylabel = None
verbose = False

loss = 'mean_squared_error'
optimizer = 'adam'
study_type = 'n_to_one'


def test_foo(trainX, trainY, testX, testY,
             lstm_model='simple', study_type='n_to_one', outcome_type='regression',
             hidden_units=50,
             epochs=200, batch_size=16,
             output_activation='linear',
             loss='mean_squared_error',
             optimizer='adam',
             plot=False, log_dir=None,
             verbose=False, **kwargs):
    """
    # Purpose:
        This is the wraper function for LSTM model training and evaluating.

    # Argument:
        trainX: numpy ndarray for training X. shape requirment: n_samples x n_timepoints x n_features.
        trainY: numpy ndarray for training Y. shape requirement: n_samples.
        testX: numpy ndarray for test X. shape requirment: n_samples x n_timepoints x n_features.
        testY: numpy ndarray for test Y. shape requirement: n_samples.
        lstm_model: string. the type of LSTM model to use.
        study_type: string. the type of study, 'n_to_one' or 'n_to_n'. More to be added.
        outcome_type: string. the type of the outcome, 'regression' or 'classification'.
        hidden_units: int. number of hidden units in the first layer.
        epochs: int. number of epochs to use for LSTM modelling.
        batch_size: int. batch size for each modelling iteration.
        output_activation: string. activation for the output layer, 
                            following the keras Dense() activation argument.
        loss: name of the loss function.
        optimizer: name of the optimization function.
        plot: boolean. if to plot the loss per epochs.
        log_dir: string. The output directory for the tensorboard results.
        verbose: verbose setting.
        **kwargs: keyword arguments passed to the plot function epochs_loss_plot().

    # Return:
        A compiled LSTM model object, its modelling history, as well as the evaluation RMSE and R2 (on the hold-off fold) results

    # Details:
        This function trains and evaluates single LSTM model. Thus, the function is used as an
            intermediate functioin for evaluting ensemble models from k-fold CV.
        The output_action function "sigmoid" can be used for the min-max scaled data to (0, 1)
        NOTE: the function might not work for classification: TO BE TESTED
    """
    # argument check
    if not all(isinstance(input, np.ndarray) for input in [trainX, trainY, testX, testY]):
        raise TypeError('All input needs to be np.ndarray.')
    if not all(len(x_input.shape) == 3 for x_input in [trainX, testX]):
        raise NpArrayShapeError('trainX and testX need to be in 3D shape.')
    if not (trainX.shape[1] == testX.shape[1] and trainX.shape[2] == testX.shape[2]):
        raise NpArrayShapeError(
            'trainX and testX should have the same second and third dimensions.')

    # arguments
    # training_n_samples, test_n_samples = trainX.shape[0], testX.shape[0]
    n_timepoints, n_features = trainX.shape[1], trainX.shape[2]

    # y data processing
    # this might not work: TO BE TESTED
    if outcome_type == 'classification':
        trainY = to_categorical(trainY)

    # modelling
    # callback
    earlystop_callback = EarlyStopping(monitor='val_loss', patience=5)
    if log_dir:
        tfbaord_callback = TensorBoard(log_dir=log_dir)
        callbacks = [tfbaord_callback, earlystop_callback]
    else:
        callbacks = earlystop_callback

    # this might not work for classification studies: TO BE TESTED
    if study_type == 'n_to_one':
        if lstm_model == 'simple':
            m = simple_lstm_m(n_steps=n_timepoints, n_features=n_features,
                              hidden_units=hidden_units, n_output=1,
                              output_activation=output_activation,
                              loss=loss, optimizer=optimizer)
        elif lstm_model == 'stacked':
            m = stacked_lstm_m(n_steps=n_timepoints, n_features=n_features,
                               hidden_units=hidden_units,
                               output_activation=output_activation,
                               loss=loss, optimizer=optimizer)
        elif lstm_model == 'bidirectional':
            m = bidirectional_lstm_m(n_steps=n_timepoints, n_features=n_features, hidden_units=hidden_units,
                                     output_activation=output_activation,
                                     loss=loss, optimizer=optimizer)
    # training the model
    m_history = m.fit(x=trainX, y=trainY, epochs=epochs,
                      batch_size=batch_size,
                      callbacks=[callbacks],
                      validation_data=(testX, testY),
                      verbose=True)

    # loss plot
    if plot:
        epochs_plot(model_history=m_history, **kwargs)

    # evaluating
    # NOTE: below: regressional study only outputs loss (usually MSE), no accuracy
    # NOTE: for classification, the evaluate function returns both loss and accurarcy
    if outcome_type == 'regression':
        # only returns mse for regression
        eval_res = m.evaluate(testX, testY, verbose=verbose)
        eval_res = math.sqrt(eval_res)  # rmse
        yhat_res = m.predict(testX)
        rsq = r2_score(testY, yhat_res)
    else:
        # below returns loss and acc. we only capture acc
        _, eval_res = m.evaluate(testX, testY, verbose=verbose)
        rsq = None

    # return
    return m, m_history, eval_res, rsq
