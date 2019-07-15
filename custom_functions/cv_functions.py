"""
cross validation functions
"""

# ------ libraries ------
import math

import numpy as np
import pandas as pd
# StratifiedKFold should be used for classification problems
# StratifiedKFold makes sure the fold has an equal representation of the classes
from sklearn.model_selection import KFold
from keras.utils import to_categorical

from custom_functions.lstm_functions import (bidirectional_lstm_m,
                                             simple_lstm_m, stacked_lstm_m)
from custom_functions.plot_functions import epochs_loss_plot


# ------ functions ------
class PdDataFrameTypeError(TypeError):
    pass


class NpArrayShapeError(TypeError):
    pass


def lstm_train_eval(trainX, trainY, testX, testY,
                    lstm_model='simple', study_type='n_to_one', outcome_type='regression',
                    hidden_units=50,
                    epochs=200, batch_size=16,
                    loss='mean_squared_error',
                    optimizer='adam',
                    plot=False, verbose=False, **kwargs):
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
        loss: name of the loss function.
        optimizer: name of the optimization function.
        plot: boolean. if to plot the loss per epochs.
        verbose: verbose setting.
        **kwargs: keyword arguments passed to the plot function epochs_loss_plot().

    # Return:
        A compiled LSTM model object, its modelling history, as well as the evaluation results

    # Details:
        This function trains and evaluates single LSTM model. Thus, the function is used as an
        intermediate functioin for evaluting ensemble models from k-fold CV.
        NOTE: the function might not work for classification: TO BE TESTED
    """
    # argument check
    if not all(isinstance(input, np.ndarray) for input in [trainX, trainY, testX, testY]):
        raise TypeError('All input needs to be in np.ndarray.')
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
    # this might not work for classification studies: TO BE TESTED
    if study_type == 'n_to_one':
        if lstm_model == 'simple':
            m = simple_lstm_m(n_steps=n_timepoints, n_features=n_features,
                              hidden_units=hidden_units, n_output=1,
                              loss=loss, optimizer=optimizer)
        elif lstm_model == 'stacked':
            m = stacked_lstm_m(n_steps=n_timepoints, n_features=n_features,
                               hidden_units=hidden_units, loss=loss, optimizer=optimizer)
        elif lstm_model == 'bidirectional':
            m = bidirectional_lstm_m(n_steps=n_timepoints, n_features=n_features, hidden_units=hidden_units,
                                     loss=loss, optimizer=optimizer)

    m_history = m.fit(x=trainX, y=trainY, epochs=epochs,
                      batch_size=batch_size, verbose=verbose)

    # loss plot
    if plot:
        epochs_loss_plot(model_history=m_history, **kwargs)

    # evaluating
    # NOTE: below: regressional study only outputs loss (usually MSE), no accuracy
    # NOTE: for classification, the evaluate function returns both loss and accurarcy
    if outcome_type == 'regression':
        # only returns mse for regression
        eval_res = m.evaluate(testX, testY, verbose=verbose)
        eval_res = math.sqrt(eval_res)  # rmse
    else:
        # below returns loss and acc. we only capture acc
        _, eval_res = m.evaluate(testX, testY, verbose=verbose)

    # return
    return m, m_history, eval_res


def idx_func(input, n_features, Y_colnames, remove_colnames, n_folds=10, random_state=None):
    """
    # Purpose:
        This function returns numpy array for input X and Y and KFold splitting indices

    # Arguments:
        input: input 2D pandas DataFrame
        n_folds: fold number for data spliting
        random_state: random state passed to k-fold spliting
        Y_colnames: column names for Y array. has to be a list
        remove_colnames: column to remove to generate X array. has to be a list

    # Return 
        (In this order):
        X array, Y array, kfold training indices, kfold test indices

    # Details:
        This is a temp function for testing cross validation
    """

    # set up the x y array data
    X, Y = longitudinal_cv_xy_array(input=input, Y_colnames=Y_colnames,
                                    remove_colnames=remove_colnames, n_features=n_features)

    # setup KFold
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    # extract the indices
    train_indices, test_indices = list(), list()
    for train_index, test_index in kfold.split(input):
        train_indices.append(train_index)
        test_indices.append(test_index)

    # return data
    return X, Y, train_indices, test_indices


def longitudinal_cv_xy_array(input, Y_colnames, remove_colnames, n_features):
    """
    # Purpose:
        This is a intermediate function that converts an input pandas DataFrame into X and Y numpy arrays

    # Arguments:
        input: input 2D pandas DataFrame
        Y_colnames: column names for Y array. has to be a list
        remove_colnames: column to remove to generate X array. has to be a list
        n_features: the number of features used for each timepoint

    # Details:
        For now, the function only supports longitudinal data type, i.e. with multiple timepoint.
        Therfore, the outpout X array will follow a 3D matrix shape for keras LSTM function requirement: 
            n_samples x n_timepoints x n_features

        The input should be a pandas DataFrame with first couple of columns as annotation/outcome
        , with the rest columns as the X data.

        Since input is longitudinal, the structure should be stacked as following: 
        timepoint1_features, timepoint2_features,..., timepointN_features

        NOTE: all time points show have ths same number of features

    # Return:
        X and Y as numpy arrays
    """
    # input type check
    if not isinstance(input, pd.DataFrame):
        raise PdDataFrameTypeError("Inoput needs to be a pandas DataFrame.")
    if not isinstance(Y_colnames, list) or not isinstance(remove_colnames, list):
        raise TypeError("Y_colnames and remove_colnames need to be list type.")

    # split input to X and Y DataFrames
    rm_from_X = Y_colnames + remove_colnames
    # NOTE: below: axis=1 means to look for columns
    X = input.drop(labels=rm_from_X, axis=1)
    Y = input.loc[:, Y_colnames]

    # convert to numpy arrays
    X = np.array(X)
    Y = np.array(Y)

    # split X according to timpoints and n_features
    total_feature_indices = np.arange(X.shape[1])
    # NOTE: below: np.split produces a list, instead of an np.array
    n_bins = len(total_feature_indices)/n_features
    split_feature_indices = np.split(total_feature_indices, n_bins)
    initial_X = list()  # the initial_X shape will be n_timpoints x n_samples x n_features
    for timepoint_feature_index in split_feature_indices:
        initial_X.append(X[:, timepoint_feature_index])
    initial_X = np.array(initial_X)

    # re-arrage out_X to match the shape requirement: n_samples x n_timepoints x n_features
    out_X = list()
    for n_sample in range(initial_X.shape[1]):
        tmp_list = list()
        for n_timepoint in range(initial_X.shape[0]):
            arr = initial_X[n_timepoint, n_sample, :]
            tmp_list.append(arr)
        out_X.append(tmp_list)

    # return
    out_X = np.array(out_X)
    out_Y = Y
    return out_X, out_Y
