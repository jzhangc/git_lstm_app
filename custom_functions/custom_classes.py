"""
classess (data and exceptions) for the lstm app
"""

import numpy as np
# ------ libraries ------
import pandas as pd

from custom_functions.cv_functions import (idx_func, longitudinal_cv_xy_array,
                                           lstm_cv_train)
from custom_functions.data_processing import (inverse_norm_y,
                                              training_test_spliter)


# ------ classes ------
class PdDataFrameTypeError(TypeError):
    pass


class NpArrayShapeError(ValueError):
    pass


class lstm_cv(object):
    """
    # Purpose
        The instance of this class contains all the information and results for RNN LSTM modelling
    """

    def __init__(self, data, *args, **kwargs):
        """
        # Behaviour
            To initialize data for CV processes by setting up training and test datasets, 
            as wel as the scalers (as private attributes) for min/max scaling.

        # Arugments
            (*arg, **kwargs: arguments for training_test_spliter function.)
            training_percent: float. percentage of the full data to be the training
            random_state: int. seed for resampling RNG
            min_max_scaling: boolean. if to do a Min_Max scaling to the data
            scale_column_as_y: list. column(s) to use as outcome for scaling
            scale_column_to_exclude: list. the name of the columns 
                                    to remove from the X columns for scaling. 
                                    makes sure to also inlcude the y column(s)

        # Prduct
             self.training: np.ndarray. 
             self.test: np.ndarray.
             self.__scaleX: scaler. Private. 
             self.__scaleY: scaler. Private.

        # Details
            Scalers scaleX and scaleY will return None when min_max_scaling is set to False.

            The reason to return scalers is to use them for inversing predicted 
            but normalized values back to the orignal form - something especially useful for
            regression study.        

        """
        # -- data check: done by training_test_spliter function --
        # -- define values --
        # below:
        self.training, self.test, self.__scaleX, self.__scaleY = training_test_spliter(
            data=data, *args, **kwargs)

    def process(self, *arg, **kwargs):
        """
        # Behaviour
            This method is to process the input data into separate X and Y arrays for 
            training and test data. The method also returns

        # arguments
            (*arg, **kwargs: arguments for longitudinal_cv_xy_array function.)
            input: input 2D pandas DataFrame
            Y_colnames: column names for Y array. has to be a list
            remove_colnames: column to remove to generate X array. has to be a list
            n_features: the number of features used for each timepoint

        # Product
            (All np.ndarray and private)
            self.__trainingX
            self.__trainingY
            self.__testX
            self.__testY
        """
        self.__trainingX, self.__trainingY = longitudinal_cv_xy_array(
            input=self.training, *arg, **kwargs)
        self.__testX, self.__testY = longitudinal_cv_xy_array(
            input=self.test, *arg, **kwargs)

    def cv(self, n_features, Y_colnames, remove_colnames, n_folds=10, random_state=None, *arg, **kwargs):
        """
        # Behaviour
            This method is the core CV method.

        # Arguments
            (Below: arguments for idx_func function)
            input: input 2D pandas DataFrame
            n_folds: fold number for data spliting
            random_state: random state passed to k-fold spliting
            Y_colnames: column names for Y array. has to be a list
            remove_colnames: column to remove to generate X array. has to be a list

            (*arg, **kwargs: arguments for lstm_cv_train function.)
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

        # Product
            (all np.ndarray)
            self.cv_m_ensemble: LSTM CV model ensemble.
            self.__cv_m_history_ensemble: private. loss history for plotting.
            self.cv_m_test_rmse_ensemble: private. RMSE on CV hold-off fold for each CV fold.
        """
        # calculate CV spliting indices
        self.__cv_training_idx, self.__cv_test_idx = idx_func(
            input=self.training, n_features=n_features, Y_colnames=Y_colnames, remove_colnames=remove_colnames,
            n_folds=n_folds, random_state=random_state)

        # CV
        self.cv_m_ensemble, self.__cv_m_history_ensemble, self.cv_m_test_rmse_ensemble = list(), list(), list()
        for i in range(n_folds):
            fold_id = str(i+1)
            print('fold: ', fold_id)
            cv_train_X, cv_train_Y = self.__trainingX[self.__cv_train_idx[i]
                                                      ], self.__trainingY[self.__cv_train_idx[i]]
            cv_test_X, cv_test_Y = self.__trainingX[self.__cv_test_idx[i]
                                                    ], self.__trainingY[self.__cv_test_idx[i]]
            cv_m, cv_m_history, cv_m_test_rmse = lstm_cv_train(trainX=cv_train_X, trainY=cv_train_Y,
                                                               testX=cv_test_X, testY=cv_test_Y,
                                                               lstm_model='simple',
                                                               hidden_units=6, epochs=400, batch_size=29,
                                                               plot=False, filepath=os.path.join(res_dir, 'cv_simple_loss_fold_'+fold_id+'.pdf'),
                                                               plot_title='Simple LSTM model',
                                                               ylabel='MSE',
                                                               verbose=False)
            self.cv_m_ensemble.append(cv_m)
            self.__cv_m_history_ensemble.append(cv_m_history)
            self.cv_m_test_rmse_ensemble.append(cv_m_test_rmse)

    def rmse(self, testX):
        # TBC
        return None

    def predict(self, testX):
        # TBC
        return None
