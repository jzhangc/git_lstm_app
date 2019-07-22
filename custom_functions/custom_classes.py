"""
classess (data and exceptions) for the lstm app
"""

# ------ libraries ------
import pandas as pd
import numpy as np
from custom_functions.data_processing import (inverse_norm_y,
                                              training_test_spliter)
from custom_functions.data_processing import training_test_spliter


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
        # Behaviours
            To initialize data for CV processes by setting up training and test datasets, 
            as wel as the scalers (as private attributes) for min/max scaling.

        # Arugments
        *arg, **kwargs: arguments for training_test_spliter function.
            training_percent: float. percentage of the full data to be the training
            random_state: int. seed for resampling RNG
            min_max_scaling: boolean. if to do a Min_Max scaling to the data
            scale_column_as_y: list. column(s) to use as outcome for scaling
            scale_column_to_exclude: list. the name of the columns 
                                    to remove from the X columns for scaling. 
                                    makes sure to also inlcude the y column(s)

        # Return
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

    def process(self):
        """
        This method is to process the input data into
        """
        return None

    def cv():
        return None

    def rmse(self, testX):
        # TBC
        return None

    def predict(self, testX):
        # TBC
        return None
