"""
Plot functions: all functions wil be updated with ggplot2-like matplotlib
syntax in the future.
"""
# ------ libraries ------
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc  # calculate ROC-AUC
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt  # to plot ROC-AUC


# ------ functions ------
def auc_plot(model, newdata_X, newdata_Y):  # AUC plot
    test_Y_hat = model.predict(newdata_X).ravel()  # ravel() flattens the array
    fpr, tpr, threshold = roc_curve(newdata_Y, test_Y_hat)
    AUC = auc(fpr, tpr)

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='AUC = {:.3f}'.format(AUC))
    plt.xlabel('1-Specificity')
    plt.ylabel('Sensitivity')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()
    return None


# epoch accuracy plot
def epochs_acc_plot(filepath, model_history, plot_title=None, ylabel='accuracy'):
    plt.figure(1)
    plt.plot(model_history.history['acc'])
    try:
        plt.plot(model_history.history['val_acc'])
        plt.legend(['train', 'test'], loc='best')
    except KeyError:
        plt.legend('train', loc='best')
    plt.title(plot_title)
    plt.xlabel('epoch')
    plt.ylabel(ylabel)
    plt.show()
    return None


def epochs_loss_plot(filepath, model_history, plot_title=None, ylabel='loss'):   # epoch loss plot
    plt.plot(model_history.history['loss'])
    try:
        plt.plot(model_history.history['val_loss'])
        plt.legend(['train', 'test'], loc='best')
    except KeyError:
        plt.legend('train', loc='best')
    plt.title(plot_title)
    plt.xlabel('epoch')
    plt.ylabel(ylabel)
    plt.savefig(filepath, dpi=600, bbox_inches='tight')
    plt.show()
    return None


def y_yhat_plot(filepath, y_true, training_yhat, test_yhat, plot_title=None, xlabel=None, ylabel=None, plot_type='bar',
                plot_style='dark_background', bar_width=0.25):
    """
    # Purpose:
        This function plots original outcome (y), training predicted y and test predicted y.
        Mainly useful for regression study.

    # Arguments:
        filepath: string. The export directiory. 
        y_true: np.ndarray. Input y true values, including both training and test data.
        training_yhat: np.ndarray. Input yhat data for training set y.
        test_yhat: np.ndarray. Iput yhat data for test set y.
        plot_title: string. Title displayed on top of the figure.
        xlabel: string. Label for x-axis. 
        ylabel: string. Label for y-axis.
        plot_tyle: string. The figure style setting, 'classic' or 'dark_background'.

    # Return:
        A figure file saved to the set diretiory.

    # Details:
        The plotting library is matplotlib.

        The function fills NaNs to teh training and test yhat arrays to match the length of y
        for positioning

    """
    # check arguments

    # set up data
    y = y_true
    x = np.arange(1, len(y)+1)

    training_y_hat_plot = np.empty_like(y)
    training_y_hat_plot[:, ] = np.nan
    training_y_hat_plot[0:training_yhat.shape[0], ] = training_yhat

    test_y_hat_plot = np.empty_like(y)
    test_y_hat_plot[:, ] = np.nan
    test_y_hat_plot[training_yhat.shape[0]:, ] = test_yhat

    # plot
    if plot_type == 'bar':
        # distance
        r1 = np.arange(1, len(y)+1) - bar_width/2
        r2 = np.arange(1, len(y)+1) + bar_width/2
        r3 = r2

        # plotting
        plt.bar(r1, y, width=bar_width, color='red', label='original')
        plt.bar(r2, training_y_hat_plot, width=bar_width,
                color='gray', label='training')
        plt.bar(r3, test_y_hat_plot, width=bar_width,
                color='blue', label='test')
        plt.legend(loc='upper center',
                   bbox_to_anchor=(0.5, -0.1), ncol=3)
    else:
        plt.plot(x, y)
        plt.plot(x, training_y_hat_plot)
        plt.plot(x, test_y_hat_plot)
        plt.legend(['original', 'training', 'test'], loc='upper center',
                   bbox_to_anchor=(0.5, -0.1), ncol=3)

    plt.title(plot_title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(ticks=np.arange(1, len(y)+1, step=5))
    plt.style.use(plot_style)
    plt.savefig(filepath, dpi=600, bbox_inches='tight')
    plt.show()
    return None
