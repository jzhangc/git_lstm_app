"""
Plot functions: all functions wil be updated with ggplot2-like matplotlib
syntax in the future.
"""

# ------ libraries ------
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import auc, roc_curve  # calculate ROC-AUC
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import History  # for input argument type check

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


def epochs_plot(filepath, model_history, plot_mode='loss', plot_title=None,
                xlabel=None, ylabel=None,
                figure_size=(5, 5)):
    """
    # Purpose
        The plot function for epoch history from LSTM modelling

    # Arguments
        filepath: string. Directory and file name to save the figure. Use os.path.join() to generate
        model_history: keras.callbacks.History. Keras modelling history object, generated by model.fit process.
        plot_mode: string. Either 'loss' or 'acc'. Only classification models can use 'acc'.
        plot_title: string. Plot title.
        xlabel: string. X-axis title.
        ylabel: string. Y-axis title.
        figure_size: float in two-tuple/list. Figure size.

    # Return
        The function returns a pdf figure file to the set directory and with the set file name

    # Details
        NOTE: When plot_mode='loss', the function calculates and plot RMSE, instead of MSE (default in Keras loss calculation)

    """
    # input data check
    if not isinstance(model_history, History):
        raise TypeError("model_history needs to be a keras History object.")

    # prepare data
    if plot_mode == 'loss':  # RMSE
        plot_y = np.sqrt(np.array(model_history.history['loss']))
    else:
        plot_y = np.array(model_history['acc'])
    plot_x = np.arange(1, len(plot_y) + 1)

    # plotting
    fig, ax = plt.subplots(figsize=figure_size)
    fig.set_facecolor('white')
    ax.set_facecolor('white')
    ax.plot(plot_x, plot_y, linestyle='-', color='black')
    ax.set_title(plot_title, color='black')
    ax.set_xlabel(xlabel, fontsize=10, color='black')
    ax.set_ylabel(ylabel, fontsize=10, color='black')
    ax.tick_params(labelsize=5, color='black', labelcolor='black')
    plt.setp(ax.spines.values(), color='black')
    plt.savefig(filepath, dpi=600, bbox_inches='tight', facecolor='white')
    fig
    return fig, ax


def y_yhat_plot(filepath, y_true,
                training_yhat, training_yhat_err,
                test_yhat, test_yhat_err,
                plot_title=None,
                xlabel=None, xlabel_lim=(0, 33),
                ylabel=None, plot_type='bar',
                plot_style='dark_background', bar_width=0.25,
                figure_size=(9, 3)):
    """
    # Purpose:
        This function plots original outcome (y), training predicted y and test predicted y.
        Mainly useful for regression study.

    # Arguments:
        filepath: string. The export directiory.
        y_true: np.ndarray. Input y true values, including both training and test data.
        training_yhat: np.ndarray. Input yhat data for training set y.
        training_yhat_err: np.ndarray. Error values for training_yhat.
        test_yhat: np.ndarray. Iput yhat data for test set y.
        test_yhat_err: np.ndarray. Error values for test_yhat.
        plot_title: string. Title displayed on top of the figure.
        xlabel: string. Label for x-axis.
        xlabel_lim: two tuple or list. Limits for x-axis.
        ylabel: string. Label for y-axis.
        plot_type: string. Plot stype, either 'bar' or 'scatter'
        figure_size: two tuple or list. Figure size.

    # Return:
        The matplotlib subplot objects fig and ax, as well as a figure file saved to the set diretiory.

    # Details:
        The plotting library is matplotlib.

        The function fills NaNs to teh training and test yhat arrays to match the length of y
        for positioning

    """
    # check arguments
    input_yhats = [training_yhat, test_yhat]
    input_errs = [training_yhat_err, test_yhat_err]
    if not all(isinstance(input, np.ndarray) for input in input_yhats + input_errs):
        raise TypeError('All input needs to be np.ndarray.')

    for input_yhat, input_err in zip(input_yhats, input_errs):
        """
        zip function pairs elements from iterators.
        """
        if input_yhat.shape != input_err.shape:
            raise TypeError(
                'Input yhat array should have the same shape as the input error array.')

    # set up data
    y = y_true
    x = np.arange(1, len(y)+1)

    training_yhat_plot, training_yhat_err_plot = np.empty_like(
        y), np.empty_like(y)
    training_yhat_plot[:, ], training_yhat_err_plot[:, ] = np.nan, np.nan
    training_yhat_plot[0:training_yhat.shape[0],
                       ], training_yhat_err_plot[0:training_yhat_err.shape[0], ] = training_yhat, training_yhat_err

    test_yhat_plot, test_yhat_err_plot = np.empty_like(y), np.empty_like(y)
    test_yhat_plot[:, ], test_yhat_err_plot[:, ] = np.nan, np.nan
    test_yhat_plot[training_yhat.shape[0]:,
                   ], test_yhat_err_plot[training_yhat_err.shape[0]:, ] = test_yhat, test_yhat_err

    # plot
    fig, ax = plt.subplots(figsize=figure_size)
    ax.set_xlim((0, 33))
    fig.set_facecolor('white')
    ax.set_facecolor('white')
    if plot_type == 'bar':
        # distance
        r1 = np.arange(1, len(y)+1) - bar_width/2
        r2 = np.arange(1, len(y)+1) + bar_width/2
        r3 = r2

        # plotting
        ax.bar(r1, y, width=bar_width, color='red', label='original')
        ax.bar(r2, training_yhat_plot, yerr=training_yhat_err_plot,
               width=bar_width, color='gray', label='training', ecolor='black', capsize=0)
        ax.bar(r3, test_yhat_plot, yerr=test_yhat_err_plot,
               width=bar_width, color='blue', label='test', ecolor='black', capsize=0)
        ax.axhline(color='black')
    else:  # scatter plot
        ax.scatter(x, y, color='red', label='original')
        ax.fill_between(x, training_yhat_plot-training_yhat_err_plot,
                        training_yhat_plot+training_yhat_err_plot, color='gray', alpha=0.2,
                        label='training')
        ax.fill_between(x, test_yhat_plot-test_yhat_err_plot,
                        test_yhat_plot+test_yhat_err_plot, color='blue', alpha=0.2,
                        label='test')
    ax.set_title(plot_title, color='black')
    ax.set_xlabel(xlabel, fontsize=10, color='black')
    ax.set_ylabel(ylabel, fontsize=10, color='black')
    ax.tick_params(labelsize=5, color='black', labelcolor='black')
    plt.setp(ax.spines.values(), color='black')
    leg = ax.legend(loc='upper center', bbox_to_anchor=(
        0.5, -0.2), ncol=3, fontsize=8, facecolor='white')
    for text in leg.get_texts():
        text.set_color('black')
        text.set_weight('bold')
        text.set_alpha(0.5)
    plt.savefig(filepath, dpi=600, bbox_inches='tight', facecolor='white')
    fig
    return fig, ax
