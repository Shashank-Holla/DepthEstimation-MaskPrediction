import matplotlib.pyplot as plt
import torch
import numpy as np

def train_test_metrics_graph(train_accuracy, train_loss, test_accuracy, test_loss):

    fig, axs = plt.subplots(1,2,figsize=(15,7.5))
    axs[0].set_title("Accuracy")
    axs[0].plot(train_accuracy, label = "train_accuracy")
    axs[0].plot(test_accuracy, label = "test accuracy")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="best")

    axs[1].set_title("Loss")
    axs[1].plot(train_loss, label = "train loss")
    axs[1].plot(test_loss, label = "test loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].legend(loc="best")
    
def train_test_loss_dice_metrics_graph(train_dice, train_loss, test_dice, test_loss):
  
    fig, axs = plt.subplots(1,2,figsize=(15,7.5))
    axs[0].set_title("Dice Coefficient")
    axs[0].plot(train_dice, label = "train dice")
    axs[0].plot(test_dice, label = "test dice")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Dice Coefficient")
    axs[0].legend(loc="best")

    axs[1].set_title("Loss")
    axs[1].plot(train_loss, label = "train loss")
    axs[1].plot(test_loss, label = "test loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].legend(loc="best")


def dice_coefficient_metric(yhat, y):
    '''
    Measure to check the similarity between the two samples.
    dice_coefficient = 2 * area of overlap / (total number of pixels in both images.)

    Input-
    yhat - Tensor (N x C x H x W) - Model's prediction
    y - Tensor (N x C x H x W) - Ground truth to be compared with.

    Output-
    dice_coefficient - score of the similarity between the samples. range: [0,1]
    '''
    intersection = (yhat * y).sum()
    dice_coefficient = (2.*intersection)/(yhat.sum() + y.sum())

    return dice_coefficient


def mean_absolute_error_metric(yhat, y):
    '''
    Absolute mean error of the two samples. Higher the metric, higher the error.
    Input-
    yhat - Tensor (N x C x H x W) - Model's prediction
    y - Tensor (N x C x H x W) - Ground truth to be compared with.

    Output-
    mae_error - Mean absolute error. 
    '''

    mae_error = (1/torch.numel(yhat)) * torch.abs(y-yhat).sum()
    return mae_error


def rmse_metric(yhat, y):
    '''
    Root mean square error of the two samples. Higher the metric, higher the error.
    Input-
    yhat - Tensor (N x C x H x W) - Model's prediction
    y - Tensor (N x C x H x W) - Ground truth to be compared with.

    Output-
    rms_error - Root mean square error. 
    '''
    rms_error = torch.sqrt((1/torch.numel(yhat)) * torch.square(y-yhat).sum())
    return rms_error


def loss_dice_MEA_RMS_metrics_graph(train_metrics_dict, test_metrics_dict):
    '''
    Chart Loss, Dice Coefficient, Mean Absolute error, Root Mean Square error for mask/depth images for train as well as test.
    Input
    train_metrics_dict - Dictionary with train stats.
    test_metrics_dict - Dictionary with test stats.
    '''
    train_loss = train_metrics_dict["epoch_train_loss"]
    train_dice_mask = train_metrics_dict["epoch_mask_dice_coeff"]
    train_dice_depth = train_metrics_dict["epoch_depth_dice_coeff"]
    train_MAE_mask = train_metrics_dict["epoch_mask_mean_abs_error"]
    train_MAE_depth = train_metrics_dict["epoch_depth_mean_abs_error"]
    train_RMS_mask = train_metrics_dict["epoch_mask_rms_error"]
    train_RMS_depth = train_metrics_dict["epoch_depth_rms_error"]

    test_loss = test_metrics_dict["epoch_test_loss"]
    test_dice_mask = test_metrics_dict["epoch_mask_dice_coeff"]
    test_dice_depth = test_metrics_dict["epoch_depth_dice_coeff"]
    test_MAE_mask = test_metrics_dict["epoch_mask_mean_abs_error"]
    test_MAE_depth = test_metrics_dict["epoch_depth_mean_abs_error"]
    test_RMS_mask = test_metrics_dict["epoch_mask_rms_error"]
    test_RMS_depth = test_metrics_dict["epoch_depth_rms_error"]

  
    fig, axs = plt.subplots(2,2,figsize=(12,12))

    axs[0][0].set_title("Train/Test Loss")
    axs[0][0].plot(train_loss, label = "train loss")
    axs[0][0].plot(test_loss, label = "test loss")
    axs[0][0].set_xlabel("Epoch")
    axs[0][0].set_ylabel("Loss")
    axs[0][0].legend(loc="best")

    axs[0][1].set_title("Train/Test Dice Coefficient")
    axs[0][1].plot(train_dice_mask, label = "train dice mask")
    axs[0][1].plot(train_dice_depth, label = "train dice depth")
    axs[0][1].plot(test_dice_mask, label = "test dice mask")
    axs[0][1].plot(test_dice_depth, label = "test dice depth")
    axs[0][1].set_xlabel("Epoch")
    axs[0][1].set_ylabel("Dice Coefficient")
    axs[0][1].legend(loc="best")

    axs[1][0].set_title("Train/Test Mean Absolute Error (MAE)")
    axs[1][0].plot(train_MAE_mask, label = "train MAE mask")
    axs[1][0].plot(train_MAE_depth, label = "train MAE depth")
    axs[1][0].plot(test_MAE_mask, label = "test MAE mask")
    axs[1][0].plot(test_MAE_depth, label = "test MAE depth")
    axs[1][0].set_xlabel("Epoch")
    axs[1][0].set_ylabel("Mean Absolute Error")
    axs[1][0].legend(loc="best")

    axs[1][1].set_title("Train/Test Root Mean Square Error (RMSE)")
    axs[1][1].plot(train_RMS_mask, label = "train RMSE mask")
    axs[1][1].plot(train_RMS_depth, label = "train RMSE depth")
    axs[1][1].plot(test_RMS_mask, label = "test RMSE mask")
    axs[1][1].plot(test_RMS_depth, label = "test RMSE depth")
    axs[1][1].set_xlabel("Epoch")
    axs[1][1].set_ylabel("Root Mean Square Error")
    axs[1][1].legend(loc="best")
