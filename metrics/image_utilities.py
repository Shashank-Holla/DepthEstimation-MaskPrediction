import torch
import numpy as np
import torchvision
from torchvision import transforms, utils
import os
import matplotlib.pyplot as plt
import time
from torchvision.transforms import ToTensor

#Hardcoded
gdrive_path = '/gdrive/My Drive/ColabNotebooks/SchoolOfAI/Mask_RCNN_depth'

def capture_correct_incorrect_classified_samples(net, device, testloader):
    """
    Captures incorrect sample data- such as labels, predictions and images
    Input
        net - model
        device - device to run the model
        testloader - testloader
    """
    net.eval()
    incorrect_labels = torch.tensor([], dtype = torch.long)
    incorrect_predictions = torch.tensor([], dtype = torch.long)
    incorrect_images = torch.tensor([])

    # correct_labels = torch.tensor([], dtype = torch.long)
    # correct_predictions = torch.tensor([], dtype = torch.long)

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            
            _, predicted = torch.max(outputs.data, 1)
            result = predicted.eq(labels.view_as(predicted))

            # Incorrect labels, images and predictions           
            incorrect_labels = torch.cat((incorrect_labels,labels[~result].cpu()), dim=0)
            incorrect_predictions = torch.cat((incorrect_predictions, predicted[~result].cpu()), dim=0)
            incorrect_images = torch.cat((incorrect_images, images[~result].cpu()), dim=0)
            # correct_labels = torch.cat((correct_labels,labels[result].cpu()), dim=0)
        
        return incorrect_labels.numpy(), incorrect_predictions.numpy(), incorrect_images



def dataset_calculate_mean_std():
        """
        Download train and test dataset, concatenate and calculate mean and standard deviation for this set.
        """
        set1 = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=ToTensor())
        set2 = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=ToTensor())
        data = np.concatenate([set1.data, set2.data], axis=0)
        stddev = list(np.std(data, axis=(0, 1, 2)) / 255)
        means = list(np.mean(data, axis=(0, 1, 2)) / 255)
        return stddev, means



def denormalize(tensor, mean, std):
    if not tensor.ndimension() == 4:
        raise TypeError('tensor should be 4D')

    mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)

    return tensor.mul(std).add(mean)


def image_show(image_tensor, figsize = (10,10), nrow=4, border_colour=255, img_type=None, fig_save=False):
    '''
    Function displays and option to save montage of images.
    Input:
    image_tensor - Tensors of images to be displayed (required format: N x C x H x W)
    figsize - Display size for the montage.
    nrow - Number of rows of the montage.
    border_colour - Colour of the border between images. Default=255
    img_type - Name of the file type. This will be the file name prefix for the saved images.
    fig_show - Boolean flag to show the image.
    fig_save - Boolean flag to save the image in the results directory.

    Output:
    Display image grid.
    If fig_save is enabled, save image in results folder. 
    '''
    grid_image = utils.make_grid(image_tensor, nrow =nrow, pad_value=border_colour)
    plt.figure(figsize=figsize)
    # matplotlib needs image in format- H, W, C (Channel at the end)
    plt.imshow(grid_image.permute(1,2,0))
    plt.axis('off')
    if fig_save:
        plt.savefig(os.path.join(gdrive_path,"run_results/"+str(img_type)+time.strftime("%Y%m%d-%H%M%S")+".jpg"),bbox_inches='tight')
    plt.show()


def image_save(tensor, filename, nrow=4, pad_value=255):
    '''
    tensor - (Denormalized) Tensors of images to be displayed (required format: N x C x H x W)
    filename - Name of the file to be saved as.
    nrow - Number of rows of the montage.
    border_colour - Colour of the border between images. Default=255
    '''
    utils.save_image(tensor, fp=os.path.join(gdrive_path,"run_results/"+str(filename)+time.strftime("%Y%m%d-%H%M%S")+".jpg"),nrow =nrow, pad_value=pad_value)
