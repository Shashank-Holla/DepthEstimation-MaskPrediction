# Monocular Depth Estimation and Mask Prediction

Segmentation and Depth estimate are few key tasks in Computer Vision and are some of the fields where Deep Neural network really shines through. 
Image segmentation is to locate objects and their boundaries in an image. This mainly simplifies the representation of the image into something that is easier and meaningful to analyse, which is particularly useful in medical imaging and recognition tasks. Image segmentation involves creating pixel wise mask for the object in the image. 

Monocular Depth estimate is to gather perception of depth from a single static image. It provides information of how far things are relative to point of view.

The objective of this model is to predict the monocular depth map and also to create masks of the object of interest.


Model building is broken down into the following parts. We will work on the parts and then build the sum.


## Pre model training

- [X] - [Dataset](https://github.com/Shashank-Holla/DepthEstimation-MaskPrediction/blob/master/README.md#dataset)

- [X] - [Data ingestion- Dataset and Dataloader](https://github.com/Shashank-Holla/DepthEstimation-MaskPrediction/blob/master/README.md#data-ingestion--dataset-and-dataloader)

- [] - [Image augmentation](https://github.com/Shashank-Holla/DepthEstimation-MaskPrediction/blob/master/README.md#image-augmentation)


## Model Training

- [X] - [Model Architecture](https://github.com/Shashank-Holla/DepthEstimation-MaskPrediction/blob/master/README.md#model-architecture)

- [X] - [Choose the Loss function](https://github.com/Shashank-Holla/DepthEstimation-MaskPrediction/blob/master/README.md#choose-the-loss-function)

- [X] - [Model parameters and hyperparameters](https://github.com/Shashank-Holla/DepthEstimation-MaskPrediction/blob/master/README.md#model-parameters-and-hyper-parameters)

- [X] - [Optimization](https://github.com/Shashank-Holla/DepthEstimation-MaskPrediction/blob/master/README.md#optimization)



## Output of the model

- [X] - [Evaluating the output](https://github.com/Shashank-Holla/DepthEstimation-MaskPrediction/blob/master/README.md#evaluating-the-output)


## Results and Observations

- [X] - [Run results](https://github.com/Shashank-Holla/DepthEstimation-MaskPrediction/blob/master/README.md#results)

- [X] - [Observations](https://github.com/Shashank-Holla/DepthEstimation-MaskPrediction/blob/master/README.md#observations)


## Pre model training

### Dataset

Overlayed, mask and depth images considered this model are of Home interiors and cats as background and foreground respectively.

Dataset details can be found here - [Session14](https://github.com/Shashank-Holla/TSAI-EVA4/tree/master/Session14_RCNN%26DenseDepth)

### Data ingestion- Dataset and Dataloader

Custom dataset is built to read and provide a dictionary containing quartet of images- background image, background-foreground image, mask and depth image. The quartet of images are ensured to have the same context, that is, the same background and same location of the foreground in the images.

Find the code [here](https://github.com/Shashank-Holla/DepthEstimation-MaskPrediction/blob/master/data/schrodingersCatDataset.py)

### Image Augmentation

Image normalization and image resize have been applied. 
Earlier intention was to apply padding with border reflect and random crop to provide further augmented data and to apply RGB shift pixelwise transforms. Since random crop and probability based transforms are applied an image at a time, the context that is present in the quartet of images is lost. Therefore these have not been applied. Find the code [here](https://github.com/Shashank-Holla/DepthEstimation-MaskPrediction/blob/master/data/albumentations_transform.py)

As of now, probability based transform on pair of images without losing context is possible. It is shown here. Further understanding is required to apply transform in the same order and probability to the 4 images. 

*This is not applied to the current runs*

<TABLE>
  <TR>
    <TH>Augmented bg_fg image</TH>
    <TH>Augmented mask image</TH>
  </TR>
   <TR>
      <TD><img src="https://github.com/Shashank-Holla/DepthEstimation-MaskPrediction/blob/master/Images/imageAugmentation_bgfg.jpg" alt="GT_mask"
	title="augbg_fg" width="300" height="300" /></TD>
      <TD><img src="https://github.com/Shashank-Holla/DepthEstimation-MaskPrediction/blob/master/Images/imageAugmentation_mask.jpg" alt="Pred-Mask"
	title="Prediction_Mask" width="300" height="300" /></TD>
   </TR>
</TABLE>



## Model Training

### Model Architecture

To predict the depth and mask maps, the output of the model needs to be dense. That is, the prediction's resolution needs to be equal to that of the input. For this purpose, the model is designed with encoder-decoder architecture. Features of the input (edges, gradients, parts of the object) are extracted in the encoder segment. The decoder segment has two heads- one each to predict mask and depth maps.

#### Encoder
The encoder segment of the model consists of 4 encoder blocks. Each of the encoder blocks uses 3x3 regular convolution and 3x3 dilated convolution. Dilated convolution is used to capture the spatial information of the pixels. The result of these convolutions are concatenated. Further, the channel size is halved using 1x1 pointwise convolution.

<p>
<img src="https://github.com/Shashank-Holla/DepthEstimation-MaskPrediction/blob/master/Images/Encoder.jpg" alt="Pred-Mask"
	title="Prediction_Mask" width="400" height="300" /></p>

#### Decoder
The decoder segment consists of 4 decoder blocks. The resolution of the feature maps are upscaled by a factor of 2 in each of the decoder blocks using pointwise convolution and pixel-shuffle. Pointwise convolution is here used to double the number of channels. Pixel-shuffle is later used to upscale the resolution.

<p><img src="https://github.com/Shashank-Holla/DepthEstimation-MaskPrediction/blob/master/Images/decoder.jpg" width="400" height="200" /></p>



The number of parameters used by the model is- 5,230,720. Forward/Backward pass size of the model is less than 500 MB making this a light model. Summary of the model is present in the ipynb file.

**Full Architecture**
<p>
<img src="https://github.com/Shashank-Holla/DepthEstimation-MaskPrediction/blob/master/Images/Full_architecture.jpg" width="800" height="400" /></p>


Find the code [here](https://github.com/Shashank-Holla/DepthEstimation-MaskPrediction/blob/master/models/depthmasknet.py)


### Choose the Loss function

For mask segmentation prediction, (BCE + Dice) loss function is considered.

**BCE + Dice Loss**

Binary cross entropy loss provides large penalty when incorrect predictions are made with high probability. This is combined with Dice Loss (1-Dice Coefficient). Dice coefficient provides measure of overlap between the ground truth and the predicted output.

Code for the BCE+Dice loss can be found [here](https://github.com/Shashank-Holla/DepthEstimation-MaskPrediction/blob/master/loss.py) 


For depth map prediction, using just BCE+Dice loss gives result where the edges and contours of the background is lost. One result is shared below.

![](https://github.com/Shashank-Holla/DepthEstimation-MaskPrediction/blob/master/Images/Depth_1_onePixelShuffle.jpg)

Hence, for depth predictions, SSIM and L1 loss is also considered.

**SSIM Loss** - Structural Similarity Index provides the perceptual difference between the ground truth depth map and the prediction. While, **L1 Loss** provides the absolute difference (numerical distance) between the ground truth and predicted image.


### Model Parameters and Hyper parameters

Optimizer : SGD

Loss function: (BCE + Dice loss) for mask estimation and (BCE + Dice) + SSIM + L1 loss for depth estimation

Batch size : 512 for 64x64 resolution, 128 for 112x112 resolution and 32 for 192x192 resolution.

Epochs : 3 (64x64 res) + 3 (112x112 res) + 2 (192x192 res)

L2 Regularization : 1e-5

Scheduler : Step LR


### Optimization

1. Pin Memory - Pin memory flag is set to True to speed up dataset transfer from CPU to GPU.

2. Since, the input size of the model during train/test do not vary, the following flags are set to true. This flag enables the inbuilt CUDNN auto-tuner to find the optimal algorithm for the received GPU hardware. This configuration was tested for single train run of 280K images. Improvement of 7 min was observed on this single run (23 min without, 17 with flag enabled).

`torch.backends.cudnn.benchmark = True`


`torch.backends.cudnn.enabled = True`



3. Though not advised, metric calculations for the output and ground truth tensors was done on the GPU itself. This is to avoid GPU to CPU transfer for every batch.


## Output of the model

### Evaluating the output

To evaluate the model, the following metrics are considered. Sigmoid of the prediction is taken before the metrics calculation as the ground truth of mask and depth map is in the range [0, 1].

**Dice Coefficient** - It measures to check the similarity by calculating the overlap between the two samples. Coefficient value ranges between 0 (no overlap) to 1 (highly similar).

**Mean Absolute Error** - It calculates the average of pixelwise numerical distance between the ground truth and prediction. Higher the metric, higher the error. 

**Root Mean Square Error** - It calculates the pixelwise root mean square between thr ground truth and prediction.

Refer the code for the metrics [here](https://github.com/Shashank-Holla/DepthEstimation-MaskPrediction/blob/master/metrics/metrics.py#L40)

## Results and Observations

### Results

The model was trained on 3 sets of image resolutions (64x64, 112x112 and original resolution of 192x192) for 3 epochs.  

#### Predictions

Below are the results for the run on 192x192 resolution images. Further results can be found [here](https://drive.google.com/drive/folders/1ACsG-epUmRCJ0zaKIAGGc5DzeG3SSY65)

 <TABLE>
  <TR>
    <TH>Ground Truth - Mask</TH>
    <TH>Prediction - Mask</TH>
  </TR>
   <TR>
      <TD><img src="https://github.com/Shashank-Holla/DepthEstimation-MaskPrediction/blob/master/Images/mask_E1_B8750_20200530-203614.jpg" alt="GT_mask"
	title="Ground Truth-Mask" width="400" height="400" /></TD>
      <TD><img src="https://github.com/Shashank-Holla/DepthEstimation-MaskPrediction/blob/master/Images/P_mask_E1_B8750_20200530-203614.jpg" alt="Pred-Mask"
	title="Prediction_Mask" width="400" height="400" /></TD>
   </TR>
</TABLE>


 <TABLE>
  <TR>
    <TH>Ground Truth - Depth</TH>
    <TH>Prediction - Depth</TH>
  </TR>
   <TR>
      <TD><img src="https://github.com/Shashank-Holla/DepthEstimation-MaskPrediction/blob/master/Images/depth_E1_B8750_20200530-203614.jpg" alt="GT_depth"
	title="GroundTruth_Depth" width="400" height="400" /></TD>
      <TD><img src="https://github.com/Shashank-Holla/DepthEstimation-MaskPrediction/blob/master/Images/P_depth_E1_B8750_20200530-203614.jpg" alt="Pred-Depth"
	title="Prediction_Depth" width="400" height="400" /></TD>
   </TR>
</TABLE>


#### Graphs for Train/Test Dice Coefficient, Mean Absolute Error and RMSE

Below is the trend for the metrics collected during train/test 

![](https://github.com/Shashank-Holla/DepthEstimation-MaskPrediction/blob/master/Images/run_metrics.jpg)


### Observations

1. The model's mask predictions are very close to the ground truth and dice coefficient of 0.98 is achieved.

2. But model's depth map predictions suffers from checkerboard issue. The edges and contours of the ground truth depth maps are no clearly captured. Dice coefficient on the depth map is about 0.59
 
## TODO

* Checkerboard issue for depth predictions with pixel shuffle. Fine tune the model. 

* Learning rate fine tune

* Implement Tensorboard
