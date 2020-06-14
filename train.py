from math import floor
from tqdm.autonotebook import tqdm
import time
import torch
from metrics.metrics import mean_absolute_error_metric, dice_coefficient_metric, rmse_metric
from metrics.image_utilities import denormalize, image_save


images_mean = {"bg_image": [0.68968, 0.65092, 0.60790], "bg_fg_image" : [0.68084, 0.64135, 0.59833], "mask_image":[0.06140], "depth_image":[0.49981]}
images_std =  {"bg_image": [0.18897, 0.20892, 0.23450], "bg_fg_image" : [0.19432, 0.21262, 0.23641], "mask_image":[0.23733], "depth_image":[0.27300]}

def train(model, criterion_l1, criterion_ssim, criterion_diceBCE, device, train_loader, optimizer, epoch, train_metrics, save_model=False):
    ### Metrics
    epoch_train_loss = 0

    epoch_mask_mean_abs_error = 0
    epoch_mask_dice_coeff = 0
    epoch_mask_rms_error = 0

    epoch_depth_mean_abs_error = 0
    epoch_depth_dice_coeff = 0
    epoch_depth_rms_error = 0
    ###

    pbar = tqdm(enumerate(train_loader, 1), total = len(train_loader))

    model.train()

    for batch_idx, data in pbar:
        dataload_time_tic = time.time()
        data["bg_image"] = data["bg_image"].to(device)
        data["bg_fg_image"] = data["bg_fg_image"].to(device)
        data["mask_image"] = data["mask_image"].to(device)
        data["depth_image"] = data["depth_image"].to(device)
        dataload_time_toc = time.time()

        optimizer.zero_grad()
        # Model prediction
        predict_time_tic = time.time()
        output_mask, output_depth = model(data["bg_image"], data["bg_fg_image"])
        predict_time_toc = time.time()

        # Loss calculation
        loss_calc_time_tic = time.time()
        # Mask Loss
        mask_loss = criterion_diceBCE(output_mask, data["mask_image"])
        # Depth Loss
        depth_loss = criterion_l1(output_depth, data["depth_image"]) + criterion_ssim(output_depth, data["depth_image"]) + criterion_diceBCE(output_depth, data["depth_image"])
        loss = mask_loss + depth_loss
        loss_calc_time_toc = time.time()

        #Backprop
        backprop_time_tic = time.time()
        loss.backward()
        backprop_time_toc = time.time()

        #Parameter Update
        parameter_update_time_tic = time.time()
        optimizer.step()
        parameter_update_time_toc = time.time()

        ## Stats collation
        metric_time_tic = time.time()
        #For metric calculation, remove copy of output from computation graph.
        mask_yhat = output_mask.detach()
        depth_yhat = output_depth.detach()
        # Ground truth is between [0,1]. Apply sigmoid to get prediction in the same range.
        mask_yhat = torch.sigmoid(mask_yhat)
        depth_yhat = torch.sigmoid(depth_yhat)

        # Mask Metrics - Calculate Absolute mean error, dice coefficient, mask_rms_error
        # Absolute mean error.
        mask_mean_abs_error, mask_dice_coeff, mask_rms_error = mean_absolute_error_metric(mask_yhat, data["mask_image"]), dice_coefficient_metric(mask_yhat, data["mask_image"]), rmse_metric(mask_yhat, data["mask_image"])

        # Depth Metrics - Calculate Absolute mean error, dice coefficient, mask_rms_error
        # Absolute mean error.
        depth_mean_abs_error, depth_dice_coeff, depth_rms_error = mean_absolute_error_metric(depth_yhat, data["depth_image"]), dice_coefficient_metric(depth_yhat, data["depth_image"]), rmse_metric(depth_yhat, data["depth_image"])

        epoch_train_loss += loss.item()
        #Mask epoch metrics
        epoch_mask_mean_abs_error += mask_mean_abs_error.item()
        epoch_mask_dice_coeff += mask_dice_coeff.item()
        epoch_mask_rms_error += mask_rms_error.item()
        #Depth epoch metrics
        epoch_depth_mean_abs_error += depth_mean_abs_error.item()
        epoch_depth_dice_coeff += depth_dice_coeff.item()
        epoch_depth_rms_error += depth_rms_error.item()

        metric_time_toc = time.time()

        # Show stats 4 times in an epoch
        if batch_idx in [floor(len(train_loader)*0.5), len(train_loader)]:
            print("Mask stats:\nMean absolute error={:.2f}%\tDice Coefficient={:.2f}\tRMSE error={:.2f}%".format(mask_mean_abs_error.item()*100.0, mask_dice_coeff, mask_rms_error.item()*100.0))
            print("Depth stats:\nMean absolute error={:.2f}%\tDice Coefficient={:.2f}\tRMSE error={:.2f}%".format(depth_mean_abs_error.item()*100.0, depth_dice_coeff, depth_rms_error.item()*100.0))
            print("Processing time\nDataLoad={:.2f}s, Prediction={:.2f}s, Loss calculate={:.2f}s, Backprop={:.2f}s, Parameter Update={:.2f}s, Metrics={:.2f}s".
                  format(
                        (dataload_time_toc-dataload_time_tic),
                        (predict_time_toc-predict_time_tic),
                        (loss_calc_time_toc-loss_calc_time_tic),
                        (backprop_time_toc-backprop_time_tic),
                        (parameter_update_time_toc-parameter_update_time_tic),
                        (metric_time_toc-metric_time_tic)))

            bg_fg_i = denormalize(data["bg_fg_image"][:16].detach().cpu(), images_mean["bg_fg_image"], images_std["bg_fg_image"])

            image_save(bg_fg_i, filename='train/bgfg_E'+str(epoch)+"_B"+str(batch_idx)+"_")
            image_save(data["mask_image"][:16].detach().cpu(), filename='train/mask_E'+str(epoch)+"_B"+str(batch_idx)+"_")
            image_save(output_mask[:16].detach().cpu(), filename='train/P_mask_E'+str(epoch)+"_B"+str(batch_idx)+"_")
            image_save(data["depth_image"][:16].detach().cpu(), filename='train/depth_E'+str(epoch)+"_B"+str(batch_idx)+"_")
            image_save(output_depth[:16].detach().cpu(), filename='train/P_depth_E'+str(epoch)+"_B"+str(batch_idx)+"_")


        pbar_details = f'L={loss.item():0.4f} - MDice={mask_dice_coeff:0.2f} - DDice={depth_dice_coeff:0.2f}'
        pbar.set_description(desc= pbar_details)

    # Save the weights
    if save_model:
        state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
                }
        torch.save(state, "model_save/Model-Epoch"+str(epoch)+"_"+time.strftime("%Y%m%d-%H%M%S")+".pt")

    # Test metrics with metrics
    epoch_train_loss /= len(train_loader)
    epoch_mask_mean_abs_error /= len(train_loader)
    epoch_mask_dice_coeff /= len(train_loader)
    epoch_mask_rms_error /= len(train_loader)
    epoch_depth_mean_abs_error /= len(train_loader)
    epoch_depth_dice_coeff /= len(train_loader)
    epoch_depth_rms_error /= len(train_loader)

    train_metrics["epoch_train_loss"].append(epoch_train_loss)
    train_metrics["epoch_mask_mean_abs_error"].append(epoch_mask_mean_abs_error)
    train_metrics["epoch_mask_dice_coeff"].append(epoch_mask_dice_coeff)
    train_metrics["epoch_mask_rms_error"].append(epoch_mask_rms_error)
    train_metrics["epoch_depth_mean_abs_error"].append(epoch_depth_mean_abs_error)
    train_metrics["epoch_depth_dice_coeff"].append(epoch_depth_dice_coeff)
    train_metrics["epoch_depth_rms_error"].append(epoch_depth_rms_error)
