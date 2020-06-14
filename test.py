from math import floor
import torch
import time

from metrics.metrics import mean_absolute_error_metric, dice_coefficient_metric, rmse_metric
from metrics.image_utilities import denormalize, image_save

images_mean = {"bg_image": [0.68968, 0.65092, 0.60790], "bg_fg_image" : [0.68084, 0.64135, 0.59833], "mask_image":[0.06140], "depth_image":[0.49981]}
images_std =  {"bg_image": [0.18897, 0.20892, 0.23450], "bg_fg_image" : [0.19432, 0.21262, 0.23641], "mask_image":[0.23733], "depth_image":[0.27300]}

def test(model, criterion_l1, criterion_ssim, criterion_diceBCE, device, test_loader, epoch, test_metrics):

    ### Metrics
    epoch_test_loss = 0
    epoch_mask_mean_abs_error = 0
    epoch_mask_dice_coeff = 0
    epoch_mask_rms_error = 0
    epoch_depth_mean_abs_error = 0
    epoch_depth_dice_coeff = 0
    epoch_depth_rms_error = 0
    ###

    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader, 1):
            data["bg_image"] = data["bg_image"].to(device)
            data["bg_fg_image"] = data["bg_fg_image"].to(device)
            data["mask_image"] = data["mask_image"].to(device)
            data["depth_image"] = data["depth_image"].to(device)

            output_mask, output_depth = model(data["bg_image"], data["bg_fg_image"])

            # Mask loss. Only BCE Dice loss
            mask_loss = criterion_diceBCE(output_mask, data["mask_image"])
            # Depth loss. L1 loss + SSIM + BCE Dice loss
            depth_loss = criterion_l1(output_depth, data["depth_image"]) + criterion_ssim(output_depth, data["depth_image"]) + criterion_diceBCE(output_depth, data["depth_image"])
            loss = mask_loss + depth_loss

            mask_yhat = torch.sigmoid((output_mask.detach()))
            depth_yhat = torch.sigmoid((output_depth.detach()))

            # Calculate metrics
            mask_mean_abs_error, mask_dice_coeff, mask_rms_error = mean_absolute_error_metric(mask_yhat, data["mask_image"]), dice_coefficient_metric(mask_yhat, data["mask_image"]),rmse_metric(mask_yhat, data["mask_image"])
            depth_mean_abs_error, depth_dice_coeff, depth_rms_error = mean_absolute_error_metric(depth_yhat, data["depth_image"]),dice_coefficient_metric(depth_yhat, data["depth_image"]), rmse_metric(depth_yhat, data["depth_image"])
            epoch_test_loss += loss.item()

            #Mask epoch metrics
            epoch_mask_mean_abs_error += mask_mean_abs_error.item()
            epoch_mask_dice_coeff += mask_dice_coeff.item()
            epoch_mask_rms_error += mask_rms_error.item()
            #Depth epoch metrics
            epoch_depth_mean_abs_error += depth_mean_abs_error.item()
            epoch_depth_dice_coeff += depth_dice_coeff.item()
            epoch_depth_rms_error += depth_rms_error.item()

            if batch_idx in [floor(len(test_loader)*0.5), len(test_loader)]:
                print("Mask stats:\nMean absolute error={:.2f}%\tDice Coefficient={:.2f}\tRMSE error={:.2f}%".format(mask_mean_abs_error.item()*100.0, mask_dice_coeff, mask_rms_error.item()*100.0))
                print("Depth stats:\nMean absolute error={:.2f}%\tDice Coefficient={:.2f}\tRMSE error={:.2f}%".format(depth_mean_abs_error.item()*100.0, depth_dice_coeff, depth_rms_error.item()*100.0))

                bg_fg_i = denormalize(data["bg_fg_image"][:16].detach().cpu(), images_mean["bg_fg_image"], images_std["bg_fg_image"])

                image_save(bg_fg_i, filename='test/bgfg_E'+str(epoch)+"_B"+str(batch_idx)+"_")
                image_save(data["mask_image"][:16].detach().cpu(), filename='test/mask_E'+str(epoch)+"_B"+str(batch_idx)+"_")
                image_save(output_mask[:16].detach().cpu(), filename='test/P_mask_E'+str(epoch)+"_B"+str(batch_idx)+"_")
                image_save(data["depth_image"][:16].detach().cpu(), filename='test/depth_E'+str(epoch)+"_B"+str(batch_idx)+"_")
                image_save(output_depth[:16].detach().cpu(), filename='test/P_depth_E'+str(epoch)+"_B"+str(batch_idx)+"_")

    # Test metrics with metrics
    epoch_test_loss /= len(test_loader)
    epoch_mask_mean_abs_error /= len(test_loader)
    epoch_mask_dice_coeff /= len(test_loader)
    epoch_mask_rms_error /= len(test_loader)
    epoch_depth_mean_abs_error /= len(test_loader)
    epoch_depth_dice_coeff /= len(test_loader)
    epoch_depth_rms_error /= len(test_loader)

    test_metrics["epoch_test_loss"].append(epoch_test_loss)
    test_metrics["epoch_mask_mean_abs_error"].append(epoch_mask_mean_abs_error)
    test_metrics["epoch_mask_dice_coeff"].append(epoch_mask_dice_coeff)
    test_metrics["epoch_mask_rms_error"].append(epoch_mask_rms_error)
    test_metrics["epoch_depth_mean_abs_error"].append(epoch_depth_mean_abs_error)
    test_metrics["epoch_depth_dice_coeff"].append(epoch_depth_dice_coeff)
    test_metrics["epoch_depth_rms_error"].append(epoch_depth_rms_error)

            
