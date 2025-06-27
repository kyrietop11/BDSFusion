from matplotlib import image
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.vgg import vgg16
import numpy as np
from utils.utils_color import RGB_HSV, RGB_YCbCr
from models.loss_ssim import ssim
import torchvision.transforms.functional as TF
import cv2
from torchvision.utils import save_image

from models.ADMD import ADMD
from models.Equalhist_target import equalhist, expand_mask
from models.Filter_background import filter_background

class L_Intensity(nn.Module):
    def __init__(self):
        super(L_Intensity, self).__init__()

    def forward(self, image_A, image_B, image_fused):       
        intensity_joint = torch.max(image_A, image_B)    
        Loss_intensity = F.l1_loss(image_fused, intensity_joint)
        return Loss_intensity


# Background interference residue loss function
class L_Background_Residue(nn.Module):
    def __init__(self):
        super(L_Background_Residue, self).__init__()

    def forward(self, image_A, image_B, image_fused):
        Background_Residue_joint = torch.zeros(ADMD(image_fused).shape)
        Loss_Background_Residue = F.l1_loss(ADMD(image_fused), Background_Residue_joint, reduction='mean')
        # print(image_fused)
        return Loss_Background_Residue


# Mask fusion L1 loss function A-L B-M
class L_MaskFusionL1loss(nn.Module):
    def __init__(self):
        super(L_MaskFusionL1loss, self).__init__()

    def forward(self, image_A, image_B, image_fused, mask):
        img_l = image_A[:, :1, :, :]
        img_m = image_B[:, :1, :, :]
        save_image(img_m, 'img_m.jpg')
        # Medium wave mask
        mask_vi = mask
        mask_vi[mask_vi != 0] = 1
        target_m = mask_vi * img_m
        # if torch.sum(target_m)!=0 :
        target_m = equalhist(target_m)

        # Long wave inverse mask
        mask_ir = torch.ones_like(mask_vi)
        mask_ir[mask_vi == 1] = 0
        save_image(mask_ir, 'mask_ir.jpg')
        background_l = mask_ir * img_l
        background_l = filter_background(background_l)

        # Expected fused image
        expected_img = target_m + background_l

        # Loss function
        Loss_maskfusion = F.l1_loss(image_fused, expected_img, reduction='mean')
        save_image(expected_img, 'expected_img.jpg')

        return Loss_maskfusion


class fusion_loss_vif(nn.Module):
    def __init__(self):
        super(fusion_loss_vif, self).__init__()
        self.L_Inten = L_Intensity()
        self.L_Background_Residue = L_Background_Residue()
        self.L_MaskFusionL1loss = L_MaskFusionL1loss()

    def forward(self, image_A, image_B, image_fused, image_M):
        loss_l1 = 10 * self.L_Inten(image_A, image_B, image_fused)  # 10 -> 20
        loss_BR = 500 * self.L_Background_Residue(image_A, image_B, image_fused)  # 500  ->200
        loss_MF = 20 * self.L_MaskFusionL1loss(image_A, image_B, image_fused, image_M)  # 20 -> 10

        fusion_loss = loss_l1 + loss_BR + loss_MF
        return fusion_loss, loss_l1, loss_BR, loss_MF


