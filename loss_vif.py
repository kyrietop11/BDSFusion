
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

class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x ):

        b,c,h,w = x.shape

        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr-mg,2)
        Drb = torch.pow(mr-mb,2)
        Dgb = torch.pow(mb-mg,2)
        k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)
        return k

class L_Grad(nn.Module):
    def __init__(self):
        super(L_Grad, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self, image_A, image_B, image_fused):
        gradient_A = self.sobelconv(image_A)
        gradient_B = self.sobelconv(image_B)
        gradient_fused = self.sobelconv(image_fused)
        gradient_joint = torch.max(gradient_A, gradient_B)
        Loss_gradient = F.l1_loss(gradient_fused, gradient_joint)
        return Loss_gradient
        
class L_SSIM(nn.Module):
    def __init__(self):
        super(L_SSIM, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self, image_A, image_B, image_fused):
        gradient_A = self.sobelconv(image_A)
        gradient_B = self.sobelconv(image_B)
        weight_A = torch.mean(gradient_A) / (torch.mean(gradient_A) + torch.mean(gradient_B))
        weight_B = torch.mean(gradient_B) / (torch.mean(gradient_A) + torch.mean(gradient_B))
        Loss_SSIM = weight_A * ssim(image_A, image_fused) + weight_B * ssim(image_B, image_fused)
        return Loss_SSIM
    
    
class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)

class L_Intensity(nn.Module):
    def __init__(self):
        super(L_Intensity, self).__init__()

    def forward(self, image_A, image_B, image_fused):       
        intensity_joint = torch.max(image_A, image_B)    
        Loss_intensity = F.l1_loss(image_fused, intensity_joint)
        return Loss_intensity

# 背景干扰残余度损失函数
# 输入图像为已经过归一化处理的图像 在处理时需反归一化
class L_Background_Residue(nn.Module):
    def __init__(self):
        super(L_Background_Residue, self).__init__()
    def forward(self, image_A, image_B, image_fused): 

        Background_Residue_joint = torch.zeros(ADMD(image_fused).shape)
        Loss_Background_Residue = F.l1_loss(ADMD(image_fused), Background_Residue_joint, reduction='mean')
        # print(image_fused)
        return Loss_Background_Residue
    

# mask损失函数    A-L B-M
# 输入图像为已经过归一化处理的图像 在处理时需反归一化
class L_MaskFusionL1loss(nn.Module):
    def __init__(self):
        super(L_MaskFusionL1loss, self).__init__()

    def forward(self,image_A, image_B, image_fused, mask):

        img_l = image_A[:, :1, :, :]
        img_m = image_B[:, :1, :, :]
        save_image(img_m, 'img_m.jpg')
        # 中波 mask
        mask_vi = mask
        mask_vi[mask_vi != 0] = 1 
        target_m = mask_vi * img_m
        # if torch.sum(target_m)!=0 :
        target_m = equalhist(target_m)
            
        # 长波 逆mask
        mask_ir = torch.ones_like(mask_vi)
        mask_ir[mask_vi == 1] = 0
        save_image(mask_ir,'mask_ir.jpg')
        background_l = mask_ir * img_l
        background_l = filter_background(background_l)

        #期望融合图像
        expected_img =  target_m + background_l

        # 损失函数
        Loss_maskfusion = F.l1_loss(image_fused, expected_img , reduction='mean') 
        save_image(expected_img, 'expected_img.jpg')

        return Loss_maskfusion



class fusion_loss_vif(nn.Module):
    def __init__(self):
        super(fusion_loss_vif, self).__init__()
        self.L_Grad = L_Grad()
        self.L_Inten = L_Intensity()
        self.L_SSIM = L_SSIM()
        self.L_Background_Residue = L_Background_Residue()
        self.L_MaskFusionL1loss = L_MaskFusionL1loss()



    def forward(self, image_A, image_B, image_fused, image_M):
        loss_l1 = 10 * self.L_Inten(image_A, image_B, image_fused)         #10 -> 20
        # loss_gradient = 10 * self.L_Grad(image_A, image_B, image_fused)    #20 -> 10
        # loss_SSIM = 5 * (1 - self.L_SSIM(image_A, image_B, image_fused))   #10 -> 5

        loss_BR = 500 * self.L_Background_Residue(image_A, image_B, image_fused) # 500  ->200
        loss_MF = 20 * self.L_MaskFusionL1loss(image_A, image_B, image_fused, image_M)  # 20 -> 10

        fusion_loss =  loss_l1  + loss_BR + loss_MF
        return fusion_loss, loss_l1, loss_BR, loss_MF


