#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 17:09:34 2020
Authors : Aayush Kumar Chaudhary,Prashnna K Gyawali,Linwei Wang,Jeff B Pelz
Paper : Semi-Supervised Learning for Eye Image Segmentation

This is extract from
A. K. Chaudhary et al., "RITnet: Real-time Semantic Segmentation of the Eye for Gaze Tracking," 2019 IEEE/CVF International Conference on Computer Vision Workshop (ICCVW), 2019, pp. 3698-3702, doi: 10.1109/ICCVW.2019.00568.
"""


from models import model_dict
from torch.utils.data import DataLoader 
from dataset import IrisDataset_ss
import torch
from utils import mIoU, CrossEntropyLoss2d,total_metric,get_nparams,Logger,GeneralizedDiceLoss,SurfaceLoss
import numpy as np
from dataset import transform
from opt import parse_args
import os
from utils import get_predictions
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
args = parse_args()
kwargs = vars(args)


def unlabelled_inverse(label_convert,T_inv_array,device,channel_dim=4):
    label_convert=label_convert.cpu()
    label_final=label_convert.clone()
    for bs in range (label_convert.shape[0]):
        if len(np.shape(T_inv_array[bs]))==2:
            if channel_dim==4:
                for channel in range(channel_dim):
                    img = transforms.ToPILImage()(label_convert[bs,channel])
                    img_transformed = img.transform((np.array(img).shape[1], np.array(img).shape[0]), Image.AFFINE, data=np.float64(T_inv_array[bs]).flatten()[:6], resample=Image.NEAREST)
                    label_final[bs,channel]=transforms.ToTensor()(img_transformed)
                label_final[bs,0]=1-label_final[bs,1]-label_final[bs,2]-label_final[bs,3]
            if channel_dim==1:
                for channel in range(channel_dim):
                    img = transforms.ToPILImage()((label_convert[bs,channel]+1)/2)
                    img_transformed = img.transform((np.array(img).shape[1], np.array(img).shape[0]), Image.AFFINE, data=np.float64(T_inv_array[bs]).flatten()[:6], resample=Image.NEAREST)
                    label_final[bs,channel]=transforms.ToTensor()(img_transformed)
    if channel_dim==1:
        label_final=(label_final-0.5)/0.5
    return label_final.to(device)

def train_setup(labelledloader,unlabelledloader, model, criterion, criterion_DICE,criterion_SL, optimizer, scheduler, device, iterations,alpha,beta, logger, T = 0.5, mode='labeled'):
    print (device)
    model.train()
    labeled_train_iter = iter(labelledloader)
    unlabeled_train_iter = iter(unlabelledloader)
    ious=[]
    
    for i in range(len(unlabeled_train_iter)):
        iterations+=1
        iter_value=iterations//len(unlabeled_train_iter)
        
        try:
            img,labels,index,spatialWeights,maxDist,_,_ = labeled_train_iter.next()
        except:
            labeled_train_iter = iter(labelledloader)
            img,labels,index,spatialWeights,maxDist,_,_ = labeled_train_iter.next()

        if mode != 'labeled': # only for ssl training
            try:
                img_ul1,img_ul2,index_ul,img_ul1_ssl,img_ul2_ssl,ang1,ang2 = unlabeled_train_iter.next()
            except:
                unlabeled_train_iter = iter(unlabelledloader)
                img_ul1,img_ul2,index_ul,img_ul1_ssl,img_ul2_ssl,ang1,ang2 = unlabeled_train_iter.next()
            
            with torch.no_grad():
                data_ul1 =img_ul1.to(device)
                data_ul2 =img_ul2.to(device)
                
                label_ul2=model(data_ul2)
                label_ul1=model(data_ul1) #Bx4xHxW:softmax
          
                label_ul1_softmax=torch.softmax(label_ul1, dim=1)
                label_ul2_softmax=torch.softmax(label_ul2, dim=1)
                
                if mode == 'ssl_augu':
                    data_ul1_ssl =img_ul1_ssl.to(device)
                    data_ul2_ssl =img_ul2_ssl.to(device)
                
                    label_ul2_ssl=model(data_ul2_ssl)
                    label_ul1_ssl=model(data_ul1_ssl) #Bx4xHxW:softmax
                
                    label_ul1_softmax_ssl=torch.softmax(label_ul1_ssl, dim=1)
                    label_ul2_softmax_ssl=torch.softmax(label_ul2_ssl, dim=1)
                    label_ul1_softmax_ssl=unlabelled_inverse(label_ul1_softmax_ssl,ang1,device)
                    label_ul2_softmax_ssl=unlabelled_inverse(label_ul2_softmax_ssl,ang2,device)

                    p = (label_ul1_softmax_ssl + label_ul2_softmax_ssl) / 2 
                    pt = p**(1/T) 
                    label_ul_ssl = pt / pt.sum(dim=1, keepdim=True)
                    label_ul_ssl = label_ul_ssl.detach()
                
                # addition for smoothing the labels instead of hard assignment via get_prediction function
                p = (label_ul1_softmax + label_ul2_softmax) / 2 # softmax over 4 class dimension
                pt = p**(1/T) 
                label_ul = pt / pt.sum(dim=1, keepdim=True)
                label_ul = label_ul.detach()

            if mode == 'ssl_augu':
                data_ul1_ssl=unlabelled_inverse(data_ul1_ssl,ang1,device,channel_dim=1)
                data_ul2_ssl=unlabelled_inverse(data_ul2_ssl,ang2,device,channel_dim=1)

        data = img.to(device)
        target = labels.to(device).long()
        optimizer.zero_grad()            
        
        if mode != 'labeled': # only for ssl training
            output1 = model(data_ul1)
            output2 = model(data_ul2)
            CE_loss2 = (torch.mean((torch.softmax(output1, dim=1) - label_ul.to(device))**2) + torch.mean((torch.softmax(output2, dim=1) - label_ul.to(device))**2)) / 2
        
        if mode == 'ssl_augu':
            output1_ssl = model(data_ul1_ssl)
            output2_ssl = model(data_ul2_ssl)
            CE_loss3 = (torch.mean((torch.softmax(output1_ssl, dim=1) - label_ul_ssl.to(device))**2) + torch.mean((torch.softmax(output2_ssl, dim=1) - label_ul_ssl.to(device))**2)) / 2
   
        output = model(data)
        CE_loss = criterion(output,target)

        ## loss from cross entropy is weighted sum of pixel wise loss and Canny edge loss *20
        loss = CE_loss*(torch.from_numpy(np.ones(spatialWeights.shape)).to(torch.float32).to(device)+(spatialWeights).to(torch.float32).to(device))
        
        loss= torch.mean(loss).to(torch.float32).to(device)
        loss_dice = criterion_DICE(output,target).to(device)
        loss_sl = torch.mean(criterion_SL(output.to(device),(maxDist).to(device))).to(device)

        ##total loss is the weighted sum of suface loss and dice loss plus the boundary weighted cross entropy loss
        if mode != 'labeled':
            if mode =='ssl_augu':
                loss = (1-alpha[iter_value])*loss_sl+alpha[iter_value]*(loss_dice)+loss+args.SSLvalue*(beta[iter_value])*(CE_loss2.to(device)+CE_loss3.to(device)/10)          
            else:  
                loss = (1-alpha[iter_value])*loss_sl+alpha[iter_value]*(loss_dice)+loss+args.SSLvalue*(beta[iter_value])*CE_loss2.to(device) 
        else:
            loss = (1-alpha[iter_value])*loss_sl+alpha[iter_value]*(loss_dice)+loss
        
        loss= loss.to(device)

        loss.backward()
        optimizer.step()
        
        predict = get_predictions(output)
        iou = mIoU(predict,labels)
        ious.append(iou)

        if i%10 == 0:
            if mode != 'labeled':
                logger.write('Epoch:{} [{}/{}], Loss: {:.3f}, SSL: {:.6f}'.format(iterations,i,len(labelledloader),loss.item(),CE_loss2.item()))
            else:
                logger.write('Epoch:{} [{}/{}], Loss: {:.3f}'.format(iterations,i,len(labelledloader),loss.item()))
    return ious,iterations
  
  
def valid_setup(loader,model,factor,device, criterion, criterion_DICE, criterion_SL):
    epoch_loss = []
    ious = []    
    model.eval()
    with torch.no_grad():
        for i, batchdata in enumerate(loader):
            img,labels,index,spatialWeights,maxDist,_,_=batchdata
            data = img.to(device)
            output = model(data)
            target = labels.to(device).long()  
            CE_loss = criterion(output,target)            

            loss = CE_loss*(torch.from_numpy(np.ones(spatialWeights.shape)).to(torch.float32).to(device)+(spatialWeights).to(torch.float32).to(device))
        
            loss= torch.mean(loss).to(torch.float32).to(device)
            loss_dice = criterion_DICE(output,target).to(device)
            loss_sl = torch.mean(criterion_SL(output.to(device),(maxDist).to(device))).to(device)

            loss = (1-factor)*loss_sl+factor*(loss_dice)+loss             
            epoch_loss.append(loss.item())
            predict = get_predictions(output)
            iou = mIoU(predict,labels)
            ious.append(iou)
    return np.average(epoch_loss),np.average(ious)


