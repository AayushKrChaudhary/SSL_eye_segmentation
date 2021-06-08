#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:47:44 2019

@author: Aayush

This file contains the dataloader and the augmentations and preprocessing done

Required Preprocessing for all images (test, train and validation set):
1) Gamma correction by a factor of 0.8
2) local Contrast limited adaptive histogram equalization algorithm with clipLimit=1.5, tileGridSize=(8,8)
3) Normalization
    
Train Image Augmentation Procedure Followed 
1) Random horizontal flip with 50% probability.
2) Starburst pattern augmentation with 20% probability. 
3) Random length lines augmentation around a random center with 20% probability. 
4) Gaussian blur with kernel size (7,7) and random sigma with 20% probability. 
5) Translation of image and labels in any direction with random factor less than 20.


Modified from :
A. K. Chaudhary et al., "RITnet: Real-time Semantic Segmentation of the Eye for Gaze Tracking," 2019 IEEE/CVF International Conference on Computer Vision Workshop (ICCVW), 2019, pp. 3698-3702, doi: 10.1109/ICCVW.2019.00568.

Added code for:
- Multiple dataset (OpenEDS-2019 and OpenEDS-2020).
- Variation in Gamma correction and CLAHE
- Image translation and Rotation for SSL setups
"""

import numpy as np
import torch
from torch.utils.data import Dataset 
import os
from PIL import Image
from torchvision import transforms
import cv2
import random
import os.path as osp
from utils import one_hot2dist
import copy

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])])
  
#%%
class RandomHorizontalFlip(object):
    def __call__(self, img,label):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT),label.transpose(Image.FLIP_LEFT_RIGHT)
        return img,label
    
class Starburst_augment(object):
    ## We have generated the starburst pattern from a train image 000000240768.png
    ## Please follow the file Starburst_generation_from_train_image_000000240768.pdf attached in the folder 
    ## This procedure is used in order to handle people with multiple reflections for glasses
    ## a random translation of mask of starburst pattern
    def __call__(self, img, filepath,factor_decrease=1):
        x=np.random.randint(1, 40/factor_decrease)
        y=np.random.randint(1, 40/factor_decrease)
        mode = np.random.randint(0, 2)
        starburst=Image.open('starburst_black.png').convert("L")
        if filepath=='Semantic_Segmentation_Dataset':
            starburst = starburst.resize((int(240/factor_decrease),int(320/factor_decrease)), Image.ANTIALIAS)
        if filepath=='openEDS2020-SparseSegmentation':
            starburst = starburst.resize((int(320/factor_decrease),int(240/factor_decrease)), Image.ANTIALIAS)
        if mode == 0:
            starburst = np.pad(starburst, pad_width=((0, 0), (x, 0)), mode='constant')
            starburst = starburst[:, :-x]
        if mode == 1:
            starburst = np.pad(starburst, pad_width=((0, 0), (0, x)), mode='constant')
            starburst = starburst[:, x:]

        img=np.array(img)*((255-np.array(starburst))/255)+np.array(starburst)
        return Image.fromarray(img)

def getRandomLine(xc, yc, theta,factor_decrease):
    x1 = xc - 50/factor_decrease*np.random.rand(1)*(1 if np.random.rand(1) < 0.5 else -1)
    y1 = (x1 - xc)*np.tan(theta) + yc
    x2 = xc - (150/factor_decrease*np.random.rand(1) + 50/factor_decrease)*(1 if np.random.rand(1) < 0.5 else -1)
    y2 = (x2 - xc)*np.tan(theta) + yc
    return x1, y1, x2, y2

class Gaussian_blur(object):
    def __call__(self, img):
        sigma_value=np.random.randint(2, 7)
        return Image.fromarray(cv2.GaussianBlur(img,(7,7),sigma_value))

class Translation(object):
    def __call__(self, base,mask,factor_decrease=1):
        factor_h = np.random.randint(1, 40/factor_decrease)
        factor_v = np.random.randint(1, 40/factor_decrease)
        mode = np.random.randint(0, 4)
        if mode == 0:
            aug_base = np.pad(base, pad_width=((factor_v, 0), (0, 0)), mode='constant')
            aug_mask = np.pad(mask, pad_width=((factor_v, 0), (0, 0)), mode='constant')
            aug_base = aug_base[:-factor_v, :]
            aug_mask = aug_mask[:-factor_v, :]
        if mode == 1:
            aug_base = np.pad(base, pad_width=((0, factor_v), (0, 0)), mode='constant')
            aug_mask = np.pad(mask, pad_width=((0, factor_v), (0, 0)), mode='constant')
            aug_base = aug_base[factor_v:, :]
            aug_mask = aug_mask[factor_v:, :]
        if mode == 2:
            aug_base = np.pad(base, pad_width=((0, 0), (factor_h, 0)), mode='constant')
            aug_mask = np.pad(mask, pad_width=((0, 0), (factor_h, 0)), mode='constant')
            aug_base = aug_base[:, :-factor_h]
            aug_mask = aug_mask[:, :-factor_h]
        if mode == 3:
            aug_base = np.pad(base, pad_width=((0, 0), (0, factor_h)), mode='constant')
            aug_mask = np.pad(mask, pad_width=((0, 0), (0, factor_h)), mode='constant')
            aug_base = aug_base[:, factor_h:]
            aug_mask = aug_mask[:, factor_h:]
        return Image.fromarray(aug_base), Image.fromarray(aug_mask),factor_h,factor_v,mode   
            
    #https://stackabuse.com/affine-image-transformations-in-python-with-numpy-pillow-and-opencv/ 
def aug_unlabelled(img):
    T_translate=np.eye(3)
    T_rotate=np.eye(3)
    T_scale=np.eye(3)
    if random.random() < 0.8:
        tx = -20+np.random.randint(1, 40)
        ty = -20+np.random.randint(1, 40)

        T_translate = np.array([
            [1, 0, tx],
            [0, 1, ty],
            [0, 0, 1]])

    if random.random() < 0.5:

        angle = 5*2*(np.random.rand(1) - 0.5)
        theta = np.deg2rad(angle)
        c = np.float(np.cos(theta))
        s = np.float(np.sin(theta))
        T_rotate = np.array([
            [c, s, 0],
            [-s, c, 0],
            [0, 0, 1]])

    T = T_translate @ T_rotate @ T_scale
    T_inv = np.linalg.inv(T)         

    img_transformed = img.transform((np.array(img).shape[1], np.array(img).shape[0]), Image.AFFINE, data=T_inv.flatten()[:6], resample=Image.NEAREST)
    return img_transformed,np.linalg.inv(T_inv)

class Line_augment(object):
    def __call__(self, base,factor_decrease=1):
        yc, xc = (0.3 + 0.4*np.random.rand(1))*base.shape
        aug_base = copy.deepcopy(base)
        num_lines = np.random.randint(1, 10)
        for i in np.arange(0, num_lines):
            theta = np.pi*np.random.rand(1)
            x1, y1, x2, y2 = getRandomLine(xc, yc, theta,factor_decrease)
            aug_base = cv2.line(aug_base, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 4)
        aug_base = aug_base.astype(np.uint8)
        return Image.fromarray(aug_base)       
        
class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()       

def rotate_image(base):
    ang = 5*2*(np.random.rand(1) - 0.5)
    aug_base=base.rotate(ang)
    return aug_base, ang

def rotate_image_inverse(base,ang):
    return (base.rotate(-ang))

class IrisDataset_ss(Dataset):
    def __init__(self, filepath,setup=0, factor_decrease=1,split='labelled',transform=None, mode='', **args):
        self.transform = transform
        self.filepath_original=filepath
        self.filepath= osp.join(filepath,split)
        self.split = split
        self.factor_decrease=factor_decrease
        self.mode = mode
        listall = []
        if self.filepath_original=='Semantic_Segmentation_Dataset':
            if split=='labelled':
                self.setupsplit=''
                
                fileDescriptor = open(self.filepath_original+'/Data/'+setup+'.txt', "r")
                line = True
    
                while line:
                    line = fileDescriptor.readline()
                    if line:
                        imagePath = line.split()[0]
                        listall.append(imagePath.split('/')[-1].strip('.png'))                  
            else:
                self.setupsplit=''       
                for file in os.listdir(osp.join(self.filepath,'images'+self.setupsplit)):   
                    if file.endswith(".png"):
                        listall.append(file.strip(".png"))
            
        if self.filepath_original=='openEDS2020-SparseSegmentation':
            if split=='labelled':
                fileDescriptor = open(self.filepath_original+'/Data/'+setup+'.txt', "r")
            else:
                fileDescriptor = open(self.filepath_original+'/Data/'+split+'.txt', "r")
            line = True
            while line:
                line = fileDescriptor.readline()
                if line:
                    lineItems = line.split()[0]
                    listall.append('.'+lineItems.strip('.png'))                  
        self.list_files=listall
        

        self.testrun = args.get('testrun')
        
        #PREPROCESSING STEP FOR ALL TRAIN, VALIDATION AND TEST INPUTS 
        #local Contrast limited adaptive histogram equalization algorithm
        self.clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))

    def __len__(self):
        if self.testrun:
            return 10
        return len(self.list_files)

    def __getitem__(self, idx):
        if self.filepath_original=='Semantic_Segmentation_Dataset':
              imagepath = osp.join(self.filepath,'images'+self.setupsplit,self.list_files[idx]+'.png')
          
        if self.filepath_original=='openEDS2020-SparseSegmentation':
            imagepath = self.list_files[idx]+'.png'


        pilimg = Image.open(imagepath).convert("L")      
        if self.filepath_original=='Semantic_Segmentation_Dataset':
            pilimg = pilimg.resize((240,320), Image.ANTIALIAS)       

        if self.filepath_original=='openEDS2020-SparseSegmentation':
            pilimg = pilimg.resize((320,240), Image.ANTIALIAS)       
     
        H, W = pilimg.width , pilimg.height
        H=int(H/self.factor_decrease)
        W=int(W/self.factor_decrease) 
        pilimg = pilimg.resize((H,W), Image.ANTIALIAS)       
        #PREPROCESSING STEP FOR ALL TRAIN, VALIDATION AND TEST INPUTS 
    
        gamma = [0.8,0.85,0.9,0.95,1.0,1.05,1.10, 1.15, 1.2][np.random.randint(0,9)]
        table = 255.0*(np.linspace(0, 1, 256)**gamma)
        pilimg = cv2.LUT(np.array(pilimg), table)        
        pilimg=Image.fromarray(pilimg)  
        
        if (self.split!='unlabelled'):
            if self.filepath_original=='Semantic_Segmentation_Dataset':
                labelpath = osp.join(self.filepath,'labels',self.list_files[idx]+'.npy')
            if self.filepath_original=='openEDS2020-SparseSegmentation':
                tempPath=self.list_files[idx]
                labelpath =  ('/').join(tempPath.split('/')[:-1])+'/label_'+tempPath.split('/')[-1]+'.npy'

            label = np.load(labelpath)    
            label = cv2.resize(label,(H,W))
            label = Image.fromarray(label) 
            
            
        if (self.split=='unlabelled'): # to pass the augmentation that needs labels
            label=np.zeros((W,H))
            label = Image.fromarray(label)     
               
        if self.transform is not None:
            if (self.split == 'labelled'):
                if random.random() < 0.2: 
                    pilimg = Starburst_augment()(np.array(pilimg),self.filepath_original)  
                if random.random() < 0.2: 
                    pilimg = Line_augment()(np.array(pilimg))    
                if random.random() < 0.2:
                    pilimg = Gaussian_blur()(np.array(pilimg))   
                if random.random() < 0.4:
                    pilimg, label,_,_,_ = Translation()(np.array(pilimg),np.array(label))         

        img = pilimg.copy()    
        if self.split == 'labelled':
            img, label = RandomHorizontalFlip()(img,label)
            
            clip_param = [1,1.2,1.5,1.5,1.5,2][np.random.randint(0,6)]
            grid_size = [2,4,8,8,8,16][np.random.randint(0,6)]
            clahe_param = cv2.createCLAHE(clipLimit=clip_param, tileGridSize=(grid_size,grid_size))
            img = clahe_param.apply(np.array(np.uint8(img)))    
            img = Image.fromarray(img)  
            img = self.transform(img)   

        elif self.split == 'unlabelled':
            img, label = RandomHorizontalFlip()(img,label) #since label is absent sending image twice
            img2= img.copy()
            #data aug_for semi supervised setup
            gamma = [0.8,0.85,0.9,0.95,1.0,1.05,1.10, 1.15, 1.2][np.random.randint(0,9)]


            table = 255.0*(np.linspace(0, 1, 256)**gamma)
            
            img2 = cv2.LUT(np.uint8(np.array(img2)), table)
            if random.random() < 0.2:
                img2 = np.uint8(Gaussian_blur()(np.array(img2))) 
           
            
            clip_param = [1,1.2,1.5,1.5,1.5,2][np.random.randint(0,6)]
            grid_size = [2,4,8,8,8,16][np.random.randint(0,6)]
            clahe_param = cv2.createCLAHE(clipLimit=clip_param, tileGridSize=(grid_size,grid_size))            
            img = clahe_param.apply(np.array(np.uint8(img)))    
            img = Image.fromarray(img)      
            clip_param = [1,1.2,1.5,1.5,1.5,2][np.random.randint(0,6)]
            grid_size = [2,4,8,8,8,16][np.random.randint(0,6)]
            clahe_param = cv2.createCLAHE(clipLimit=clip_param, tileGridSize=(grid_size,grid_size))
            img2 = clahe_param.apply(np.array(np.uint8(img2)))    
            img2 = Image.fromarray(img2)
            
            ang2=np.eye((3))
            ang1=np.eye((3))
            
            imgT=img
            img2T=img2
            if self.mode == 'ssl_augu':
                #Only rotate with 30% probability
                ang2=np.eye((3))
                ang1=np.eye((3))
                if random.random() < 0.5:
                    img2T,ang2=aug_unlabelled(img2)

                if random.random() < 0.5:
                    imgT,ang1=aug_unlabelled(img)
                
                
            img2T =self.transform(img2T)
            imgT = self.transform(imgT)    
           
            img2 =self.transform(img2)
            img = self.transform(img)    
            
        else:
            img = self.clahe.apply(np.array(np.uint8(img)))    
            img = Image.fromarray(img)      
            img = self.transform(img)    

        if (self.split!='test') & (self.split!='unlabelled') & ('images' not in self.split):
            ## This is for boundary aware cross entropy calculation
            spatialWeights = cv2.Canny(np.array(label),0,3)/255
            spatialWeights=cv2.dilate(spatialWeights,(3,3),iterations = 1)*20
            
            ##This is the implementation for the surface loss
            # Distance map for each class
            distMap = []
            for i in range(0, 4):
                distMap.append(one_hot2dist(np.array(label)==i))
            distMap = np.stack(distMap, 0)           
#           spatialWeights=np.float32(distMap) 
            
            
        if (self.split=='test') or ('images' in self.split):
            ##since label, spatialWeights and distMap is not needed for test images
            return img,0,self.list_files[idx],0,0,0,0
        if (self.split == 'unlabelled'):
            return img,img2,self.list_files[idx],imgT,img2T,torch.from_numpy(ang1).type(torch.FloatTensor),torch.from_numpy(ang2).type(torch.FloatTensor)

        label = MaskToTensor()(label)
        return img, label, self.list_files[idx],spatialWeights,np.float32(distMap),0,0

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    split_name='labelled'
    factor=1
    setup='S10_1'

    ds = IrisDataset_ss('Semantic_Segmentation_Dataset',setup,factor,mode='ssl_augu',split=split_name,transform=transform)


    idx=np.random.randint(100)
    idx=0
    img, label, idx,x,y,M1,M2= ds[idx]
    plt.subplot(121)
    if split_name=='unlabelled':
      plt.imshow(np.array(label)[0,:,:],cmap='gray')    
    else:
      plt.imshow(np.array(label))
    plt.subplot(122)
    plt.imshow(np.array(img)[0,:,:],cmap='gray')
    print (M1,M2)

