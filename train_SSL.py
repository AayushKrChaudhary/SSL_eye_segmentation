#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 17:09:34 2020
Authors : Aayush Kumar Chaudhary,Prashnna K Gyawali,Linwei Wang,Jeff B Pelz
Paper : Semi-Supervised Learning for Eye Image Segmentation

Some code snipets are from 
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
from train_val_functions import unlabelled_inverse, train_setup, valid_setup

if __name__ == '__main__': 
    args = parse_args()
    kwargs = vars(args)

    if args.useGPU:
        device_info="cuda:"+args.deviceID
        device=torch.device(device_info)
        torch.cuda.manual_seed(12)
    else:
        device=torch.device("cpu")
        torch.manual_seed(12)
        
    torch.backends.cudnn.deterministic=False
  
    if args.model not in model_dict:
        print ("Model not found !!!")
        print ("valid models are:",list(model_dict.keys()))
        exit(1)
    
    LOGDIR = 'logs/{}'.format(args.expname)
    os.makedirs(LOGDIR,exist_ok=True)
    os.makedirs(LOGDIR+'/models_ss',exist_ok=True)
    logger = Logger(os.path.join(LOGDIR,'logs.log'))
    
    model = model_dict[args.model]
    filename = args.load
    
    if filename is not None:
        print ('...................................................')
        print ('Loaded from file')
        print ('...................................................')

        model.load_state_dict(torch.load(filename))
        model = model.to(device)
        
    if args.mode not in ['labeled','ssl','ssl_augu']:
        print ('mIoU ERROR occured: args mode not in list')
    
    model  = model.to(device)
    torch.save(model.state_dict(), '{}/models_ss/'.format(LOGDIR)+args.model+'{}.pkl'.format('_0'))
    nparams = get_nparams(model)

    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)      
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=5)

    criterion = CrossEntropyLoss2d()
    criterion_DICE = GeneralizedDiceLoss(softmax=True, reduction=True)
    criterion_SL = SurfaceLoss()
    
    Path2file = args.dataset

    print ('Args_label is '+str(args.labeltype))
    labelled = IrisDataset_ss(filepath = Path2file,setup=args.labeltype,factor_decrease=args.factor,
                              split='labelled', transform = transform, **kwargs)
    labelledloader = DataLoader(labelled, batch_size =  args.bs,
                             shuffle=True, num_workers = args.workers)

    unlabelled = IrisDataset_ss(filepath = Path2file,factor_decrease=args.factor,split='unlabelled',
                             transform = transform, **kwargs)
    unlabelledloader = DataLoader(unlabelled, batch_size =  args.bs_U,
                             shuffle=True, num_workers = args.workers)
 
    valid = IrisDataset_ss(filepath = Path2file ,factor_decrease=args.factor, split='validation',
                            transform = transform, **kwargs)
    validloader = DataLoader(valid, batch_size = args.bs,
                             shuffle= False, num_workers = args.workers)
  
    alpha=np.zeros(((args.epochs*5)))
    if args.SegLoss:
        alpha[0:np.min([125,args.epochs])]= 1 - np.arange(1,np.min([125,args.epochs])+1)/np.min([125,args.epochs])
        if args.epochs>125:
            alpha[125:]=0

    beta=np.zeros(((args.epochs*5)))
    beta=np.arange(1,np.min([250,args.epochs])+1)/np.min([250,args.epochs])
    beta[0:5]=0
  
    T = args.T
    ious = []

    iterations=0

    for epoch in range(args.epochs):
        ious,iterations=train_setup(labelledloader,unlabelledloader, model, criterion,criterion_DICE,criterion_SL, optimizer, scheduler, device, iterations,alpha,beta, logger,T, args.mode)            
        logger.write('Epoch:{}, Train mIoU: {}'.format(iterations,np.average(ious)))
        lossvalid , miou = valid_setup(validloader,model,alpha[iterations//1000],device,criterion, criterion_DICE, criterion_SL)
        totalperf = total_metric(nparams,miou)
        f = 'Epoch:{}, Valid Loss: {:.3f} mIoU: {} Complexity: {} total: {}'
        logger.write(f.format(epoch,lossvalid, miou,nparams,totalperf))

        ##save the model every epoch
        if epoch %1 == 0:
            torch.save(model.state_dict(), '{}/models_ss/ritnet{}.pkl'.format(LOGDIR,epoch))


