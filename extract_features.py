# %%
HEADS = 8
last_n = 1
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=5e-6)
parser.add_argument("--num_epochs", type=int, default=200)
parser.add_argument("--num_workers", type=int, default=70)
parser.add_argument("--num_decoder_layers", type=int, default=3)
parser.add_argument("--load_checkpoint", action="store_true")
parser.add_argument("--image_size", type=int, default=256)
args = parser.parse_args()
# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2
import torchvision.transforms.functional

import numpy as np
import cv2
import os
import cv2 as cv
import matplotlib.pyplot as plt
import random

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, mode="train"):
        self.mode = mode
        self.datapath = "/home/wg25r/fastdata/CDNet"
       
        if mode == "train":
            with open(f"{self.datapath}/train.txt") as f:
                self.images = f.read().split("\n")
            assert "lowFramerate_port_0_17fps_in001097.jpg" in self.images
        else: 
            with open(f"{self.datapath}/val.txt") as f:
                self.images = f.read().split("\n")
 

        self.ignore_after = 40 
        self.space_trans = torchvision.transforms.v2.Compose([
            torchvision.transforms.v2.RandomResizedCrop(args.image_size, scale=(0.6, 3)), 
            torchvision.transforms.v2.RandomHorizontalFlip(0.5),
            torchvision.transforms.v2.RandomRotation(20), 
            torchvision.transforms.v2.RandomApply(
                [torchvision.transforms.v2.ElasticTransform(alpha=50)], p=0.2
            ),
            torchvision.transforms.v2.RandomApply(
                [torchvision.transforms.v2.RandomPerspective()], p=0.2
            ),
            torchvision.transforms.v2.RandomApply(
                [torchvision.transforms.v2.RandomAffine(20,  scale=(0.5, 1.1))], p=0.2
            ),
        ])
        self.color_trans = torchvision.transforms.v2.Compose([
            torchvision.transforms.v2.RandomApply([torchvision.transforms.v2.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.4)
        ])

    def __len__(self): 
        if self.mode == "train":
            return int(len(self.images)) 
        else:  
            return int(len(self.images))
        

    def __getitem__(self, idx):  
        filename = self.images[idx]

        current_frame = cv2.resize(cv2.imread(f"{self.datapath}/in/{filename}"), (args.image_size, args.image_size))
        long_bg = cv2.resize(cv2.imread(f"{self.datapath}/long/{filename}"), (args.image_size, args.image_size))
        short_bg = cv2.resize(cv2.imread(f"{self.datapath}/short/{filename}"), (args.image_size, args.image_size)) 
        label_ = cv2.imread(f"{self.datapath}/gt/{filename}")
        label = (label_ == 255) * 255.0
        ROI =  (label_ != 85) * 255.0 
        ROI = cv2.resize(ROI, (args.image_size, args.image_size)).mean(2)[None,:,:]
        label = cv2.resize(label, (args.image_size, args.image_size))

        current_frame = torch.tensor(current_frame).permute(2,0,1)
        long_bg = torch.tensor(long_bg).permute(2,0,1)
        short_bg = torch.tensor(short_bg).permute(2,0,1)
        label = torch.tensor(label).permute(2,0,1)
        ROI = torch.tensor(ROI).float()
        X = torch.cat([current_frame, long_bg, short_bg, ROI], axis=0)
        # print(X.shape)
        
        Y = label.max(0)[0][None,:,:] 

        if self.mode == "train":  
            # X = self.color_trans(X) 
            YX = torch.cat((Y, X), axis=0) 
            YX = self.space_trans(YX)
            Y = YX[:1]/255.0  
            X = YX[1:-1]/255.0
            ROI = YX[-1] > 0 
            ROI = ROI.float().unsqueeze(0).clone()
            X += torch.randn(X.shape) * 0.005 
            X += torch.tensor(cv.resize(np.random.normal(0, 0.005, (10, 10)), X.shape[1:])).float()
            X *= 1 + torch.randn(9)[:,None,None] * 0.005
        else:
            X = X[:-1]/255.0
            Y = Y/255.0 
            ROI = ROI > 0
        Y = torchvision.transforms.functional.resize(Y, (args.image_size//4, args.image_size//4))[0] > 0
        ROI = torchvision.transforms.functional.resize(ROI, (args.image_size//4, args.image_size//4))[0] > 0
        Y = Y.float() #if not this loss will be issue, maybe from the resize, identical not 0
        X[X<0]=0
        return X, Y, ROI, filename


    
# %%
vits8 = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
dataloader = torch.utils.data.DataLoader(MyDataset("val"), batch_size=256, num_workers=args.num_workers, drop_last=False)
print("lowFramerate_port_0_17fps_in001097.jpg" in MyDataset("train").images)
mymodel = vits8.cuda()  
# %%
import wandb
import tqdm 
os.makedirs("/home/wg25r/fastdata/features", exist_ok=True)
os.makedirs("/home/wg25r/fastdata/raw_images", exist_ok=True) 

with torch.no_grad():
    mymodel.eval()
    total_loss = 0 
    total_iou = 0
    for X_val, Y_val, ROI_val, filenames in tqdm.tqdm(dataloader):
        X_val = X_val.cuda().float()
      
        current = (X_val[:,:3])
        long_bg = (X_val[:,3:6])
        short_bg = (X_val[:,6:9])
        ROIs = ROI_val
        ROIs = ROIs.float().clone() 
        current_features = mymodel.get_intermediate_layers(current, 1)[0]
        long_bg_features = mymodel.get_intermediate_layers(long_bg, 1)[0]
        short_bg_features = mymodel.get_intermediate_layers(short_bg, 1)[0]
        ROIs = ROIs.float()
        # for i, (current_feature, long_bg_feature, short_bg_feature, ROI, filename) in enumerate(zip(current_features, long_bg_features, short_bg_features, ROIs, filenames)):
        print(len((current_features, long_bg_features, short_bg_features, ROIs, filenames)[0]))
        for i in range(len(filenames)):
            current_feature = current_features[i]
            long_bg_feature = long_bg_features[i]
            short_bg_feature = short_bg_features[i]
            ROI = ROIs[i]
            filename = filenames[i]

            current_feature = current_feature.cpu().numpy()
            long_bg_feature = long_bg_feature.cpu().numpy()
            short_bg_feature = short_bg_feature.cpu().numpy()
            ROI = ROI.cpu().numpy()
            np.save(f"/home/wg25r/fastdata/features/{filename}_current.npy", current_feature)
            np.save(f"/home/wg25r/fastdata/features/{filename}_long_bg.npy", long_bg_feature)
            np.save(f"/home/wg25r/fastdata/features/{filename}_short_bg.npy", short_bg_feature)
            np.save(f"/home/wg25r/fastdata/features/{filename}_ROI.npy", ROI)
            np.save(f"/home/wg25r/fastdata/features/{filename}_label.npy", Y_val[i].cpu().numpy())
            cv2.imwrite(f"/home/wg25r/fastdata/raw_images/{filename}_current.png", (current[i].cpu().numpy().transpose(1,2,0)*255).astype(np.uint8))
            cv2.imwrite(f"/home/wg25r/fastdata/raw_images/{filename}_long_bg.png", (long_bg[i].cpu().numpy().transpose(1,2,0)*255).astype(np.uint8))
            cv2.imwrite(f"/home/wg25r/fastdata/raw_images/{filename}_short_bg.png", (short_bg[i].cpu().numpy().transpose(1,2,0)*255).astype(np.uint8))
            cv2.imwrite(f"/home/wg25r/fastdata/raw_images/{filename}_ROI.png", (ROI*255).astype(np.uint8))
            cv2.imwrite(f"/home/wg25r/fastdata/raw_images/{filename}_label.png", (Y_val[i].cpu().numpy()*255).astype(np.uint8))

    total_iou /= len(dataloader) 
