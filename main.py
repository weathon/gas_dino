# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=5e-6)
parser.add_argument("--num_epochs", type=int, default=200)
parser.add_argument("--num_workers", type=int, default=70)
parser.add_argument("--num_decoder_layers", type=int, default=3)
parser.add_argument("--load_checkpoint", action="store_true")
parser.add_argument("--image_size", type=int, default=256)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--ffn_dim", type=int, default=2048)
parser.add_argument("--last_n", type=int, default=12)

args = parser.parse_args()
last_n = args.last_n
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
        print("Using MyDataset")
        self.mode = mode
        self.datapath = "/home/wg25r/fastdata/CDNet"
       
        if mode == "train":
            with open(f"{self.datapath}/train.txt") as f:
                self.images = f.read().split("\n")
        else: 
            with open(f"{self.datapath}/val.txt") as f:
                self.images = f.read().split("\n")
            self.images = random.sample(self.images, 128)


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
            ROI = ROI.float().unsqueeze(0)
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
        # print(X.shape, Y.shape, ROI.shape) 
        return X, Y, ROI 


class PreExtractedDataset(torch.utils.data.Dataset):
    def __init__(self, mode="train"):
        print("Using PreExtractedDataset")
        self.mode = mode
        self.datapath = "/home/wg25r/fastdata/CDNet"
        self.feature_path = "/home/wg25r/fastdata/features"
        if mode == "train":
            with open(f"{self.datapath}/train.txt") as f:
                self.images = f.read().split("\n")
        else: 
            with open(f"{self.datapath}/val.txt") as f:
                self.images = f.read().split("\n")
            self.images = random.sample(self.images, 128)

            with open(f"{self.datapath}/train.txt") as f:
                train_images = f.read().split("\n")

            with open(f"{self.datapath}/val.txt") as f:
                val_images = f.read().split("\n")

            assert set(train_images).intersection(set(val_images)) == set(), "Train and Val overlap"
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        filename = self.images[idx]
        current_frame = np.load(f"{self.feature_path}/{filename}_current.npy")
        long_bg = np.load(f"{self.feature_path}/{filename}_long_bg.npy")
        short_bg = np.load(f"{self.feature_path}/{filename}_short_bg.npy")
        label = np.load(f"{self.feature_path}/{filename}_label.npy")
        ROI = np.load(f"{self.feature_path}/{filename}_ROI.npy")[None,:,:]
        X = np.stack([current_frame, long_bg, short_bg], axis=0)
        # print(X.shape)
        Y = label[None, :, :]
        Y = torch.tensor(Y).float()
        X = torch.tensor(X).float()
        ROI = torch.tensor(ROI).float()
        X[X<0] = 0
        Y = torchvision.transforms.functional.resize(Y, (args.image_size//4, args.image_size//4))[0] > 0
        ROI = torchvision.transforms.functional.resize(ROI, (args.image_size//4, args.image_size//4))[0] > 0
        Y = Y.float()
        ROI = ROI.float()
        return X, Y, ROI

        
# %%
vits8 = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')


# %%
class BCA_lite(nn.Module):
    """
    Background-CurrentFrame Attention
    """
    def __init__(self, dim=384):
        super(BCA_lite, self).__init__()
        self.dim = dim
        self.key_projection = nn.Linear(dim * 12, dim)
        self.query_projection = nn.Linear(dim * 12, dim)
        self.value_projection = nn.Linear(dim * 2 * 12, dim)
        self.mlp = nn.Sequential(
            nn.Linear(2 * dim, 1024),
            nn.Dropout(args.dropout),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.Dropout(args.dropout),
            nn.ReLU(), 
            nn.Linear(2048, dim),
            nn.Dropout(args.dropout)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.head = nn.Linear(dim * 2, dim)
          

    def forward(self, current_frame, long_background, short_background):
        """
        long_background: torch.Tensor, shape (batch, L, dim)
        short_background: torch.Tensor, shape (batch, L, dim)
        current_frame: torch.Tensor, shape (batch, L, dim)
        """

        # long background attention with frame
        key = self.key_projection(long_background)
        query = self.query_projection(current_frame)
        value = self.value_projection(torch.concatenate([long_background, current_frame], dim=-1))

        attn_score = torch.einsum("bld,bld->bl", query, key) / self.dim**0.5
        attn_score = F.softmax(attn_score, dim=1)
        attn_output = torch.einsum("bl,bld->bld", attn_score, value)
        attn_output = self.norm1(attn_output + current_frame) 
        mlp_output = self.mlp(torch.concatenate([attn_output, current_frame], dim=-1))
        mlp_output = self.norm2(mlp_output + attn_output)

        # short background attention with frame
        key = self.key_projection(short_background)
        query = self.query_projection(current_frame)
        value = self.value_projection(torch.concatenate([short_background, current_frame], dim=-1))

        attn_score = torch.einsum("bld,bld->bl", query, key) / self.dim**0.5
        attn_score = F.softmax(attn_score, dim=1)
        attn_output = torch.einsum("bl,bld->bld", attn_score, value)
        attn_output = self.norm1(attn_output + mlp_output)
        mlp_output = self.mlp(torch.concatenate([attn_output, mlp_output], dim=-1))
        mlp_output2 = self.norm2(mlp_output + attn_output)
        return self.head(torch.cat([mlp_output, mlp_output2], dim=-1))
    
        
        

        

# %%
class MyModel(nn.Module):
    def __init__(self, backbone):
        super(MyModel, self).__init__()
        self.backbone = backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.bca = BCA_lite()
        self.decoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=384 * 2, nhead=8, dim_feedforward=args.ffn_dim, dropout=args.dropout, batch_first=True),
                                             num_layers=args.num_decoder_layers)
        self.upsample = nn.Upsample(scale_factor=2, mode='bicubic')

        self.head = nn.Sequential(
            nn.Conv2d(384, 32, 1),
            nn.Dropout(args.dropout), 
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
        )   
        self.textual_encoder = nn.AvgPool2d(4)

    def forward(self, current_frame, long_bg, short_bg):
        long_bg = self.backbone.get_intermediate_layers(long_bg, n=last_n)
        short_bg = self.backbone.get_intermediate_layers(short_bg, n=last_n)
        current_frame = self.backbone.get_intermediate_layers(current_frame, n=last_n)
        current_frame = torch.concatenate(current_frame, dim=-1)
        long_bg = torch.concatenate(long_bg, dim=-1)
        short_bg = torch.concatenate(short_bg, dim=-1)
        print(current_frame.shape, long_bg.shape, short_bg.shape)

        current_frame = current_frame[:,1:,:]
        long_bg = long_bg[:,1:,:]
        short_bg = short_bg[:,1:,:]

        current_frame = self.bca(current_frame, long_bg, short_bg)
        current_frame = self.decoder(current_frame)

        current_frame = current_frame.reshape(long_bg.shape[0], args.image_size//8, args.image_size//8, 384).permute(0,3,1,2)
        current_frame = self.upsample(current_frame)
        return self.head(current_frame)
    


# MyModel(vits8)(torch.randn(1, 3, args.image_size, args.image_size), torch.randn(1, 3, args.image_size, args.image_size), torch.randn(1, 3, args.image_size, args.image_size)).shape


# %%
cpu_counts = args.num_workers
train_dataloader = torch.utils.data.DataLoader(MyDataset("train"), batch_size=args.batch_size, shuffle=True, num_workers=cpu_counts, persistent_workers=True, prefetch_factor=3)
val_dataloader = torch.utils.data.DataLoader(MyDataset("val"), batch_size=args.batch_size, shuffle=True, num_workers=cpu_counts, persistent_workers=True, prefetch_factor=3)

print("Train", len(train_dataloader), "Val", len(val_dataloader))

# %%
import gc
gc.collect()
torch.cuda.empty_cache()

# %%
def iou_loss(pred, target):
    pred = torch.sigmoid(pred)
    assert pred.shape == target.shape, f"pred shape {pred.shape} target shape {target.shape}"
    e = 1e-6
    iou = ((pred * target).sum() + e) / (pred.sum() + target.sum() - (pred * target).sum() + e)
    # print(pred.sum(), target.sum(), (pred * target).sum())
    return 1 - iou

# %%
import wandb
mymodel = torch.nn.DataParallel(MyModel(vits8).cuda()) 
optimizer = torch.optim.Adam(mymodel.parameters(), lr=args.lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, eta_min=1e-7) 
wandb.init(resume=False, config = args)

assert iou_loss(torch.tensor([10000.0]), torch.tensor([1.0])) == 0, "iou loss not 0, but " + str(iou_loss(torch.tensor([1.0]), torch.tensor([1.0])))
# %%
loss_fn = iou_loss
epoch = 0

if args.load_checkpoint:
    mymodel.load_state_dict(torch.load("main.pth"))
    print("Loaded main.pth")

import tqdm
while 1:
    for i, (X, Y, ROI) in enumerate(train_dataloader):
        if i%50 == 0:
            scheduler.step() 

        if i%50==0:
            with torch.no_grad():
                mymodel.eval()
                total_loss = 0
                total_iou = 0
                for X_val, Y_val, ROI_val in val_dataloader:
                    # print(X_val.shape, Y_val.shape, ROI_val.shape)
                    X_val = X_val.cuda().float()
                    Y_val = Y_val.cuda().float() * (ROI_val.cuda().float() > 0) 
                    # test_loss = loss_fn((Y_val - 0.5) * 1000, Y_val)
                    # print(Y_val.shape)
                    # assert test_loss == 0, f"Test loss is {test_loss}" #why test loss not 0 
                    pred = mymodel(X_val[:,:3], X_val[:,3:6], X_val[:,6:]) * (ROI_val.cuda().float().unsqueeze(1) > 0)
                    loss = loss_fn(pred, Y_val.cuda().unsqueeze(1)) 
                    total_loss += loss.item() 
                    iou = (((pred > 0) & (Y_val.cuda().unsqueeze(1) > 0)).float().mean() + 1e-6)/(((pred > 0) | ((Y_val.cuda().unsqueeze(1) > 0))).float().mean() + 1e-6)
                    total_iou += iou.float() 
                total_iou /= len(val_dataloader) 


                total_loss /= len(val_dataloader) 
                for b in range(1): 
                    pred_ = pred[b].reshape((args.image_size//4, args.image_size//4))
                    ROI_tmp = ROI_val[b].cpu().detach().numpy()
                    ROI_green = np.stack([np.zeros_like(ROI_tmp), ROI_tmp, np.zeros_like(ROI_tmp)], axis=-1)
                    wandb.log({"val_iou": total_iou, "has_gas_ratio":(Y_val.sum((1,2)) > 0).float().sum()/len(Y_val),
                        "real": wandb.Image(Y_val[b].cpu().detach().numpy().reshape(args.image_size//4, args.image_size//4)), 
                        "pred": wandb.Image(pred_.cpu().detach().numpy()>0),
                        # "X_val": wandb.Image(X_val[b][:3].cpu().permute(1,2,0).detach().numpy()),
                        # "X_BGS": wandb.Image(X_val[b][3:6].cpu().permute(1,2,0).detach().numpy()),

                        "ROI": wandb.Image(ROI_green), 
                        "val_loss": total_loss}) 
                mymodel.train()
                epoch += 1
            # zero the bar
            progress = tqdm.tqdm(range(50), ncols=50)
        progress.update(1)

            
        optimizer.zero_grad()
        X = X.cuda().float() 
        Y = Y.cuda().float() * (ROI.cuda().float() > 0)
        pred = mymodel(X[:,0], X[:,1], X[:,2]) * (ROI.cuda().float().unsqueeze(1) > 0)
        # loss = torchvision.ops.sigmoid_focal_loss(pred, Y.cuda().unsqueeze(1), alpha=1/(labels == 1).sum(), gamma=10, reduction="mean")
        loss = loss_fn(pred, Y.cuda().unsqueeze(1))
        acc = (pred > 0) == Y.cuda().unsqueeze(1)
        # f1 = f1_score(Y.unsqueeze(1).cpu().detach().numpy().reshape(-1).astype(int), pred.cpu().detach().numpy().reshape(-1) > 0)
        iou = (((pred > 0) & (Y.cuda().unsqueeze(1) > 0)).float().mean()  + 1e-6 )/(((pred > 0) | (Y.cuda().unsqueeze(1) > 0)) + 1e-6).float().mean()
        loss.backward() 
        optimizer.step()
        wandb.log({"loss": loss.item(), "acc": acc.float().mean().item(), "iou": iou.float(), "lr": optimizer.param_groups[0]["lr"]}) # cannot do iou mean here otherwise it average non overlapping area
        # same pred as image to wandb
        if i % 2000 == 0:
            torch.save(mymodel.state_dict(), "main.pth")

# %%
pred.shape, Y_val.unsqueeze(1).shape


