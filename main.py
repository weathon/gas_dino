# %%
import os
import random
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=5e-6)
parser.add_argument("--num_epochs", type=int, default=200)
parser.add_argument("--num_workers", type=int, default=30)
parser.add_argument("--num_decoder_layers", type=int, default=3)
parser.add_argument("--load_checkpoint", type=str, default=None)
parser.add_argument("--image_size", type=int, default=256)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--ffn_dim", type=int, default=1024)
parser.add_argument("--last_n", type=int, default=12)
parser.add_argument("--backbone", type=str, default="segformer", choices=["segformer", "dino"])
parser.add_argument("--frozen_backbone", action="store_true")
parser.add_argument("--decoder", type=str, default="conv", choices=["transformer", "conv", "none"])
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--loss_fn", type=str, default="iou", choices=["iou", "f1"])
parser.add_argument("--use_bsvunet2_style_dataaug", action="store_true")
parser.add_argument("--conf_pen", type=float, default=0)
parser.add_argument("--base_aug_spice", type=float, default=0.3)
parser.add_argument("--use_train_as_val", action="store_true")
parser.add_argument("--parallel_bca", type=int, default=0)
parser.add_argument("--note", type=str, default="")
parser.add_argument("--log_file", type=str, default="log.txt")
parser.add_argument("--fusion", type=str, default="concat", choices=["concat", "cross_attention", "early", "slow"])
parser.add_argument("--use_tqdm", action="store_true")
parser.add_argument("--cross_attention_type", type=str, default="lite", choices=["lite", "full"])
parser.add_argument("--gpus", type=str, default="0")
parser.add_argument("--print_every", type=int, default=2000)
parser.add_argument("--normalize_image", action="store_true")
parser.add_argument("--resume", action="store_true")
parser.add_argument("--show_sample", action="store_true")
parser.add_argument("--val_size", type=int, default=2048)
parser.add_argument("--save_path", type=str, default=f"{random.randint(10000000,99999999)}.pth")
parser.add_argument("--use_optical_flow", action="store_true")
parser.add_argument("--strong_crop", action="store_true")

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
print(f"Saving model checkpoints into {args.save_path}")

# if args.use_optical_flow:
#     raise NotImplementedError("Optical flow is not implemented yet")

def write_log(log):
    with open(args.log_file, "a") as f:
        f.write(log + "\n")

write_log("")

if args.use_train_as_val:
    # prrint yellow waning in bash
    print("\033[93m" + "Warning: Using train as val This is meant for purposely introduce overfitting for debugging purpose" + "\033[0m")

if args.backbone == "segformer":  
    args.last_n = 1
    print("\033[93m" + "Warning: last_n set to 1 for segformer" + "\033[0m")


# if last_n is not 1, and concate, set last_n to 1
if args.last_n != 1 and args.fusion == "concat":
    print("\033[93m" + "Warning: last_n set to 1 for concat" + "\033[0m")
    args.last_n = 1
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
            if not args.use_train_as_val:
                with open(f"{self.datapath}/val.txt") as f:
                    self.images = f.read().split("\n")
            else:
                with open(f"{self.datapath}/train.txt") as f:
                    self.images = f.read().split("\n")
            random.seed(19890604)
            if args.val_size > len(self.images):
                print("\033[93m" + f"Warning: val_size is larger than the dataset size {len(self.images)}, using the whole dataset."
                + "\033[0m")
                    
                args.val_size = len(self.images)
            self.images = random.sample(self.images, args.val_size)
        

        with open(f"{self.datapath}/train.txt") as f:
            train_ = f.read().split("\n") 

        with open(f"{self.datapath}/val.txt") as f:
            val_ = f.read().split("\n") 

        assert set(train_).intersection(set(val_)) == set(), "Train and Val overlap"

        self.ignore_after = 40  
        self.space_trans = torchvision.transforms.v2.Compose([
            # torchvision.transforms.v2.RandomApply([torchvision.transforms.v2.RandomResizedCrop(args.image_size, scale=(1 - args.base_aug_spice/4, 2))], p=args.base_aug_spice),
            torchvision.transforms.v2.RandomApply([torchvision.transforms.v2.RandomResizedCrop(args.image_size, scale=((0.1 if args.strong_crop else 0.8), 2))], p=args.base_aug_spice),
            torchvision.transforms.v2.RandomApply([torchvision.transforms.v2.RandomHorizontalFlip(0.5)], p=args.base_aug_spice),
            torchvision.transforms.v2.RandomApply([torchvision.transforms.v2.RandomRotation(50)], p=args.base_aug_spice), 
            torchvision.transforms.v2.RandomApply(  
                [torchvision.transforms.v2.ElasticTransform(alpha=50)], p=args.base_aug_spice 
            ),
            torchvision.transforms.v2.RandomApply( 
                [torchvision.transforms.v2.RandomPerspective()], p=args.base_aug_spice 
            ),
            torchvision.transforms.v2.RandomApply( 
                [torchvision.transforms.v2.RandomAffine(30,  scale=(0.8, 1.1))], p=args.base_aug_spice
            ),            
        ])
        # mean for RGB RGB RGB 3 images
        if args.use_optical_flow:
            self.mean = torch.tensor([0.485, 0.456, 0.406, 0.485, 0.456, 0.406, 0.485, 0.456, 0.406, 0.485, 0.456, 0.406])
            self.std = torch.tensor([0.229, 0.224, 0.225, 0.229, 0.224, 0.225, 0.229, 0.224, 0.225, 0.229, 0.224, 0.225])
        else:
            self.mean = torch.tensor([0.485, 0.456, 0.406, 0.485, 0.456, 0.406, 0.485, 0.456, 0.406])
            self.std = torch.tensor([0.229, 0.224, 0.225, 0.229, 0.224, 0.225, 0.229, 0.224, 0.225])
        if args.use_bsvunet2_style_dataaug:
            self.sperate_aug = torchvision.transforms.v2.Compose([
                torchvision.transforms.v2.RandomResizedCrop(args.image_size, scale=(0.95, 1.05)),
                torchvision.transforms.v2.RandomPerspective(0.1),
                torchvision.transforms.v2.RandomRotation(10),
                torchvision.transforms.v2.ColorJitter(0.1, 0.1, 0.1, 0.1), 
                torchvision.transforms.v2.RandomApply(
                    [torchvision.transforms.v2.ElasticTransform(alpha=5)], p=0.1
                ),
                torchvision.transforms.v2.RandomApply([self.pan], p=0.1),
                torchvision.transforms.v2.RandomApply(
                    [
                        torchvision.transforms.v2.GaussianBlur(5, sigma=(0.1, 2.0))
                    ], 0.05
                )
            ])


        self.color_trans = torchvision.transforms.v2.Compose([
            torchvision.transforms.v2.RandomApply([torchvision.transforms.v2.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.4)
        ])
        self.pancrop = torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.9, 1.1))

    def pan(self, img): 
        shifteds = []
        for i in range(random.randint(10, 30)):
            shift = random.randint(10, 30)
            tmp = torch.roll(img, shifts=shift, dims=1) 
            tmp = self.pancrop(tmp)
            shifteds.append(tmp)
        return torch.stack(shifteds).mean(0)
            


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
        if args.use_optical_flow:
            flow = cv2.imread(f"{self.datapath}/flow/{filename}")
            flow = cv2.resize(flow, (args.image_size, args.image_size)) 
        label = (label_ >= 250) * 255.0 # because there is resave so we need this
        # ROI =  (label_ != 85) * 255.0 
        ROI = ((80 >= label) & (label >= 90)) * 255 #need it !=85

        ROI = (80 <= label_ <= 90) * 255.0
        ROI = cv2.resize(ROI, (args.image_size, args.image_size)).mean(2)[None,:,:]
        label = cv2.resize(label, (args.image_size, args.image_size))

        current_frame = torch.tensor(current_frame).permute(2,0,1)
        long_bg = torch.tensor(long_bg).permute(2,0,1)
        short_bg = torch.tensor(short_bg).permute(2,0,1)
        label = torch.tensor(label).permute(2,0,1)
        if args.use_optical_flow:
            flow = torch.tensor(flow).permute(2,0,1)
        ROI = torch.tensor(ROI).float()
        if args.use_optical_flow:
            X = torch.cat([current_frame, long_bg, short_bg, flow, ROI], axis=0)
        else:
            X = torch.cat([current_frame, long_bg, short_bg, ROI], axis=0)
        
        Y = label.max(0)[0][None,:,:] 

        if self.mode == "train":  
            # X = self.color_trans(X) 
            YX = torch.cat((Y, X), axis=0) 
            # if args.use_base_aug:
            YX = self.space_trans(YX)
            Y = YX[:1]/255.0  
            X = YX[1:-1]/255.0
            if args.use_bsvunet2_style_dataaug:
                # X[:3] = self.sperate_aug(X[:3]) # do not do for current since it needss to be used for output location
                X[3:6] = self.sperate_aug(X[3:6])
                X[6:9] = self.sperate_aug(X[6:9]) 

            ROI = YX[-1] > 0
            ROI = ROI.float().unsqueeze(0)
            X += torch.randn(X.shape) * 0.008 
            X += torch.tensor(cv.resize(np.random.normal(0, 0.01, (10, 10)), X.shape[1:])).float()
            X *= 1 + torch.randn(X.shape[0])[:,None,None] * 0.1
        else:
            X = X[:-1]/255.0
            Y = Y/255.0 
            ROI = ROI > 0 
        if args.normalize_image:
            X = (X - self.mean[:,None,None]) / self.std[:,None,None] 
        Y = torchvision.transforms.functional.resize(Y, (args.image_size//4, args.image_size//4))[0] > 0
        ROI = torchvision.transforms.functional.resize(ROI, (args.image_size//4, args.image_size//4))[0] > 0
        Y = Y.float() #if not this loss will be issue, maybe from the resize, identical not 0
        X[X<0]=0
        # print(X.shape, Y.shape, ROI.shape) 
        if self.mode != "train":
            return X, Y, ROI, filename
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
class BCA(nn.Module):
    """
    Background-CurrentFrame Attention
    """
    def __init__(self, dim=384):
        super(BCA, self).__init__()
        self.dim = dim
        if args.cross_attention_type == "lite":
            self.key_projection = nn.Linear(dim * args.last_n, dim * args.last_n)
            self.query_projection = nn.Linear(dim * args.last_n, dim * args.last_n)
            self.value_projection = nn.Linear(dim * 2 * args.last_n, dim * args.last_n)
            self.projection_dropout = nn.Dropout(args.dropout)
            self.mlp = nn.Sequential(
                nn.Linear(2 * dim * args.last_n, args.ffn_dim),
                nn.Dropout(args.dropout),
                nn.ReLU(),
                nn.Linear(args.ffn_dim, args.ffn_dim),
                nn.Dropout(args.dropout), 
                nn.ReLU(), 
                nn.Linear(args.ffn_dim, dim * args.last_n),
                nn.Dropout(args.dropout)
            )
            self.norm1 = nn.LayerNorm(dim * args.last_n) 
            self.norm2 = nn.LayerNorm(dim * args.last_n)
            self.head = nn.Linear(dim * args.last_n * 2, dim)
        elif args.cross_attention_type == "full":
            self.cross_attention = nn.MultiheadAttention(dim, 8, dropout=args.dropout, batch_first=True, vdim = 2 * dim, kdim = dim)
            self.mlp = nn.Sequential(
                nn.Linear(2 * dim, args.ffn_dim),
                nn.Dropout(args.dropout),
                nn.ReLU(),
                nn.Linear(args.ffn_dim, args.ffn_dim),
                nn.Dropout(args.dropout), 
                nn.ReLU(), 
                nn.Linear(args.ffn_dim, dim),
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
        if args.cross_attention_type == "lite":
            # long background attention with frame
            key = self.projection_dropout(self.key_projection(long_background))
            query = self.projection_dropout(self.query_projection(current_frame))
            value = self.projection_dropout(self.value_projection(torch.concatenate([long_background, current_frame], dim=-1)))

            attn_score = torch.einsum("bld,bld->bl", query, key) / self.dim**0.5
            attn_score = F.softmax(attn_score, dim=1)
            attn_output = torch.einsum("bl,bld->bld", attn_score, value)
            attn_output = self.norm1(attn_output + current_frame) 
            mlp_output = self.mlp(torch.concatenate([attn_output, current_frame], dim=-1))
            mlp_output = self.norm2(mlp_output + attn_output)

            # short background attention with frame
            key = self.projection_dropout(self.key_projection(short_background))
            query = self.projection_dropout(self.query_projection(current_frame))
            value = self.projection_dropout(self.value_projection(torch.concatenate([short_background, current_frame], dim=-1)))

            attn_score = torch.einsum("bld,bld->bl", query, key) / self.dim**0.5
            attn_score = F.softmax(attn_score, dim=1)
            attn_output = torch.einsum("bl,bld->bld", attn_score, value) 
            attn_output = self.norm1(attn_output + mlp_output)
            mlp_output = self.mlp(torch.concatenate([attn_output, mlp_output], dim=-1))
            mlp_output2 = self.norm2(mlp_output + attn_output)
            return self.head(torch.cat([mlp_output, mlp_output2], dim=-1))
        elif args.cross_attention_type == "full":
            key = long_background
            query = current_frame
            value = torch.cat([long_background, current_frame], dim=-1)
            attn_output, _ = self.cross_attention(query, key, value)
            attn_output = self.norm1(attn_output + current_frame)
            mlp_output = self.mlp(torch.cat([attn_output, current_frame], dim=-1))
            mlp_output1 = self.norm2(mlp_output + attn_output)

            key = short_background
            query = current_frame
            value = torch.cat([short_background, current_frame], dim=-1)
            attn_output, _ = self.cross_attention(query, key, value)
            attn_output = self.norm1(attn_output + mlp_output)
            mlp_output = self.mlp(torch.cat([attn_output, mlp_output], dim=-1))
            mlp_output2 = self.norm2(mlp_output + attn_output)

            return self.head(torch.cat([mlp_output1, mlp_output2], dim=-1)) 


            
        

        



# %%
class MyModel(nn.Module):
    def __init__(self, backbone):
        super(MyModel, self).__init__()
        self.backbone = backbone
        if args.frozen_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        if args.fusion == "cross_attention":
            self.bca = BCA()
            self.bca2 = BCA()
            if args.parallel_bca != 0:
                self.bcas = nn.ModuleList([BCA() for _ in range(args.parallel_bca)])
                self.bcas2 = nn.ModuleList([BCA() for _ in range(args.parallel_bca)])
                self.down_fc = nn.Linear(384 * args.parallel_bca, 384)
                self.down_fc2 = nn.Linear(384 * args.parallel_bca, 384)
        elif args.fusion == "concat":
            self.fusion = nn.Linear(384 * 3, 384)
        elif args.fusion == "early": 
            if args.use_optical_flow:
                self.backbone.segformer.encoder.patch_embeddings[0].proj = torch.nn.Conv2d(12, 64, kernel_size=(7, 7), stride=(4, 4), padding=(3, 3))
            else:
                self.backbone.segformer.encoder.patch_embeddings[0].proj = torch.nn.Conv2d(9, 64, kernel_size=(7, 7), stride=(4, 4), padding=(3, 3))
            self.proj_early = torch.nn.Sequential(
                torch.nn.Dropout1d(args.dropout),
                torch.nn.Conv1d(256, 384, 1)
            )
        elif args.fusion == "slow":
            from transformers.models.segformer import TwoStreamSegformerEncoder, SegformerConfig, TwoStreamSegformerModel, TwoStreamSegformerForSemanticSegmentation
            self.backbone = TwoStreamSegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b1-finetuned-ade-512-512")
            self.backbone.decode_head.classifier = torch.nn.Identity()

            self.proj_slow = torch.nn.Sequential(
                torch.nn.Dropout1d(args.dropout),
                torch.nn.Conv1d(256, 384, 1)
            )

        if args.decoder == "conv":
            self.decoder = nn.Sequential(
                nn.Conv2d(384, args.ffn_dim, 3, padding='same'),
                nn.Dropout(args.dropout),
                nn.GELU(), 
                nn.Conv2d(args.ffn_dim, args.ffn_dim, 3, padding='same'),
                nn.Dropout(args.dropout),
                nn.GELU(),
                nn.Conv2d(args.ffn_dim, 384, 3, padding='same'),
            )
        elif args.decoder == "transformer":
            self.decoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=384, nhead=8, dim_feedforward=args.ffn_dim, dropout=args.dropout
                ), 
                num_layers=args.num_decoder_layers
            )
        elif args.decoder == "none":
            self.decoder = nn.Identity()
            
        self.upsample = nn.Upsample(scale_factor=2, mode='bicubic')

        self.head = nn.Sequential(
            nn.Conv2d(384, 1, 5, padding='same'),
        )   
        self.textual_encoder = nn.AvgPool2d(4)
        if args.backbone == "segformer":
            self.projection = nn.Linear(256, 384)

    def forward(self, current_frame, long_bg, short_bg):
            # short_bg could contain flow as well 
            if args.backbone == "dino": 
                # warning dino is not fully implemented, yellow text
                print("\033[93m" + "Warning: DINO is not fully implemented, the output may not be as expected" + "\033[0m")
                long_bg = self.backbone.get_intermediate_layers(long_bg, n=last_n)
                short_bg = self.backbone.get_intermediate_layers(short_bg, n=last_n)
                current_frame = self.backbone.get_intermediate_layers(current_frame, n=last_n)
                current_frame = torch.concatenate(current_frame, dim=-1)
                long_bg = torch.concatenate(long_bg, dim=-1)
                short_bg = torch.concatenate(short_bg, dim=-1)
                current_frame = current_frame[:,1:,:]
                long_bg = long_bg[:,1:,:] 
                short_bg = short_bg[:,1:,:]

                if args.fusion == "cross_attention":
                    current_frame = self.bca(current_frame, long_bg, short_bg)
                    current_frame = self.bca2(current_frame, long_bg, short_bg)
                    if args.parallel_bca != 0:
                        current_frames = []
                        for bca in self.bcas:
                            current_frames.append(bca(current_frame, long_bg, short_bg))
                        current_frame = self.down_fc(torch.cat(current_frames, dim=-1))

                        current_frames = []
                        for bca in self.bcas2:
                            current_frames.append(bca(current_frame, long_bg, short_bg))
                        current_frame = self.down_fc2(torch.cat(current_frames, dim=-1))
                elif args.fusion == "concat":
                    current_frame = self.fusion(torch.cat([current_frame, long_bg, short_bg], dim=-1))

                if args.decoder == "conv":
                    current_frame = current_frame.reshape(long_bg.shape[0], args.image_size//8, args.image_size//8, 384).permute(0,3,1,2)
                    current_frame = self.decoder(current_frame) + current_frame
                elif args.decoder == "transformer":
                    current_frame = self.decoder(current_frame)
                    current_frame = current_frame.reshape(long_bg.shape[0], args.image_size//8, args.image_size//8, 384).permute(0,3,1,2)
                else:
                    current_frame = current_frame.reshape(long_bg.shape[0], args.image_size//8, args.image_size//8, 384).permute(0,3,1,2)
                current_frame = self.upsample(current_frame)
                return self.head(current_frame)     
        
            else:
                # print(long_bg.shape, short_bg.shape, current_frame.shape)
                if args.fusion == "early":
                    feature = torch.cat([current_frame, long_bg, short_bg], dim=1)
                    current_frame = self.backbone(feature).logits.squeeze(1).flatten(-2, -1)
                    current_frame = self.proj_early(current_frame).permute(0, 2, 1)
                elif args.fusion == "slow": 
                    current_frame = self.backbone(current_frame, long_bg, short_bg).logits.squeeze(1).flatten(-2, -1)
                    current_frame = self.proj_slow(current_frame).permute(0, 2, 1)
                else:
                    long_bg = self.projection(self.backbone(long_bg).logits.flatten(-2, -1).permute(0, 2, 1))
                    short_bg = self.projection(self.backbone(short_bg).logits.flatten(-2, -1).permute(0, 2, 1))
                    current_frame = self.projection(self.backbone(current_frame).logits.flatten(-2, -1).permute(0, 2, 1))
                    if args.fusion == "cross_attention":
                        current_frame = self.bca(current_frame, long_bg, short_bg)
                        current_frame = self.bca2(current_frame, long_bg, short_bg)
                        if args.parallel_bca != 0:
                            current_frames = []
                            for bca in self.bcas:
                                current_frames.append(bca(current_frame, long_bg, short_bg))
                            current_frame = self.down_fc(torch.cat(current_frames, dim=-1))

                            current_frames = []
                            for bca in self.bcas2:
                                current_frames.append(bca(current_frame, long_bg, short_bg))
                            current_frame = self.down_fc2(torch.cat(current_frames, dim=-1))
                    else:
                        current_frame = self.fusion(torch.cat([current_frame, long_bg, short_bg], dim=-1))

                if args.decoder == "conv":
                    current_frame = current_frame.reshape(long_bg.shape[0], args.image_size//4, args.image_size//4, 384).permute(0,3,1,2)
                    current_frame = self.decoder(current_frame) + current_frame
                elif args.decoder == "transformer":
                    current_frame = self.decoder(current_frame)
                    current_frame = current_frame.reshape(long_bg.shape[0], args.image_size//4, args.image_size//4, 384).permute(0,3,1,2)
                else:
                    # print(current_frame.shape)
                    current_frame = current_frame.reshape(long_bg.shape[0], args.image_size//4, args.image_size//4, 384).permute(0,3,1,2)
                return self.head(current_frame) 
                



    


# MyModel(vits8)(torch.randn(1, 3, args.image_size, args.image_size), torch.randn(1, 3, args.image_size, args.image_size), torch.randn(1, 3, args.image_size, args.image_size)).shape


# %%
cpu_counts = args.num_workers
train_dataloader = torch.utils.data.DataLoader(MyDataset("train"), batch_size=args.batch_size, shuffle=True, num_workers=cpu_counts, persistent_workers=True, prefetch_factor=3)
val_dataloader = torch.utils.data.DataLoader(MyDataset("val"), batch_size=args.batch_size, shuffle=True, num_workers=cpu_counts, persistent_workers=True, prefetch_factor=3)


# generate a batch of train images show all 3 images in each sample side by side in a row and save it
os.makedirs("train_images_sample", exist_ok=True)
if args.show_sample:
    import pylab
    for i in range(10):
        X, Y, ROI = MyDataset("train")[i]
        current = X[:3].permute(1,2,0).numpy()
        long_bg = X[3:6].permute(1,2,0).numpy()
        short_bg = X[6:9].permute(1,2,0).numpy()
        Y = Y.numpy()
        pylab.subplot(1, 4, 1)
        pylab.imshow(current)
        pylab.title("current")
        pylab.subplot(1, 4, 2)
        pylab.imshow(long_bg)
        pylab.title("long_bg")
        pylab.subplot(1, 4, 3)
        pylab.imshow(short_bg)
        pylab.title("short_bg")
        pylab.subplot(1, 4, 4)
        pylab.imshow(Y)
        pylab.title("label")
        pylab.savefig(f"train_images_sample/{i}.png")



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
    return 1 - iou + torch.mean((pred - 0.5)**2) * args.conf_pen

def f1_loss(pred, target):
    pred = torch.sigmoid(pred)
    assert pred.shape == target.shape, f"pred shape {pred.shape} target shape {target.shape}"
    e = 1e-6
    assert len(pred.shape) == 4, f"pred shape should be (batch, 1, H, W) but get {pred.shape}"
    soft_f1 = (2 * (pred * target).sum() + e)/ (pred.sum() + target.sum() + e)
    soft_f1 = soft_f1.mean() 
    return 1 - soft_f1 + torch.mean((pred - 0.5)**2) * args.conf_pen 

# %%
import wandb
if args.backbone == "dino":
    mymodel = torch.nn.DataParallel(MyModel(vits8).cuda()) 
elif args.backbone == "segformer":
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
    backbone = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b1-finetuned-ade-512-512")
    backbone.decode_head.classifier = torch.nn.Identity()

    mymodel = torch.nn.DataParallel(MyModel(backbone).cuda())
else:
    raise NotImplementedError("I think I am lost, do you have some soup for me?")

if args.load_checkpoint:
    mymodel.load_state_dict(torch.load(args.load_checkpoint))
    print("Loaded " + args.load_checkpoint)

optimizer = torch.optim.AdamW(mymodel.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, eta_min=1e-8) 
wandb.init(resume=args.resume, config = args)

# assert iou_loss(torch.tensor([10000.0]), torch.tensor([1.0])) == 0, "iou loss not 0, but " + str(iou_loss(torch.tensor([1.0]), torch.tensor([1.0])))
# %%
if args.loss_fn == "iou":
    loss_fn = iou_loss
elif args.loss_fn == "f1":
    loss_fn = f1_loss
else:
    raise NotImplementedError("Loss function not implemented, so I am lost!")
epoch = 0



import tqdm
import sys
epoch = 0
train_running_loss = [1]
train_running_iou = [0]
train_running_f1 = [0]
wandb.log({"epoch": 0})
# wandb.watch(mymodel, log="all")

def watch(model):
    grads = [torch.tensor([0])] #it was just [0]
    params = [torch.tensor([0])]
    for param in model.parameters():
        if param.grad is not None: #why need this
            grads.append(param.grad.view(-1).cpu())
            params.append(param.view(-1).cpu())
    grads = torch.cat(grads) 
    params = torch.cat(params)
    return grads, params

    


while 1:
    for i, (X, Y, ROI) in enumerate(train_dataloader):
        if i%50 == 0:
            scheduler.step() 
            if epoch > args.num_epochs:
                sys.exit(0)
            epoch += 1
        if i%100 == 0:
            torch.save(mymodel.state_dict(),args.save_path)
            

        if i%args.print_every==0:
            with torch.no_grad():
                mymodel.eval()
                total_loss = 0
                total_pred = []
                total_Y = []
                for X_val, Y_val, ROI_val, filenames in val_dataloader:
                    # print(X_val.shape, Y_val.shape, ROI_val.shape)
                    X_val = X_val.cuda().float() 
                    Y_val = Y_val.cuda().float() * (ROI_val.cuda().float() > 0) 
                    pred = mymodel(X_val[:,:3], X_val[:,3:6], X_val[:,6:]) * (ROI_val.cuda().float().unsqueeze(1) > 0)
                    loss = loss_fn(pred, Y_val.cuda().unsqueeze(1)) 
                    total_loss += loss.item() 
                    total_pred.extend(pred.cpu().numpy().flatten())
                    total_Y.extend(Y_val.cpu().numpy().flatten())
                total_iou = (((np.array(total_pred) > 0) & (np.array(total_Y) > 0)).mean()  + 1e-6 )/(((np.array(total_pred) > 0) | (np.array(total_Y) > 0)) + 1e-6).mean()
                total_f1 = (2 * ((np.array(total_pred) > 0) & (np.array(total_Y) > 0)).sum() + 1e-6)/((np.array(total_pred) > 0).sum() + (np.array(total_Y) > 0).sum() + 1e-6)
                conf_score = torch.sigmoid(pred) 
                conf_score = conf_score[conf_score > 0.5].mean() +  (1 - conf_score[~(conf_score > 0.5)]).mean()
                conf_score = conf_score/2
                
                total_loss /= len(val_dataloader) 
                if args.normalize_image:
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                else:
                    mean = 0
                    std = 1
                
                for b in range(1): 
                    pred_ = pred[b].reshape((args.image_size//4, args.image_size//4))
                    ROI_tmp = ROI_val[b].cpu().detach().numpy()
                    ROI_green = np.stack([np.zeros_like(ROI_tmp), ROI_tmp, np.zeros_like(ROI_tmp)], axis=-1)
                    wandb.log({"val_iou": total_iou, "has_gas_ratio":(Y_val.sum((1,2)) > 0).float().sum()/len(Y_val),
                        "real": wandb.Image(Y_val[b].cpu().detach().numpy().reshape(args.image_size//4, args.image_size//4)), 
                        "pred": wandb.Image(pred_.cpu().detach().numpy()>0), 
                        "X_val": wandb.Image((X_val[b][:3].cpu().permute(1,2,0).detach().numpy() * std + mean).clip(0,1), caption=filenames[b]),
                        "X_BGS1": wandb.Image((X_val[b][3:6].cpu().permute(1,2,0).detach().numpy() * std + mean).clip(0,1)),
                        "X_BGS2": wandb.Image((X_val[b][6:9].cpu().permute(1,2,0).detach().numpy() * std + mean).clip(0,1)),
                        # "filenames": filenames[b],
                        "flow": (wandb.Image(X_val[b][9:12].cpu().permute(1,2,0).detach().numpy()) if args.use_optical_flow else None),
                        "conf_score": conf_score,
                        "F1": total_f1, 
                        "epoch": epoch,
                        "ROI": wandb.Image(ROI_green), 
                        "epoch": epoch,
                        "val_loss": total_loss}) 
                    

                    write_log(f"Epoch {epoch} Val Loss {round(total_loss, 3)} Val f1 {round(total_f1, 3)}")
                    write_log(f"Epoch {epoch} Train Loss {round(np.mean(train_running_loss), 3)} Train f1 {round(np.mean(train_running_f1), 3)}")
                    train_running_loss = []
                    train_running_iou = []
                    train_running_f1 = []

                mymodel.train()
            # zero the bar
            if args.use_tqdm:
                progress = tqdm.tqdm(range(args.print_every), ncols=50)
        if args.use_tqdm:
            progress.update(1)

            
        optimizer.zero_grad()
        X = X.cuda().float() 
        Y = Y.cuda().float() * (ROI.cuda().float() > 0)
        pred = mymodel(X[:,:3], X[:,3:6], X[:,6:]) * (ROI.cuda().float().unsqueeze(1) > 0)
        # loss = torchvision.ops.sigmoid_focal_loss(pred, Y.cuda().unsqueeze(1), alpha=1/(labels == 1).sum(), gamma=10, reduction="mean")
        loss = loss_fn(pred, Y.cuda().unsqueeze(1))
        acc = (pred > 0) == Y.cuda().unsqueeze(1)
        # f1 = f1_score(Y.unsqueeze(1).cpu().detach().numpy().reshape(-1).astype(int), pred.cpu().detach().numpy().reshape(-1) > 0)
        iou = (((pred > 0) & (Y.cuda().unsqueeze(1) > 0)).float().mean()  + 1e-6 )/(((pred > 0) | (Y.cuda().unsqueeze(1) > 0)) + 1e-6).float().mean()
        f1 = (2 * ((pred > 0) & (Y.cuda().unsqueeze(1) > 0)).float().sum() + 1e-6)/((pred > 0).float().sum() + (Y.cuda().unsqueeze(1) > 0).float().sum() + 1e-6)
        loss.backward()  
        if i%1000 == 1: 
            grads, params = watch(mymodel)
            wandb.log({"grads": wandb.Histogram(grads.cpu().detach().numpy()), "params": wandb.Histogram(params.cpu().detach().numpy())})
            write_log(f"Grad Mean: {round(grads.abs().mean().item(), 4)} Grad Std: {round(grads.std().item(), 4)} Param abs().: {round(params.abs().mean().item(), 4)} Param Std: {round(params.std().item(), 4)}") 
            # write_log(f"Grad Mean: {grads.mean()} Grad Std: {grads.std()} Param Mean: {params.mean()} Param Std: {params.std()}")  wrong
        optimizer.step() 
        wandb.log({"loss": loss.item(), "acc": acc.float().mean().item(), "iou": iou.float(), "lr": optimizer.param_groups[0]["lr"]}) # cannot do iou mean here otherwise it average non overlapping area
        # same pred as image to wandb
        # if i % 2000 == 0:

        train_running_loss.append(loss.item())
        train_running_iou.append(iou.item())
        train_running_f1.append(f1.item())
# %%
pred.shape, Y_val.unsqueeze(1).shape


