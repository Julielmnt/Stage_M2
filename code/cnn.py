#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import torch.optim as optim
import matplotlib.pyplot as plt
import math
import os
import time
import torch.nn.functional as F
import wandb
import data_analysis
from data_analysis import Simulation
from data_analysis import compatible_path

import utils
from utils import get_freer_gpu
from utils import info_text
import argparse
from autoencoder import SimuDataset
from autoencoder import training


class CNN_debug(nn.Module):

    def __init__(
        self,
        device,
        sizes,
        n_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=True,
    ):
        super().__init__()
        # shape : B * 3 * 81 * 51
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.device = device
        self.n_channels = n_channels
        self.sizes = sizes
        self.N = n_channels * sizes[-1][0] * sizes[-1][1]

        self.encoder = nn.Sequential(
            self.conv_block(3, self.n_channels),
            nn.Upsample(size=sizes[0], mode="bilinear", align_corners=False),
            self.conv_block(self.n_channels, self.n_channels),
            nn.Upsample(size=sizes[1], mode="bilinear", align_corners=False),
            self.conv_block(self.n_channels, self.n_channels),
            nn.Upsample(size=sizes[2], mode="bilinear", align_corners=False),
            self.conv_block(self.n_channels, self.n_channels),
            nn.Upsample(size=sizes[3], mode="bilinear", align_corners=False),
            self.conv_block(self.n_channels, self.n_channels),
        ).to(device)

        self.decoder = nn.Sequential(
            self.conv_block(self.n_channels, self.n_channels),
            nn.Upsample(size=sizes[2], mode="bilinear", align_corners=False),
            self.conv_block(self.n_channels, self.n_channels),
            nn.Upsample(size=sizes[1], mode="bilinear", align_corners=False),
            self.conv_block(self.n_channels, self.n_channels),
            nn.Upsample(size=sizes[0], mode="bilinear", align_corners=False),
            self.conv_block(self.n_channels, self.n_channels),
            nn.Upsample(size=(81, 51), mode="bilinear", align_corners=False),
            self.conv_block(self.n_channels, 3),
        ).to(device)

    def conv_block(self, ch_in, ch_out):
        return nn.Sequential(
            nn.Conv2d(
                ch_in,
                ch_out,
                self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                bias=self.bias,
            ),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(),
            nn.Conv2d(
                ch_out,
                ch_out,
                self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                bias=self.bias,
            ),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded


class CNN(nn.Module):

    def __init__(
        self,
        device,
        sizes,
        n_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=True,
    ):
        super().__init__()
        # shape : B * 3 * 81 * 51
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.device = device
        self.n_channels = n_channels
        self.sizes = sizes
        self.N = n_channels * sizes[-1][0] * sizes[-1][1]

        self.encoder = nn.Sequential(
            self.conv_block(3, self.n_channels),
            nn.Upsample(size=sizes[0], mode="bilinear", align_corners=False),
            self.conv_block(self.n_channels, self.n_channels),
            nn.Upsample(size=sizes[1], mode="bilinear", align_corners=False),
            self.conv_block(self.n_channels, self.n_channels),
            nn.Upsample(size=sizes[2], mode="bilinear", align_corners=False),
            self.conv_block(self.n_channels, self.n_channels),
            nn.Upsample(size=sizes[3], mode="bilinear", align_corners=False),
            self.conv_block(self.n_channels, self.n_channels),
        ).to(device)

        self.decoder = nn.Sequential(
            self.conv_block(self.n_channels, self.n_channels),
            nn.Upsample(size=sizes[2], mode="bilinear", align_corners=False),
            self.conv_block(self.n_channels, self.n_channels),
            nn.Upsample(size=sizes[1], mode="bilinear", align_corners=False),
            self.conv_block(self.n_channels, self.n_channels),
            nn.Upsample(size=sizes[0], mode="bilinear", align_corners=False),
            self.conv_block(self.n_channels, self.n_channels),
            nn.Upsample(size=(81, 51), mode="bilinear", align_corners=False),
            nn.ConvTranspose2d(
                n_channels, 3, kernel_size, stride=stride, padding=padding, bias=bias
            ),
        ).to(device)

    def conv_block(self, ch_in, ch_out):
        return nn.Sequential(
            nn.Conv2d(
                ch_in,
                ch_out,
                self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                bias=self.bias,
            ),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(),
            nn.Conv2d(
                ch_out,
                ch_out,
                self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                bias=self.bias,
            ),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded


class CNN_sizes(nn.Module):

    def __init__(
        self,
        device,
        sizes,
        n_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=True,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.device = device
        self.n_channels = n_channels
        self.sizes = sizes
        self.N = n_channels * sizes[-1][0] * sizes[-1][1]

        self.encoder = nn.Sequential(
            self.conv_block(3, self.n_channels),
            nn.Upsample(size=sizes[0], mode="bilinear", align_corners=False),
            self.conv_block(self.n_channels, self.n_channels),
            nn.Upsample(size=sizes[1], mode="bilinear", align_corners=False),
            self.conv_block(self.n_channels, self.n_channels),
            nn.Upsample(size=sizes[2], mode="bilinear", align_corners=False),
            self.conv_block(self.n_channels, self.n_channels),
            nn.Upsample(size=sizes[3], mode="bilinear", align_corners=False),
            self.conv_block(self.n_channels, self.n_channels),
            nn.Upsample(size=sizes[4], mode="bilinear", align_corners=False),
            # self.conv_block(self.n_channels, self.n_channels),
            # nn.Upsample(size=sizes[5], mode='bilinear', align_corners=False),
            self.conv_block(self.n_channels, self.n_channels),
        ).to(device)

        self.decoder = nn.Sequential(
            # self.conv_block(self.n_channels, self.n_channels),
            # nn.Upsample(size=sizes[4], mode='bilinear', align_corners=False),
            self.conv_block(self.n_channels, self.n_channels),
            nn.Upsample(size=sizes[3], mode="bilinear", align_corners=False),
            self.conv_block(self.n_channels, self.n_channels),
            nn.Upsample(size=sizes[2], mode="bilinear", align_corners=False),
            self.conv_block(self.n_channels, self.n_channels),
            nn.Upsample(size=sizes[1], mode="bilinear", align_corners=False),
            self.conv_block(self.n_channels, self.n_channels),
            nn.Upsample(size=sizes[0], mode="bilinear", align_corners=False),
            self.conv_block(self.n_channels, self.n_channels),
            nn.Upsample(size=(81, 51), mode="bilinear", align_corners=False),
            nn.ConvTranspose2d(
                n_channels, 3, kernel_size, stride=stride, padding=padding, bias=bias
            ),
        ).to(device)

    def conv_block(self, ch_in, ch_out):
        return nn.Sequential(
            nn.Conv2d(
                ch_in,
                ch_out,
                self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                bias=self.bias,
            ),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(),
            nn.Conv2d(
                ch_out,
                ch_out,
                self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                bias=self.bias,
            ),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded


class CNN_pandeylike(nn.Module):

    def __init__(
        self, device, sizes, n_channels=8, kernel_size=3, stride=1, padding=1, bias=True, convnext=False, pandey_data=False, h=81, l=51,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        print(f"kernel size inside algo {kernel_size}")
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.device = device
        self.n_channels = n_channels
        self.sizes = sizes
        self.N = n_channels * sizes[-1][0] * sizes[-1][1]
        self.convnext = convnext
        self.pandey_data = pandey_data

        if self.pandey_data:
            ch_in_first = 1
        else:
            ch_in_first = 3

        if self.convnext:    
            convBlock = convnext_block
        else:   
            convBlock = conv_block

        self.encoder = nn.Sequential(
            convBlock(ch_in_first, self.n_channels * 16, device=device, kernel_size=kernel_size),
            nn.Upsample(size=sizes[0], mode="bilinear", align_corners=False),
            convBlock(self.n_channels * 16, self.n_channels * 8, device=device, kernel_size=kernel_size),
            nn.Upsample(size=sizes[1], mode="bilinear", align_corners=False),
            convBlock(self.n_channels * 8, self.n_channels * 8, device=device, kernel_size=kernel_size),
            nn.Upsample(size=sizes[2], mode="bilinear", align_corners=False),
            convBlock(self.n_channels * 8, self.n_channels * 4, device=device, kernel_size=kernel_size),
            nn.Upsample(size=sizes[3], mode="bilinear", align_corners=False),
            convBlock(self.n_channels * 4, self.n_channels * 4, device=device, kernel_size=kernel_size),
            nn.Upsample(size=sizes[4], mode="bilinear", align_corners=False),
            convBlock(self.n_channels * 4, self.n_channels, device=device, kernel_size=kernel_size),
            nn.Upsample(size=sizes[5], mode="bilinear", align_corners=False),
            convBlock(self.n_channels, self.n_channels, device=device, kernel_size=kernel_size),
        ).to(device)

        self.decoder = nn.Sequential(
            convBlock(self.n_channels, self.n_channels * 4, device=device, kernel_size=kernel_size),
            nn.Upsample(size=sizes[4], mode="bilinear", align_corners=False),
            convBlock(self.n_channels * 4, self.n_channels * 4, device=device, kernel_size=kernel_size),
            nn.Upsample(size=sizes[3], mode="bilinear", align_corners=False),
            convBlock(self.n_channels * 4, self.n_channels * 8, device=device, kernel_size=kernel_size),
            nn.Upsample(size=sizes[2], mode="bilinear", align_corners=False),
            convBlock(self.n_channels * 8, self.n_channels * 8, device=device, kernel_size=kernel_size),
            nn.Upsample(size=sizes[1], mode="bilinear", align_corners=False),
            convBlock(self.n_channels * 8, self.n_channels * 16, device=device, kernel_size=kernel_size),
            nn.Upsample(size=sizes[0], mode="bilinear", align_corners=False),
            convBlock(self.n_channels * 16, self.n_channels * 16, device=device, kernel_size=kernel_size),
            nn.Upsample(size=(h, l), mode="bilinear", align_corners=False),
            nn.ConvTranspose2d(
                n_channels * 16, ch_in_first, kernel_size=3, stride=stride, padding=padding, bias=bias
            ),
        ).to(device)

    def forward(self, x):
        # print(f"shape of x {x.shape}")
        # if self.pandey_data:
            # x = torch.unsqueeze(x, 1)
            # print(f"shape of x after unsqueezing {x.shape}")
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        # if self.pandey_data:
            # decoded = x = torch.squeeze(x, 1)
        return decoded

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out, device, kernel_size=3, stride=1, padding=1, bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias    
        self.ch_in = ch_in  
        self.ch_out = ch_out

        self.block = nn.Sequential(
            nn.Conv2d(
                self.ch_in,
                self.ch_out,
                self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                bias=self.bias,
            ),
            nn.BatchNorm2d(self.ch_out),
            nn.ReLU(),
            nn.Conv2d(
                self.ch_out,
                self.ch_out,
                self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                bias=self.bias,
            ),
            nn.BatchNorm2d(self.ch_out),
            nn.ReLU(),
        ).to(device)

    def forward(self, x):
        out = self.block(x)
        return out

class convnext_block(nn.Module):
    def __init__(self, ch_in, ch_out, device, stride=1, padding=1, bias=True, skip=False, kernel_size=7):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.stride = stride
        # self.padding = padding
        self.bias = bias
        self.skip = skip
        self.kernel_size = kernel_size  
        self.padding = kernel_size // 2

        self.block = nn.Sequential(
                nn.Conv2d(
                    self.ch_in,
                    self.ch_in,
                    self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    bias=self.bias,
                    groups=self.ch_in,
                ),
                # nn.LayerNorm(self.ch_in),
                nn.BatchNorm2d(self.ch_in),
                nn.Conv2d(
                    self.ch_in,
                    self.ch_in*2,
                    1,
                    stride=self.stride,
                    padding=0,
                    bias=self.bias,
                ),
                nn.GELU(),
                nn.Conv2d(
                    self.ch_in*2,
                    self.ch_out,
                    1,
                    stride=self.stride,
                    padding=0,
                    bias=self.bias,
                ),).to(device)
        

    #     for layer in self.block:
    #             if isinstance(layer, nn.Conv2d):
    #                 layer.register_forward_hook(self.print_tensor_shape)

    # def print_tensor_shape(self, module, input, output):
    #     print(f"After {module}: {output.shape}")

    def forward(self, x):
        out = self.block(x)

        if self.skip:
            out = nn.Sequential(out, nn.Identity())

        return out





def main(
    n_channels=8,
    batch_size=4,
    lr=1e-4,
    debug=False,
    pandey=False,
    Lambda=1e-2,
    num_epochs=100,
    wandb_on=False,
    Ra=1e8,
    training_mode="num_epochs",
    group="run",
    convnext= False,
    kernel_size=3,
    convnext_kernel=7,
    pandey_data=False,
    dynamics=None
):
    """Main exec function

    Keyword Arguments:
        n_channels -- number of channels (default: {64})
        batch_size -- size of the batch (default: {4})
    """
    device = get_freer_gpu() if torch.cuda.is_available() else "cpu"
    print(f"wandb_on {wandb_on}")
    print(f"convnext {convnext}")
    print(f"pandey_data = {pandey_data}")
    # Dataloading
    current_directory = compatible_path("../")

    directory = f"{current_directory}/results/cnn/"

    if pandey:
        directory += "pandeylike/"

    if convnext:
        directory += "convnext/"

    if Ra == 1e8 and not pandey_data and dynamics==None:
        directory += "Ra1e8/"

    if Ra == 1e7 and not pandey_data:
        directory += "Ra1e7/"

    if Ra == 2e6 and not pandey_data:
        directory += "Ra2e6/"   

    if pandey_data:
        directory += "pandey_data/"

    if convnext_kernel != 7:
        directory += f"kernel{convnext_kernel}/"

    
    if dynamics is not None:
        if dynamics == "HC":
            directory += "/HCRa1e8"
        if dynamics == "pureRB":
            directory += "pureRB"
        group = dynamics

    if not os.path.isdir(directory):
        raise FileNotFoundError(f"The directory {directory} does not exist.")


    print(f"saving NN at {directory}")
    filename = f"model_bs{batch_size}_nc{n_channels}_lr{lr}"

    if pandey_data:
        simulation = Simulation(current_directory, normalize=True, Lambda=Lambda, Ra=Ra, pandey=pandey_data)
        simulation.import_data_pandey(mean_type="scaled")

    else:
        simulation = Simulation(current_directory, normalize=True, Lambda=Lambda, Ra=Ra)
        time_array, x, z, u, w, T, umean, wmean, Tmean = map(lambda x: torch.tensor(x).to(device), simulation.import_data(mean_type="scaled"))
        simulation.image_rgb()

    h, l = simulation.h, simulation.l
    N = h * l * 3

    print("ABOUT DATA")
    print(f"Lambda = {Lambda}")
    print(f"Ra = {Ra}")
    print(f"number of snapshots : {simulation.m}")
    print(f"(h,l) = {h,l}")

    # Setting  Dataset
    batch_size = bs
    training_ratio = 0.9
    print(f"BATCHSIZE = {batch_size}")
    dataset = SimuDataset(simulation, device, rgb=True, training_ratio=training_ratio, pandey_data=pandey_data)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    valset = SimuDataset(
        simulation, device, rgb=True, mode="val", training_ratio=training_ratio, pandey_data=pandey_data,
    )
    valloader = DataLoader(dataset=valset, batch_size=4, shuffle=True)

    if convnext:
        kernel_size = convnext_kernel

    #wandb
    print(f"wandb_on : {wandb_on}")
    if wandb_on:
        import wandb
        name = filename
        wandb.init(
            project="CNN_alone",
            config={
                "Lambda": Lambda,
                "Ra": Ra,
                "batch_size ": batch_size,
                "kernel_size": kernel_size,
                "lr": lr,
                "training_mode": training_mode,
                "training_ratio": training_ratio,
                "directory": directory,
                "Ra_power": simulation.Ra_power,
                "n_channels": n_channels,
                "pandey": pandey,
                "convnext": convnext,
                "group": group,
                "pandey_data": pandey_data
            },
            name=name,
            group=group,
        )

    # Model
    weight_decay = 0
    print("MODEL PARAMETERS")
    print(f"n_channels = {n_channels}")
    print(f"batch_size = {bs}")
    print(f"lr = {lr}")
    
    # print(f"debug = {debug}")

    size1 = (50, 30)
    size2 = (25, 15)
    size3 = (13, 8)
    size4 = (7, 4)
    size5 = (4, 2)
    size6 = (2, 1)

    # sizes = [size1, size2, size3, size4]
    # sizes = [size1, size2, size3, size4, size5, size6]
    # sizes = [size1, size2, size3, size4, size5]
    sizes = [size1, size2, size3, size4, size5, size6]
    print(f"sizes : {sizes}")

    # if debug == True:
    #     model = CNN_debug(device=device, n_channels=n_channels, sizes=sizes)

    # else:
    #     model = CNN_sizes(device=device, n_channels=n_channels, sizes=sizes)
    if convnext:
        kernel_size = convnext_kernel
    
    print(f"kernel size = {kernel_size}")
    print(f"padding {kernel_size // 2}")

    if pandey:
        model = CNN_pandeylike(device=device, n_channels=8, sizes=sizes, convnext=convnext, kernel_size=kernel_size, pandey_data=pandey_data, h=h, l=l)

    for param in model.parameters():
        assert param.requires_grad

    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=int(num_epochs * 0.8)
    )



    # Training
    model, info, execution_time = training(
            dataloader,
            valloader,
            model,
            criterion,
            optimizer,
            device,
            scheduler=scheduler,
            num_epoch=num_epochs,
            noisy=False,
            wandb_on=wandb_on,
            directory = directory, 
            filename = filename,
        )

    # Saving
    torch.save(model.state_dict(), directory + filename + ".pt")
    print("model saved !")
    info_text(
        directory,
        batch_size,
        info,
        title=filename + "_info.txt",
        Lambda=Lambda,
        Ra=Ra,
        lr=lr,
        kernel_size=kernel_size,
        n_channels=n_channels,
        weight_decay=weight_decay,
        execution_time_in_min=execution_time, 
        sizes=sizes,
        group=str(group),
        wandb_on=wandb_on,
        convnext=convnext,
        pandey=pandey

    )
    if wandb_on:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script with parameter bs and n_channels"
    )
    parser.add_argument(
        "--bs", type=int, default=4, help="Value for batchsize (default: 4)"
    )
    parser.add_argument(
        "--n_channels", type=int, default=8, help="Value for n_channels (default: 8)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Value for learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Booleen controlling version of code (default: False)",
    )
    parser.add_argument(
        "--pandey",
        action="store_true",
        help="Booleen controlling version of code (default: False)",
    )
    parser.add_argument(
        "--Ra", type=float, default=1e8, help="Value for Ra (default: 1e8)"
    )
    
    parser.add_argument(
        "--wandb_on", action="store_true", help="wandbing (default: False)")
    
    parser.add_argument(
        "--group", type=str, default="run", help="group in which to save the model in wandb (default: run)")
    
    parser.add_argument(
        "--Lambda", type=float, default=1e-2, help="Value for Lambda (default: 1e-2)")
    
    parser.add_argument(
        "--convnext", action="store_true", help="Value for convnext (default: False)")
    
    parser.add_argument(
        "--convnext_kernel", type=int, default=7, help="Size of convnext kernel(default: 7)")
    
    parser.add_argument(
        "--pandey_data", action="store_true", help="Value for pandey data (default: False)")
    
    parser.add_argument(
        "--dynamics", type=str, default=None, help="Value for dynamics (default: None)")
    
    args = parser.parse_args()
    n_channels = args.n_channels
    bs = args.bs
    lr = args.lr
    debug = args.debug
    pandey = args.pandey
    Ra = args.Ra
    group = args.group
    wandb_on = args.wandb_on
    Lambda = args.Lambda
    convnext = args.convnext
    convnext_kernel = args.convnext_kernel
    pandey_data = args.pandey_data
    dynamics = args.dynamics

    main(n_channels=n_channels, Lambda=Lambda, batch_size=bs, lr=lr, debug=debug, pandey=pandey, Ra=Ra, group=group, wandb_on=wandb_on, convnext=convnext, convnext_kernel=convnext_kernel, pandey_data=pandey_data, dynamics=dynamics)
