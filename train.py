import os
import sys
from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import logger
from data_loader import ImageFolder128
from utils import save_imgs

from train_options import parser

args = parser.parse_args()
print(args)

if args.model=="network_origin":
    from network_origin import Generator
elif args.model=="network":
    from network import Generator

def train():
    # train-related code
    patch_height=128
    patch_width=128
    model = Generator()
    model=model.cuda()
    model.train()
    print(model)

    dataset = ImageFolder128(args.data_folder)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=1
    )

    print("Data loaded")

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_criterion = nn.MSELoss()

    avg_loss, epoch_avg = 0.0, 0.0

    for epoch_idx in range(args.start_epoch,args.start_epoch+args.num_epochs + 1):

        for batch_idx, data in enumerate(dataloader, start=1):
            img, _ = data

            img = img.cuda()

            loss_per_image = 0.0

            optimizer.zero_grad()
            #print(patches.shape)
            x = Variable(img)
            y = model(x)
            loss = loss_criterion(y, x)

            loss_per_image += loss.item()

            loss.backward()
            optimizer.step()

            avg_loss += loss_per_image
            epoch_avg += loss_per_image

            if batch_idx % 20 ==0:
                logger.debug(
                '[%3d/%3d][%5d/%5d] avg_loss: %.8f' %
                (epoch_idx, args.start_epoch+args.num_epochs, batch_idx, len(dataloader), avg_loss / args.batch_size)
            )
            avg_loss = 0.0

        if epoch_idx % 5 == 0:
            epoch_avg = epoch_avg/(len(dataloader) * 1)/5

            logger.debug("Epoch avg = %.8f" % epoch_avg)
            epoch_avg = 0.0
            torch.save(model.state_dict(), f"./experiments/{args.exp_name}/chkpt/model_{epoch_idx}.pth")
    torch.save(model.state_dict(), f"./experiments/{args.exp_name}/model_final.pth")
    
    
train()import os
import sys
from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import logger
from data_loader import ImageFolder
from utils import save_imgs

from train_options import parser

args = parser.parse_args()
print(args)

if args.model=="network_origin":
    from network_origin import Generator
elif args.model=="network":
    from network import Generator

def train():
    # train-related code
    model = Generator()
    model.cuda()
    model.train()
    print(model)

    train_transform = transforms.Compose([
        transforms.RandomCrop((128, 128)),
        transforms.ToTensor(),
    ])

    train_set = dataset.ImageFolder(root=args.data_folder, transform=train_transform)

    train_loader = data.DataLoader(
        dataset=train_set, batch_size=args.batch_size, shuffle=True, num_workers=1)

    print('total images: {}; total batches: {}'.format(
        len(train_set), len(train_loader)))


    print("Data loaded")

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_criterion = nn.MSELoss()

    avg_loss, epoch_avg = 0.0, 0.0

    for epoch_idx in range(args.start_epoch,args.start_epoch+args.num_epochs + 1):

        for batch_idx, data in enumerate(dataloader, start=1):
            img, _ = data

            img = img.cuda()

            loss_per_image = 0.0

            optimizer.zero_grad()
            #print(patches.shape)
            x = Variable(img)
            y = model(x)
            loss = loss_criterion(y, x)
            
            # add regularization
            l1=0
            for name, param in model.named_parameters():
                if name=='de_g.bias' or name=='de_g.weight':
                    alpha=0.01
                    print(param.shape)
                    x = torch.ceil(32*param+0.5)/32
                    num=torch.log(torch.sum(torch.abs(x)))
                    print(num)
                    den=torch.log(torch.tensor(10).float())
                    l1+=num/den
                    regularization=alpha * l1 / (16 * 14 * 14)


            loss_per_image += loss.item()+Variable(regularization)

            loss.backward()
            optimizer.step()

            avg_loss += loss_per_image
            epoch_avg += loss_per_image

            if batch_idx % 20 ==0:
                logger.debug(
                '[%3d/%3d][%5d/%5d] avg_loss: %.8f' %
                (epoch_idx, args.start_epoch+args.num_epochs, batch_idx, len(dataloader), avg_loss / args.batch_size)
            )
            avg_loss = 0.0

        if epoch_idx % 5 == 0:
            epoch_avg = epoch_avg/(len(dataloader) * 1)/5

            logger.debug("Epoch avg = %.8f" % epoch_avg)
            epoch_avg = 0.0
            torch.save(model.state_dict(), f"./experiments/{args.exp_name}/chkpt/model_{epoch_idx}.pth")
    torch.save(model.state_dict(), f"./experiments/{args.exp_name}/model_final.pth")
    
    
train()