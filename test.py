import os
import sys
from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import gc
import logger
from data_loader import ImageFolder128

from test_options import parser
from utils import get_config, get_args, dump_cfg, save_imgs

args = parser.parse_args()
print(args)

if args.model=="network_origin":
    from network_origin import Generator
elif args.model=="network":
    from network import Generator

def test():
    model = Generator()
    model.eval()
    model.cuda()

    dataset = ImageFolder128(args.data_folder)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=1
    )

    for batch_idx, data in enumerate(dataloader, start=1):
        img, _ = data

        img = img.cuda()

        x = Variable(img)
        y = model(x)

        y = np.transpose(y.data, (2, 0, 1))

        number= "%03d" % batch_idx

        y = torch.cat((img[0], y), dim=2)

        save_imgs(y.data, to_size=(3, 128, 128*128),name=f"../experiments/{args.exp_name}/test_{batch_idx}.png")
        logger.debug("test_%d.png printed" %batch_idx)
        torch.cuda.empty_cache()
