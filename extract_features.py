#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) 2020 Tongzhou Wang
# Copyright (c) 2021 MSK
import argparse
import builtins
import math
import os
import random
import socket
import warnings

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models

from tqdm import tqdm

from dataset import WSINucleiDataset, TileNucleiDataset
from transforms import RandomResizedCenterCrop, BaseNuclearTransform, WSINuclearTransform
import moco.builder


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Extract Nuclear Embeddings')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--img', metavar='DIR', help='path to image folder')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--input-mode', type=str, default='wsi', choices=['tile', 'wsi'],
                    help='mode of input: '+'tile or wsi' + '(default:wsi)')
parser.add_argument('--contrast-view', type=str, default='bkg-replacement', choices=['bkg-removal', 'bkg-reserve'], 
                    help='mode of constrastive view:'+'bkg-removal or bkg-reserve'+'(default:bkg-reserve')
parser.add_argument('--save-folder', default='./results', type=str, help='saving directory')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:10001', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--gpus', default=None, nargs='+', type=int,
                    help='GPU id(s) to use. Default is all visible GPUs.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# Loss specific configs:
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=65536, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-contr-w', default=0, type=float,
                    help='contrastive weight (default: 0)')
parser.add_argument('--moco-contr-tau', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--moco-align-w', default=3, type=float,
                    help='align weight (default: 3)')
parser.add_argument('--moco-align-alpha', default=2, type=float,
                    help='alignment alpha (default: 2)')
parser.add_argument('--moco-unif-w', default=1, type=float,
                    help='uniform weight (default: 1)')
parser.add_argument('--moco-unif-t', default=3, type=float,
                    help='uniformity t (default: 3)')
parser.add_argument('--moco-unif-no-intra-batch', action='store_true',
                    help='do not use intra-batch distances in uniformity loss')

parser.add_argument('--pretrained', default='', type=str,
                    help='path to moco pretrained checkpoint')
parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')


def main():
    args = parser.parse_args()

    file_name = args.data.split("/")[-1].split(".")[0]

    args.moco_contr_tau = None

    os.makedirs(args.save_folder, exist_ok=True)
    print(f"save_folder: '{args.save_folder}'")

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    if args.gpus is None:
        args.gpus = list(range(torch.cuda.device_count()))

    if args.multiprocessing_distributed and len(args.gpus) == 1:
        args.multiprocessing_distributed = False

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if args.multiprocessing_distributed:
        # Assuming we have len(args.gpus) processes per node, we need to adjust
        # the total world_size accordingly
        args.world_size = len(args.gpus) * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=len(args.gpus), args=(args,))
    else:
        # Simply call main_worker function
        main_worker(0, args)


def main_worker(index, args):
    # We will do a bunch of `setattr`s such that
    #
    # args.rank               the global rank of this process in distributed training
    # args.index              the process index to this node
    # args.gpus               the GPU ids for this node
    # args.gpu                the default GPU id for this node
    # args.batch_size         the batch size for this process
    # args.workers            the data loader workers for this process
    # args.seed               if not None, the seed for this specific process, computed as `args.seed + args.rank`

    args.index = index
    args.gpu = args.gpus[index]
    assert args.gpu is not None
    torch.cuda.set_device(args.gpu)

    # suppress printing for all but one device per node
    if args.multiprocessing_distributed and args.index != 0:
        def print_pass(*args, **kwargs):
            pass
        builtins.print = print_pass

    print(f"Use GPU(s): {args.gpus} for training on '{socket.gethostname()}'")

    # init distributed training if needed
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            ngpus_per_node = len(args.gpus)
            # For distributed training, rank is the global rank among all
            # processes
            args.rank = args.rank * ngpus_per_node + index
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size and data
            # loader workers based on the total number of GPUs we have.
            assert args.batch_size % ngpus_per_node == 0
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    else:
        args.rank = 0


    cudnn.deterministic = True
    cudnn.benchmark = True

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = moco.builder.MoCo(
        models.__dict__[args.arch],
        args.moco_dim, args.moco_k, args.moco_m,
        contr_tau=args.moco_contr_tau,
        align_alpha=args.moco_align_alpha,
        unif_t=args.moco_unif_t,
        unif_intra_batch=not args.moco_unif_no_intra_batch,
        mlp=args.mlp)
    print(model)

    model.cuda(args.gpu)
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.multiprocessing_distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=args.gpus)

    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location=torch.device('cuda', args.gpu))

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith('module'):
                    # remove prefix
                    state_dict[k[len("module."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]
            args.start_epoch = 0
            model.load_state_dict(state_dict, strict=True)

            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            raise RuntimeError("=> no checkpoint found at '{}'".format(args.pretrained))

    train_loader = create_data_loader(args)

    # train for one epoch
    extract(train_loader, model, args)


def create_data_loader(args):
    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    augmentation = [
        transforms.ToTensor(),
        normalize,
    ]

    if args.contrast_view == 'bkg-removal':
        if args.input_mode == 'wsi':
            raise NotImplementedError()
        else:
            NuclearTransform = BaseNuclearTransform(transforms.Compose(augmentation), mode='br')
    else:
        NuclearTransform = BaseNuclearTransform(transforms.Compose(augmentation))
    if args.input_mode == 'wsi':
        NuclearTransform = WSINuclearTransform(transforms.Compose(augmentation))
        train_dataset = WSINucleiDataset(args.data, args.img, NuclearTransform)
    else:
        train_dataset = TileNucleiDataset(args.data, args.img, NuclearTransform)

    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    return train_loader


def extract(train_loader, model, args):

    model.eval()
    features = []

    for i, images in enumerate(tqdm(train_loader)):
        images = images.cuda(args.gpu, non_blocking=True)
        features.append(model(im_q=images, im_k=None).detach().cpu())

    features = torch.cat(features)
    file_name = args.data.split("/")[-1].split(".")[0]
    torch.save(features, os.path.join(args.save_folder, file_name + '.pt'))


if __name__ == '__main__':
    main()
