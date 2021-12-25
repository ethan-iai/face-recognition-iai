import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from functools import partial

from backbone.cbam import CBAMResNet
from backbone.attention import ResidualAttentionNet_56, ResidualAttentionNet_92

from metrics.InnerProduct import InnerProduct
from metrics.ArcMarginProduct import ArcMarginProduct
from metrics.MultiMarginProduct import MultiMarginProduct
from metrics.CosineMarginProduct import CosineMarginProduct
from metrics.SphereMarginProduct import SphereMarginProduct

from utils.meter import AverageMeter, ProgressMeter, Meter
from utils.augmentation import RandAugment
# TODO: implement visualizer
# from utils.visualize import Visualizer

from test import validate

backbones = {
    'Res50_IR'      :   partial(CBAMResNet, num_layers=50, mode='ir'),
    'SERes50_IR'    :   partial(CBAMResNet, num_layers=50, mode='ir_se'),
    'Res100_IR'     :   partial(CBAMResNet, num_layers=100, mode='ir'),
    'SERes100_IR'   :   partial(CBAMResNet, num_layers=100, mode='ir_se'),
    'Res152_IR'     :   partial(CBAMResNet, num_layers=152, mode='ir'),
    'SERes152_IR'   :   partial(CBAMResNet, num_layers=152, mode='ir_se'),
    'Attention_56'  :   ResidualAttentionNet_56,
    'Attention_92'  :   ResidualAttentionNet_92,
}

metircs = {
    'ArcFace'       :   ArcMarginProduct,
    'MultiMargin'   :   MultiMarginProduct,
    'CosFace'       :   CosineMarginProduct,
    'Softmax'       :   InnerProduct, 
    'SphereFace'    :   SphereMarginProduct,
}

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--backbone', metavar='BACKBONE', default='Res50_IR',
                    choices=backbones.keys(),
                    help='model architecture: ' +
                        ' | '.join(backbones.keys()) +
                        ' (default: resnet50)')
parser.add_argument('--metric', type=str, default='ArcFace', 
                    choices=metircs.keys(),
                    help='ArcFace, CosFace, SphereFace, MultiMargin, Softmax')
parser.add_argument('--feature-dim', type=int, default=512, 
                    help='feature dimension, 128 or 512')
parser.add_argument('-m', '--margin', type=float, default=0.5,
                    help='margin m in Arc face')
parser.add_argument('-s', '--scale', type=float, default=32.0,
                    help='scale s in Arcface')
parser.add_argument('--n-classes', type=int, default=1000,
                    help='number of classes')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
# parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
#                     metavar='W', help='weight decay (default: 1e-4)',
#                     dest='weight_decay')
parser.add_argument('--save-dir', type=str, default='./model', 
                    help='directory to save model checkpoint')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--save-freq', default=1000, type=int,
                    metavar='N', help='save checkpoints frequency (default: 1000)')
parser.add_argument('--resume', type=str, default='', metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
# parser.add_argument('--pretrained', dest='pretrained', action='store_true',
#                     help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

best_acc1 = 0
best_thres = 0

def main():
    args = parser.parse_args()

    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    global best_thres
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    
    # create model
    print("=> creating backbone '{}'".format(args.backbone))
    net = backbones[args.backbone](feature_dim=args.feature_dim)
    
    # create metric
    print("=> creatinig metric '{}'".format(args.metric))
    kwargs = {
        's' :   args.scale,
        'm' :   args.margin
    }
    metric = metircs[args.metric](in_feature=args.feature_dim, 
                                  out_feature=args.n_classes, 
                                  **kwargs)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            net.cuda(args.gpu)
            metric.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu])
            metric = torch.nn.parallel.DistributedDataParallel(metric, device_ids=[args.gpu])
        else:
            net.cuda()
            metric.cuda(args.gpu)
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            net = torch.nn.parallel.DistributedDataParallel(net)
            metric = torch.nn.parallel.DistributedDataParallel(metric, device_ids=[args.gpu])
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        net = net.cuda(args.gpu)
        metric = metric.cuda(args.gpu)
    else:
        net = torch.nn.DataParallel(net).cuda()
        metric = torch.nn.DataParallel(metric).cuda()
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD([
        {'params': net.parameters(), 'weight_decay': 5e-4},
        {'params': metric.parameters(), 'weight_decay': 5e-4}
    ], args.lr, momentum=args.momentum, nesterov=True)

    # NOTE: set lr decays here
    exp_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[400, 800, 1200], gamma=0.1)


    # optionally resume from a checkpoint
    if len(args.resume) > 0:
        if os.path.isfile(args.resume):
            print("==> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=torch.device(args.gpu))
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            # best_thres = checkpoint['best_thres']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            net.load_state_dict(checkpoint['net_state_dict'], strict=False)
            
            if not args.evaluate:
                metric.load_state_dict(checkpoint['margin_state_dict'], strict=False)
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("==> loaded checkpoint '{}' (epoch {})"
                    .format(args.resume, checkpoint['epoch']))
        else:
            print("==> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    
    # TODO: add data properties
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])

    original_train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.Resize(128),
            transforms.RandomResizedCrop(112),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    augmented_train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.Resize(128),
            transforms.RandomResizedCrop(112),
            RandAugment(n=2, m=10),
            transforms.ToTensor(),
            normalize
        ])
    )

    train_dataset = torch.utils.data.ConcatDataset([
        original_train_dataset, augmented_train_dataset
    ])

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    # FIXME: centrecrop may corp some useful info 
    val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(128),
            transforms.CenterCrop(112),
            transforms.ToTensor(),
            normalize,
    ]))
    val_dataset_size = len(val_dataset)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, val_dataset_size, net, 
                 args.start_epoch, args, visualizer=None)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, net, metric, criterion, 
              optimizer, exp_lr_scheduler, epoch, args)

        # adjust_learning_rate
        exp_lr_scheduler.step()

        # evaluate on validation set
        thres, acc1 = validate(val_loader, val_dataset, net, 
                               epoch, args, visualizer=None)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        best_thres = thres if is_best else best_thres

        if ((not args.multiprocessing_distributed or 
            (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0)) and
            epoch % args.save_freq == 0):
            filename = f"checkpoint-{epoch}.pth.tar"
            save_checkpoint({
                'epoch'             :   epoch + 1,
                'backbone'          :   args.backbone,
                'metric'            :   args.metric,
                'net_state_dict'    :   net.state_dict(),
                'margin_state_dict' :   metric.state_dict(),
                'best_acc1'         :   best_acc1,
                'best_thres'        :   best_thres,
                'optimizer'         :   optimizer.state_dict(),
            }, is_best, args.save_dir, filename)


def train(train_loader, net, metric, criterion, 
          optimizer, lr_scheduler, epoch, args, visualizer=None):
    batch_time = AverageMeter('Time', ':5.3f')
    data_time = AverageMeter('Data', ':5.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.02f')
    top5 = AverageMeter('Acc@5', ':6.02f')
    learning_rate = Meter('learnig rate', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, learning_rate, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    net.train()
    metric.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        raw_logits = net(images)
        output = metric(raw_logits, target)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        learning_rate.update(lr_scheduler.get_lr()[0])

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % args.print_freq == 0:
            progress.display(i + 1)

            if visualizer is not None:
                visualizer.plot_curves({'softmax loss': loss.item()}, 
                                iters=(epoch * args.batch_size + i), 
                                title='train loss', xlabel='iters', ylabel='train loss')
                visualizer.plot_curves({'train accuracy': acc1[0]}, 
                                iters=(epoch * args.batch_size + i), 
                                title='Acc@1', xlabel='iters', ylabel='Acc@1')


def save_checkpoint(state, is_best, save_dir='', filename='checkpoint.pth.tar'):
    file_path = os.path.join(save_dir, filename)
    torch.save(state, file_path)
    if is_best:
        shutil.copyfile(
            file_path, 
            os.path.join(save_dir, 'model_best.pth.tar')
        )


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()