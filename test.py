import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.meter import AverageMeter, ProgressMeter
    
def accuracy(sims, flags, thres_step=1e-2):
    assert(thres_step > 0 and thres_step < 1)
    thres_range = np.arange(0.0, 1.0, thres_step)

    successes = np.array(
        np.sum((sims > thres) == flags)
        for thres in thres_range
    )
    
    best_idx = np.argmax(successes)
    best_thres = thres_range[best_idx]
    acc = successes[best_idx] / len(flags)

    return best_thres, acc 


def validate(val_loader, val_dataset_size, 
             net, epoch, args, 
             visualizer=None):
    batch_time = AverageMeter('Time', ':6.3f')
    progress = ProgressMeter(
        len(val_loader), [batch_time, ], prefix='Test: '
    )

    # switch to evaluate mode
    net.eval()

    pairs = [(i, j) 
        for i in range(val_dataset_size) 
        for j in range(val_dataset_size)
        if i < j
    ]

    feats = []
    # fliped_feats = []
    targets = []
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            
            target = target.cuda(args.gpu, non_blocking=True)
            
            fliped_images = images.flip(-1) 
            
            # compute output
            feat = net(images)
            # fliped_feat = net(fliped_images)

            feats.append(feat)
            # fliped_feats.append(fliped_feat)
            targets.append(target) 

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)        

    feats = torch.cat(feats, dim=0)
    # fliped_feats = torch.cat(fliped_feats, dim=0)
    flags = np.array([ 
        targets[i] == targets[j] 
        for (i, j) in pairs
    ], dtype=int)

    assert(len(pairs) == len(flags))
    sims = np.array([
        F.cosine_similarity(feats[i: i+1], feats[j, j+1]).item()
        for (i, j) in pairs
    ])

    # measure accuracy and record loss
    thres, acc = accuracy(sims, flags, thres_step=5e-3)

    # TODO: this should also be done with the ProgressMeter
    print(f' * Threshold {thres:4.2f} Acc {acc*100:4.2f}')
                
    return thres, acc

