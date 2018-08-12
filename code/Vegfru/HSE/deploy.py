import argparse
import os,sys
import pdb
sys.path.insert(0,'.')
import shutil
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from dataset import VegfruDataset
from ResNetEmbed import ResNetEmbed


def arg_parse():
    parser = argparse.ArgumentParser(description='PyTorch HSE Deployment')
    parser.add_argument('data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('testlist', metavar='DIR',
                        help='path to test list')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--snapshot', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--crop_size', dest='crop_size',default=224, type=int, 
                        help='crop size')
    parser.add_argument('--scale_size', dest = 'scale_size',default=448, type=int, 
                        help='the size of the rescale image')
    args = parser.parse_args()
    return args

def print_args(args):
    print "=========================================="
    print "==========       CONFIG      ============="
    print "=========================================="
    for arg,content in args.__dict__.items():
        print "{}:{}".format(arg,content)
    print "\n"

def main():
    args = arg_parse()
    print_args(args)

    # Create dataloader
    print "==> Creating dataloader..."
    data_dir = args.data
    test_list = args.testlist
    test_loader = get_test_set(data_dir, test_list, args)
    classes_dict = {'sup': 25, 'sub': 292}

    # load the network
    print "==> Loading the network ..."
    model = ResNetEmbed(cdict=classes_dict)

    model.cuda()

    if args.snapshot:
        if os.path.isfile(args.snapshot):
            print("=> loading checkpoint '{}'".format(args.snapshot))
            checkpoint = torch.load(args.snapshot)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}'"
                  .format(args.snapshot))
        else:
            print("=> no checkpoint found at '{}'".format(args.snapshot))
            exit()

    cudnn.benchmark = True

    print "Testing..."
    # with torch.no_grad():
    validate(test_loader, model, args)
    
def validate(val_loader, model, args):
    batch_time = AverageMeter()
    top1_prior = AverageMeter()
    top5_prior = AverageMeter()
    top1_inferior = AverageMeter()
    top5_inferior = AverageMeter()
    
    # switch to evaluate mode
    model.eval() 

    end = time.time()
    for i, (input, gt_sup, gt_sub) in enumerate(val_loader):
        gt_sup = gt_sup.cuda(async=True)
        gt_sub = gt_sub.cuda(async=True)

        input_var = torch.autograd.Variable(input, volatile=True).cuda()
        
        # compute output
        pred_prior,prec1_inferior = model(input_var)
     
        # measure accuracy and record loss
        prec1_prior, prec5_prior = accuracy(pred_prior.data, gt_sup, topk=(1, 5))
        prec1_inferior, prec5_inferior = accuracy(prec1_inferior.data, gt_sub, topk=(1, 5))

        top1_prior.update(prec1_prior[0], input.size(0))
        top5_prior.update(prec5_prior[0], input.size(0))
        top1_inferior.update(prec1_inferior[0], input.size(0))
        top5_inferior.update(prec5_inferior[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print(' * L1(superclass): \tPrec@1 {top1_prior.avg:.3f} Prec@5 {top5_prior.avg:.3f}'
          .format(top1_prior=top1_prior, top5_prior=top5_prior))
    print(' * L2(subclass): \tPrec@1 {top1_inferior.avg:.3f} Prec@5 {top5_inferior.avg:.3f}'
          .format(top1_inferior=top1_inferior, top5_inferior=top5_inferior))

    return top1_prior.avg, top1_inferior.avg

def get_test_set(data_dir,test_list,args):
    # Data loading code
    # normalize for different pretrain model:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    crop_size = args.crop_size
    scale_size = args.scale_size
    # center crop
    test_data_transform = transforms.Compose([
          transforms.Resize((scale_size,scale_size)),
          transforms.CenterCrop(crop_size),
          transforms.ToTensor(),
          normalize,
      ])

    test_set = VegfruDataset(data_dir, test_list, test_data_transform)
    test_loader = DataLoader(dataset=test_set, num_workers=args.workers,batch_size=args.batch_size, shuffle=False)

    return test_loader

class AverageMeter(object):
    """Computes and stores the   average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__=="__main__":
    main()
