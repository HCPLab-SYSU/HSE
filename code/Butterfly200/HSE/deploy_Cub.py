import os,sys
import pdb
sys.path.insert(0,'.')
import shutil
import time,pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader

from torch.autograd import Variable
from dataset import Butterfly200

from ResNetEmbed import ResNetEmbed
from PIL import Image
import scipy.io as sio

import torch.utils.data as data

def load_connective_matrix():
    connect_matrix_file = "../../../data/Butterfly200/connect_matrix_Butterfly200.mat"

    c_mat = sio.loadmat(connect_matrix_file)['connect_matrix']
    c_mat = torch.autograd.Variable(torch.from_numpy(c_mat.transpose()).float())

    mat_dict = {}
    mat_dict['L1L2'] = c_mat[:5, 5:5+24] # ==> (5 * 24)
    mat_dict['L2L3'] = c_mat[5:5+24, 5+24:-200] # ==> (24, 116)
    mat_dict['L3L4'] = c_mat[5+24:-200, -200:] # ==> (116, 200)
    
    return mat_dict

def main():
    global best_prec1_order

    # Create dataloader
    print "==> Creating dataloader..."
    data_dir='../../../data/Butterfly200/images_rz'
    test_list='../../../data/Butterfly200/Butterfly200_test_V3.txt'
    test_loader = get_training_test_set(data_dir, test_list)

    # load the network
    print "==> Loading the network ..."
    classes_dict = {'family': 5, 'subfamily': 24, 'genus': 116, 'species': 200}
    model = ResNetEmbed(branch_type='last_layer', cdict=classes_dict, sm_t=4, weight_norm='ch_softmax')

    model.cuda()
  
    # optionally resume from a checkpoint
    
    # for CUB
    # resume = "../../pretrains/production/CUB_200_2011/ft_last_layer/class/model_class_FT_layer4_ImgnetPreT_RS_85.243_LRdt0.0001_WD0.00005_85.675_eph_314.pth.tar"
    resume = "../../../pretrains/production/Butterfly200/L1L2L3L4_4Loss_KL/model_L1L2L3L4_ch_softmax_2BRCH_4LOSS_A1_B1_C1_D1_E1_Minferior_LR0.001_WD0.00005_T4_86.118_eph_137.pth.tar"

    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(resume, checkpoint['epoch']))

    cudnn.benchmark = True

    connect_m = load_connective_matrix()

    validate(test_loader, model, connect_m)

def validate(val_loader, model, connect_m):
    batch_time = AverageMeter()

    top1_order = AverageMeter()
    top5_order = AverageMeter()
    top1_family = AverageMeter()
    top5_family = AverageMeter()
    top1_genus = AverageMeter()
    top5_genus = AverageMeter()
    top1_class = AverageMeter()
    top5_class = AverageMeter()

    # switch to evaluate mode
    model.eval() 

    end = time.time()
    # for i, (input, gt_order) in enumerate(val_loader):
    f = open('predict_test.txt', 'w')
    for i, (input, gt_order, gt_family, gt_genus, gt_class) in enumerate(val_loader):

        # gt_order = gt_order.cuda(async=True)
        # gt_family = gt_family.cuda(async=True)
        # gt_genus = gt_genus.cuda(async=True)
        gt_class = gt_class.cuda(async=True)

        input_var = torch.autograd.Variable(input, volatile=True).cuda()
        # gt_order_var = torch.autograd.Variable(gt_order, volatile=True)
        # gt_family_var = torch.autograd.Variable(gt_family, volatile=True)
        # gt_genus_var = torch.autograd.Variable(gt_genus, volatile=True)
        gt_class_var = torch.autograd.Variable(gt_class, volatile=True)

        # compute pred_class
        sm = nn.Softmax()
        pred1_L1, pred1_L2, pred1_L3, pred1_L4, KL_prior, KL_inferior = model(input_var, connect_m)
        pred1_L4_g, pred1_L4_r, pred1_L4_cat, pred1_L4_avg = pred1_L4
        pred_class = sm(pred1_L4_avg)
        
        # pred_class_ = Variable(pred_class.data.clone(), requires_grad=False).cuda()
        m, m_idx = torch.max(pred_class, 1)
        for p in m_idx:
            f.write("{:d}\n".format(p.data[0]))

        # pred_class_ = pred_class.eq(m.view(pred_class.size(0), -1).repeat(1, pred_class.size(1)))
        # pred_class_ = pred_class_.float()
        # pred_genus = pred_class_.mm(connect_m['L43'].cuda())
        
        # # pred_genus_ = Variable(pred_genus.data.clone(), requires_grad=False).cuda()
        # m, m_idx = torch.max(pred_genus, 1)
        # pred_genus_ = pred_genus.eq(m.view(pred_genus.size(0), -1).repeat(1, pred_genus.size(1)))
        # pred_genus_ = pred_genus_.float()
        # pred_family = pred_genus_.mm(connect_m['L32'].cuda())

        # # pred_family_ = Variable(pred_family.data.clone(), requires_grad=False).cuda()
        # m, m_idx = torch.max(pred_family, 1)
        # pred_family_ = pred_family.eq(m.view(pred_family.size(0), -1).repeat(1, pred_family.size(1)))
        # pred_family_ = pred_family_.float()
        # pred_order = pred_family_.mm(connect_m['L21'].cuda())
        

        # measure accuracy and record loss
        # prec1_order, prec5_order = accuracy(pred_order.data, gt_order, topk = (1,5))
        # prec1_family, prec5_family = accuracy(pred_family.data, gt_family, topk = (1,5))
        # prec1_genus, prec5_genus = accuracy(pred_genus.data, gt_genus, topk = (1,5))
        prec1_class, prec5_class = accuracy(pred1_L4_avg.data, gt_class, topk = (1,5))

        # top1_order.update(prec1_order[0], input.size(0))
        # top5_order.update(prec5_order[0], input.size(0))
        # top1_family.update(prec1_family[0], input.size(0))
        # top5_family.update(prec5_family[0], input.size(0))
        # top1_genus.update(prec1_genus[0], input.size(0))
        # top5_genus.update(prec5_genus[0], input.size(0))
        top1_class.update(prec1_class[0], input.size(0))
        top5_class.update(prec5_class[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0:
            print "iters: {}/{}".format(i, len(val_loader))
            print "time: {batch_time.avg:.3f}".format(batch_time=batch_time)
    f.close()
    
    # print(' * (order)@1 {top1_order.avg:.3f} Prec@5 {top5_order.avg:.3f}'
    #       .format(top1_order=top1_order, top5_order=top5_order))
    # print(' * (family)@1 {top1_family.avg:.3f} Prec@5 {top5_family.avg:.3f}'
    #       .format(top1_family=top1_family, top5_family=top5_family))
    # print(' * (genus)@1 {top1_genus.avg:.3f} Prec@5 {top5_genus.avg:.3f}'
    #       .format(top1_genus=top1_genus, top5_genus=top5_genus))
    print(' * (class)@1 {top1_class.avg:.3f} Prec@5 {top5_class.avg:.3f}'
          .format(top1_class=top1_class, top5_class=top5_class))

    return top1_order.avg

def get_training_test_set(data_dir, test_list):
    # Data loading code
    # normalize for different pretrain model:
    #   the first one for pytorch vgg
    #   the seconde one for caffe vgg
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    
    crop_size = 448
    scale_size = 512
    # center crop
    test_data_transform = transforms.Compose([
          transforms.Scale((scale_size,scale_size)),
          transforms.CenterCrop(crop_size),
          transforms.ToTensor(),
          normalize,
      ])

# from dataset import Butterfly200
    test_set = Butterfly200(data_dir, test_list, test_data_transform)
    test_loader = DataLoader(dataset=test_set, num_workers=2, batch_size=32, shuffle=False)

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

def accuracy(pred, gt, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = gt.size(0)

    _, pred = pred.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(gt.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__=="__main__":
    main()
