'''
hierarchical_resnet model
'''
import math

import torch
import torch.nn as nn
from torch.autograd import Variable
from model import Bottleneck, ResNet
from Branch import Branch
from EmbedGuiding import EmbedGuiding

class ResNetEmbed(nn.Module):
    def __init__(self, cdict={}):
        super(ResNetEmbed, self).__init__()
        
        self.num_classes = cdict

        self.trunk = self._make_trunk()
        self.branch_prior = self._make_branch(level='sup')
        self.branch_inferior_1 = self._make_branch(level='sub')
        self.branch_inferior_2 = self._make_branch(level='sub')


        self.softmax = nn.Softmax(dim=1)
        self.logsoftmax = nn.LogSoftmax()

        self.guide = EmbedGuiding(prior='sup')
        
        self.avgpool = nn.AvgPool2d(14, stride=1) # for 448*448 input

        self.fc_prior = nn.Linear(2048, self.num_classes['sup'])
        self.fc_inferior_1 = nn.Linear(2048, self.num_classes['sub'])
        self.fc_inferior_2 = nn.Linear(2048, self.num_classes['sub'])
        self.fc_inferior_cat = nn.Linear(2048*2, self.num_classes['sub'])

    def _make_trunk(self):
        trunk = ResNet(Bottleneck, [3, 4, 6], need_fc=False)
        return trunk
    
    def _make_branch(self, level):
                       
        return Branch(level)


    def forward(self, x):
        # get share feature
        fs = self.trunk(x)

        # fed to prior branch and compute scores
        fp = self.branch_prior(fs) # feature of prior
        fp = self.avgpool(fp)
        fp = fp.view(x.size(0), -1)
        sp = self.fc_prior(fp) # score of prior
        # sp_ = Variable(sp.data.clone()/self.sm_t, requires_grad=False).cuda()
        sp_ = Variable(sp.data.clone(), requires_grad=False).cuda()
        sp_ = self.softmax(sp_)
        # compute feature of inferiorResNetEmbed2BranchCat
        fi_1 = self.branch_inferior_1(fs)
        fi_2 = self.branch_inferior_2(fs)
        # embeded guiding
        fi_guided = self.guide(sp_, fi_1) # "sp_(score_prior)----<guide>---->fi(feature_inferior)"
        # predict inferior
        fi_1_pooled = torch.sum(fi_guided.view(fi_guided.size(0), fi_guided.size(1), -1), dim=2)
        fi_1_pooled = fi_1_pooled.view(x.size(0), -1)
        si_1 = self.fc_inferior_1(fi_1_pooled)

        fi_2_pooled = self.avgpool(fi_2)
        fi_2_pooled = fi_2_pooled.view(x.size(0), -1)
        si_2 = self.fc_inferior_2(fi_2_pooled)

        fi_cat = torch.cat((fi_1_pooled, fi_2_pooled), dim=1)
        
        si_cat = self.fc_inferior_cat(fi_cat)

        si_avg = (si_1 + si_2 + si_cat) / 3

        return sp, si_avg

