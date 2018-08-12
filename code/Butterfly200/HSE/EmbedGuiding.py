import math

import torch
import torch.nn as nn


class EmbedGuiding(nn.Module):
    def __init__(self, prior='genus', init=True):
        super(EmbedGuiding, self).__init__()
        if prior == 'family':
            self.fc = nn.Linear(5, 1024)
        if prior == 'subfamily':
            self.fc = nn.Linear(23, 1024)
        if prior == 'genus':
            self.fc = nn.Linear(116, 1024)

        self.conv1024 = nn.Conv2d(in_channels=2048+1024, out_channels=1024, kernel_size=1)
        self.conv2048= nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU(inplace=True)

        if init:
            self._init_weights()

    def forward(self, scores, features):
        '''
        scores: (batch_size, scores_dim)
        features: (batch_size, channels, height, width)
        '''
        batch_size = scores.size(0)
        assert features.size(2) == features.size(3), "feature map w & h must be same: {},{}".format(features.size[1], features.size[2])
        size_fm = features.size(2)
        c = features.size(1)
        '''
        prepare scores
        '''
        # fc -->1024
        s = self.fc(scores)
        # repeat to feature map size
        s = s.repeat(1, size_fm*size_fm)
        s = s.view(batch_size, size_fm, size_fm, -1)
        s = s.permute(0,3,2,1) # (n, c, w, h)
        '''
        embed and learn weights
        '''
        # concate with feature map
        cf = torch.cat((s, features), 1)
        # conv to 1024
        cf = self.conv1024(cf)
        cf = self.tanh(cf)
        # conv to 2048
        cf = self.conv2048(cf)
        
        n_cf = cf.size(0)
        c_cf = cf.size(1)
        w_cf = cf.size(2)
        h_cf = cf.size(3)
        cf = cf.view(n_cf*c_cf, w_cf*h_cf)
        cf = self.softmax(cf)
        prior_weights = cf.view(n_cf, c_cf, w_cf, h_cf)

        '''
        guiding
        '''
        # eltwise product with original feature
        embed_feature = torch.mul(features, prior_weights)
        embed_feature = self.relu(embed_feature)
        return embed_feature

    def _init_weights(self):
        self.fc.weight.data.normal_(0, 0.01)
        self.fc.bias.data.zero_()
        
        m = self.conv1024
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()

        m = self.conv2048
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()

