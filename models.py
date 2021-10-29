import torch
import torch.nn as nn
import torch.nn.functional as F

from torchmeta.modules import (MetaModule, MetaConv2d, MetaBatchNorm2d,
                               MetaSequential, MetaLinear)
from utils.utils import get_subdict
from collections import OrderedDict
from torch.distributions import Bernoulli

#####################
### GRADIENT MASK ###
#####################

class BinaryLayer(torch.autograd.Function):
    def __init__(self):
        super(BinaryLayer, self).__init__()

    @staticmethod
    def forward(ctx, input):
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class ReluStraightThrough(torch.autograd.Function):
    def __init__(self):
        super(ReluStraightThrough, self).__init__()

    @staticmethod
    def forward(ctx, input):
        return torch.relu(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class GradientMaskPlus(nn.Module):
    def __init__(self, args, layer_names, layer_sizes, feature_size):
        
        super(GradientMaskPlus,self).__init__()
        self.layer_names=layer_names
        self.layer_sizes=layer_sizes
        self.meta_relu = args.meta_relu    
        self.meta_relu_through = args.meta_relu_through
        self.meta_sgd_linear = args.meta_sgd_linear
        self.noise_std = 1.0
        self.total_layers=sum(layer_sizes)
        self.n=layer_sizes[0]*layer_sizes[0]*layer_sizes[1]
        self.alpha_size=sum(layer_sizes)*(self.n)
        self.out_shift=args.init_shift
        self.alphas=nn.Linear(feature_size, self.alpha_size)
        
        self.weight_embedding_list = nn.ParameterList([])
        self.weight_embedding_list.append(nn.Parameter(
                                   torch.ones(feature_size)))
        self.weight_embedding_list.append(nn.Parameter(
                                   torch.zeros(feature_size)))

        print("\nsparse MAML plus activated. Plus adaptation on: \n")
        print(*self.layer_names, sep = "\n")

    def forward(self):
        
    
        x = torch.randn_like(self.weight_embedding_list[0])*\
                    self.weight_embedding_list[0]*self.noise_std \
                                        + self.weight_embedding_list[1]

        alphas=self.alphas(x).reshape(self.total_layers, self.n)
        #Divide alpha into the appropriate rows
        prev=0
        masks={}
        for i, name in enumerate(self.layer_names):
           
            alpha=alphas[prev:prev+self.layer_sizes[i]]
            alpha=alpha.reshape(64, alpha.shape[0], 3, 3)

            
            if self.meta_relu_through:
                masks[name] = ReluStraightThrough.apply(alpha + self.out_shift)
            elif self.meta_relu:
                masks[name] = torch.relu(alpha + self.out_shift)
            elif self.meta_sgd_linear:
                masks[name] = alpha + self.out_shift
            else:
                sign = BinaryLayer.apply(alpha + self.out_shift)
                masks[name]= 0.5*(1 + sign)
            prev+=self.layer_sizes[i]
        return masks

class GradientMask(nn.Module):
    def __init__(self, args, weight_names, weight_shapes, mask_plus):
        super(GradientMask, self).__init__()

        self.weight_names = weight_names
        self.weight_shapes = weight_shapes
        self.weight_mask_list = nn.ParameterList([])
        self.mask_plus = mask_plus
        self.meta_relu = args.meta_relu
        self.meta_sgd_linear = args.meta_sgd_linear
        self.meta_exp = args.meta_exp
        self.meta_relu_through = args.meta_relu_through 
        self.meta_sgd_init = args.meta_sgd_init

        weight_names_new = []
            
        for i, name in zip(range(len(weight_shapes)), weight_names):

            # if this is given, the mask will come from the x dep hnet
            if self.mask_plus is not None and "conv" in name:
                continue

            weight_names_new.append(name)
            alpha = nn.Parameter(torch.zeros(weight_shapes[i]))

            if self.meta_sgd_init:
                nn.init.uniform_(alpha, a=args.step_size, b=args.step_size)
            else:
                if len(weight_shapes[i]) > 1 and args.kaiming_init:
                    nn.init.kaiming_uniform_(alpha)
                else:
                    nn.init.uniform_(alpha, a=-0.5, b=0.5)  
                # control the mean / sparsity init explicitly
                alpha.data = alpha.data-torch.mean(alpha.data)+args.init_shift

            self.weight_mask_list.append(alpha)

        self.weight_names = weight_names_new 
        
        print("\nNormal inner loop modulation on:")
        print(*self.weight_names, sep = "\n")


    def forward(self):
        masks = {}
        for name, x in zip(self.weight_names, self.weight_mask_list):
            if self.meta_relu_through: 
                x = ReluStraightThrough.apply(x)
            elif self.meta_sgd_linear:
                x = x
            elif self.meta_relu:
                x = torch.relu(x)
            elif self.meta_exp:
                x = torch.exp(x)
            else:
                x = 0.5*(BinaryLayer.apply(x) + 1)
            masks[name] = x

        if self.mask_plus is not None:
            masks = {**masks, **self.mask_plus()}

        return masks

#################
### Conv4-Net ###
#################

def conv_block(in_channels, out_channels, non_lin, **kwargs):
    return MetaSequential(OrderedDict([
      ('conv', MetaConv2d(in_channels, out_channels, **kwargs)),
      ('norm', MetaBatchNorm2d(out_channels, track_running_stats=False)),
      ('relu', non_lin),
      ('pool', nn.MaxPool2d(2))
    ]))

class MetaConvModel(MetaModule):
    def __init__(self,in_channels,out_features,hidden_size=64,feature_size=64, 
                                                             non_lin=nn.ReLU(),
                                                             bias=True):
        super(MetaConvModel,self).__init__()
        self.in_channels=in_channels
        self.out_features=out_features
        self.hidden_size=hidden_size
        self.feature_size=feature_size
        self.bias = bias,
        self.features = MetaSequential(OrderedDict([
        ('layer1', conv_block(in_channels, hidden_size, non_lin, 
                            kernel_size=3, stride=1, padding=1, bias=self.bias)),
        ('layer2', conv_block(hidden_size, hidden_size, non_lin,  
                            kernel_size=3, stride=1, padding=1, bias=self.bias)),
        ('layer3', conv_block(hidden_size, hidden_size, non_lin,
                            kernel_size=3, stride=1, padding=1, bias=self.bias)),
        ('layer4', conv_block(hidden_size, hidden_size, non_lin,
                            kernel_size=3, stride=1, padding=1, bias=self.bias))
        ]))
        self.classifier = MetaLinear(feature_size, out_features, bias=self.bias)

    def forward(self, inputs, params=None):
        features = self.features(inputs, params=get_subdict(params, 'features'))
        features = features.view((features.size(0), -1))
        logits=self.classifier(features,params=get_subdict(params,'classifier'))
        return logits


#################
### ResNet-12 ###
#################

"""
ResNet Code copied from https://github.com/HJ-Yoo/BOIL
"""

class DropBlock(MetaModule):
    def __init__(self, block_size):
        super(DropBlock, self).__init__()

        self.block_size = block_size

    def forward(self, x, gamma):

        if self.training:
            batch_size, channels, height, width = x.shape
            
            bernoulli = Bernoulli(gamma)
            mask = bernoulli.sample((batch_size, channels, height - \
                   (self.block_size - 1), width - (self.block_size - 1))).cuda()
            block_mask = self._compute_block_mask(mask)

            countM = block_mask.size()[0] * block_mask.size()[1] * \
                     block_mask.size()[2] * block_mask.size()[3]
            count_ones = block_mask.sum()

            return block_mask * x * (countM / count_ones)
        else:
            return x

    def _compute_block_mask(self, mask):
        left_padding = int((self.block_size-1) / 2)
        right_padding = int(self.block_size / 2)
        
        batch_size, channels, height, width = mask.shape
        non_zero_idxs = torch.nonzero(mask)
        nr_blocks = non_zero_idxs.shape[0]

        offsets = torch.stack(
            [
                torch.arange(self.block_size).view(-1, 1).\
                    expand(self.block_size, self.block_size).reshape(-1), 
                torch.arange(self.block_size).repeat(self.block_size), 
            ]
        ).t().cuda()
        offsets = torch.cat((torch.zeros(self.block_size**2, 2).\
                             cuda().long(), offsets.long()), 1)
        
        if nr_blocks > 0:
            non_zero_idxs = non_zero_idxs.repeat(self.block_size ** 2, 1)
            offsets = offsets.repeat(nr_blocks, 1).view(-1, 4)
            offsets = offsets.long()

            block_idxs = non_zero_idxs + offsets
            padded_mask = F.pad(mask, (left_padding, right_padding, 
                                       left_padding, right_padding))
            padded_mask[block_idxs[:, 0], block_idxs[:, 1], 
                        block_idxs[:, 2], block_idxs[:, 3]] = 1.
        else:
            padded_mask = F.pad(mask, (left_padding, right_padding, 
                                       left_padding, right_padding))
            
        block_mask = 1 - padded_mask
        return block_mask


class BasicBlock(MetaModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, 
                downsample=None, drop_rate=0.0, drop_block=False, block_size=1):
        super(BasicBlock, self).__init__()
        self.conv1 = MetaConv2d(inplanes, planes, kernel_size=3, 
                                stride=1, padding=1, bias=False)
        self.bn1 = MetaBatchNorm2d(planes, track_running_stats=False)
        self.relu1 = nn.LeakyReLU()
        self.conv2 = MetaConv2d(planes, planes, kernel_size=3, 
                                stride=1, padding=1, bias=False)
        self.bn2 = MetaBatchNorm2d(planes, track_running_stats=False)
        self.relu2 = nn.LeakyReLU()
        self.conv3 = MetaConv2d(planes, planes, kernel_size=3, 
                                stride=1, padding=1, bias=False)
        self.bn3 = MetaBatchNorm2d(planes, track_running_stats=False)
        self.relu3 = nn.LeakyReLU()
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)

    def forward(self, x, params=None):
        self.num_batches_tracked += 1

        residual = x

        out = self.conv1(x, params=get_subdict(params, 'conv1'))
        out = self.bn1(out, params=get_subdict(params, 'bn1'))
        out = self.relu1(out)

        out = self.conv2(out, params=get_subdict(params, 'conv2'))
        out = self.bn2(out, params=get_subdict(params, 'bn2'))
        out = self.relu2(out)

        out = self.conv3(out, params=get_subdict(params, 'conv3'))
        out = self.bn3(out, params=get_subdict(params, 'bn3'))

        if self.downsample is not None:
            residual=self.downsample(x,params=get_subdict(params, 'downsample'))
        out += residual
        out = self.relu3(out)
        
        out = self.maxpool(out)
        
        if self.drop_rate > 0:
            if self.drop_block == True:
                feat_size = out.size()[2]
                keep_rate = max(1.0 - self.drop_rate / (20*2000) * \
                            (self.num_batches_tracked), 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size**2 * feat_size**2 / \
                            (feat_size - self.block_size + 1)**2
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, 
                                training=self.training, inplace=True)

        return out

class BasicBlockWithoutResidual(MetaModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, 
                 drop_rate=0.0, drop_block=False, block_size=1):
        super(BasicBlockWithoutResidual, self).__init__()
        self.conv1 = MetaConv2d(inplanes, planes, kernel_size=3, 
                                stride=1, padding=1, bias=False)
        self.bn1 = MetaBatchNorm2d(planes, track_running_stats=False)
        self.relu1 = nn.LeakyReLU()
        self.conv2 = MetaConv2d(planes, planes, kernel_size=3, 
                                stride=1, padding=1, bias=False)
        self.bn2 = MetaBatchNorm2d(planes, track_running_stats=False)
        self.relu2 = nn.LeakyReLU()
        self.conv3 = MetaConv2d(planes, planes, kernel_size=3, 
                                stride=1, padding=1, bias=False)
        self.bn3 = MetaBatchNorm2d(planes, track_running_stats=False)
        self.relu3 = nn.LeakyReLU(0)
        self.maxpool = nn.MaxPool2d(stride)
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)

    def forward(self, x, params=None):
        self.num_batches_tracked += 1

        out = self.conv1(x, params=get_subdict(params, 'conv1'))
        out = self.bn1(out, params=get_subdict(params, 'bn1'))
        out = self.relu1(out)

        out = self.conv2(out, params=get_subdict(params, 'conv2'))
        out = self.bn2(out, params=get_subdict(params, 'bn2'))
        out = self.relu2(out)

        out = self.conv3(out, params=get_subdict(params, 'conv3'))
        out = self.bn3(out, params=get_subdict(params, 'bn3'))
        out = self.relu3(out)
        
        out = self.maxpool(out)
        
        if self.drop_rate > 0:
            if self.drop_block == True:
                feat_size = out.size()[2]
                keep_rate = max(1.0 - self.drop_rate / (20*2000) * \
                            (self.num_batches_tracked), 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size**2 * feat_size**2 /\
                            (feat_size - self.block_size + 1)**2
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, 
                                training=self.training, inplace=True)

        return out

class ResNet(MetaModule):
    def __init__(self, keep_prob=1.0, avg_pool=True, drop_rate=0.0, 
                dropblock_size=5, out_features=5, wh_size=1, big_network=False):

        # NOTE  keep_prob < 1 and drop_rate > 0 are NOT supported!

        self.inplanes = 3
        super(ResNet, self).__init__()

        blocks = [BasicBlock, BasicBlock, BasicBlock, BasicBlock]
        
        if big_network:
            num_chn = [64, 160, 320, 640]
        else:
            num_chn = [64, 128, 256, 512]

        self.layer1 = self._make_layer(blocks[0], num_chn[0], stride=2, 
                                        drop_rate=drop_rate, drop_block=True, 
                                        block_size=dropblock_size)
        self.layer2 = self._make_layer(blocks[1], num_chn[1], stride=2, 
                                        drop_rate=drop_rate, drop_block=True, 
                                        block_size=dropblock_size)
        self.layer3 = self._make_layer(blocks[2], num_chn[2], stride=2, 
                                        drop_rate=drop_rate, drop_block=True, 
                                        block_size=dropblock_size)
        self.layer4 = self._make_layer(blocks[3], num_chn[3], stride=2, 
                                        drop_rate=drop_rate, drop_block=True, 
                                        block_size=dropblock_size)
        if avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.keep_prob = keep_prob
        self.keep_avg_pool = avg_pool
        self.dropout = nn.Dropout(p=1 - self.keep_prob, inplace=False)
        self.drop_rate = drop_rate
        
        self.classifier = MetaLinear(512*wh_size*wh_size, out_features)
        
        for m in self.modules():
            if isinstance(m, MetaConv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, MetaBatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, stride=1, drop_rate=0.0, 
                    drop_block=False, block_size=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = MetaSequential(
                MetaConv2d(self.inplanes, planes * block.expansion,
                                kernel_size=1, stride=1, bias=False),
                MetaBatchNorm2d(planes * block.expansion, 
                                track_running_stats=False),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, 
                      downsample, drop_rate, drop_block, block_size))
        self.inplanes = planes * block.expansion

        return MetaSequential(*layers)

    def forward(self, x, params=None):
        x = self.layer1(x, params=get_subdict(params, 'layer1'))
        x = self.layer2(x, params=get_subdict(params, 'layer2'))
        x = self.layer3(x, params=get_subdict(params, 'layer3'))
        x = self.layer4(x, params=get_subdict(params, 'layer4'))
        if self.keep_avg_pool:
            x = self.avgpool(x)
        features = x.view((x.size(0), -1))
        return self.classifier(self.dropout(features), 
                               params=get_subdict(params, 'classifier'))
