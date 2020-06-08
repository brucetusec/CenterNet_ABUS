# ------------------------------------------------------------------------------
# This code is based on 
# CornerNet (https://github.com/princeton-vl/CornerNet)
# Copyright (c) 2018, University of Michigan
# Licensed under the BSD 3-Clause License
# ------------------------------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
from .basic import BasicModule

class convolution(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_gn=True, dilation=1):
        super(convolution, self).__init__()

        pad = (k - 1) // 2
        self.conv = nn.Conv3d(inp_dim, out_dim, (k, k, k), padding=(pad, pad, pad), stride=(stride, stride, stride), bias=not with_gn, dilation=dilation)
        self.relu = nn.ReLU(inplace=True)
        self.bn   = nn.GroupNorm(16, out_dim) if with_gn else nn.Sequential()

    def forward(self, x):
        conv = self.conv(x)
        relu = self.relu(conv)
        bn   = self.bn(relu)
        return bn


class asym_convolution(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_gn=True):
        super(asym_convolution, self).__init__()

        pad = (k - 1) // 2
        self.conv = nn.Conv3d(inp_dim, out_dim, (k, 3, k), padding=(pad, 1, pad), stride=(stride, stride, stride), bias=not with_gn)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.conv(x)
        relu = self.relu(conv)
        return relu


class residual(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_gn=True):
        super(residual, self).__init__()

        pad = (k - 1) // 2
        self.bn1   = nn.GroupNorm(16, inp_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(inp_dim, out_dim, (k, k, k), padding=(pad, pad, pad), stride=(stride, stride, stride), bias=not with_gn)

        self.bn2   = nn.GroupNorm(16, out_dim)
        self.relu2  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_dim, out_dim, (k, k, k), padding=(pad, pad, pad), bias=False)

        self.skip  = nn.Sequential(
            nn.Conv3d(inp_dim, out_dim, (1, 1, 1), stride=(stride, stride, stride), bias=False),
        ) if stride != 1 or inp_dim != out_dim else nn.Sequential()

    def forward(self, x):
        bn1   = self.bn1(x)
        relu1 = self.relu1(bn1)
        conv1 = self.conv1(relu1)

        bn2   = self.bn2(conv1)
        relu2 = self.relu2(bn2)
        conv2 = self.conv2(relu2)

        skip  = self.skip(x)
        return conv2 + skip


class residual_2D(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_gn=True):
        super(residual_2D, self).__init__()

        pad = (k - 1) // 2
        self.bn1   = nn.GroupNorm(16, inp_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(inp_dim, out_dim, (k, k, k), padding=(pad, pad, pad), stride=(stride, stride, stride), bias=not with_gn)

        self.bn2   = nn.GroupNorm(16, out_dim)
        self.relu2  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_dim, out_dim, (k, 1, k), padding=(pad, 0, pad), bias=False)

        self.skip  = nn.Sequential(
            nn.Conv3d(inp_dim, out_dim, (1, 1, 1), stride=(stride, stride, stride), bias=False),
        ) if stride != 1 or inp_dim != out_dim else nn.Sequential()

    def forward(self, x):
        bn1   = self.bn1(x)
        relu1 = self.relu1(bn1)
        conv1 = self.conv1(relu1)

        bn2   = self.bn2(conv1)
        relu2 = self.relu2(bn2)
        conv2 = self.conv2(relu2)

        skip  = self.skip(x)
        return conv2 + skip


def make_layer(k, inp_dim, out_dim, modules, layer=convolution, **kwargs):
    layers = [layer(k, inp_dim, out_dim, **kwargs)]
    for _ in range(1, modules):
        layers.append(layer(k, out_dim, out_dim, **kwargs))
    return nn.Sequential(*layers)

def make_layer_revr(k, inp_dim, out_dim, modules, layer=convolution, **kwargs):
    layers = []
    for _ in range(modules - 1):
        layers.append(layer(k, inp_dim, inp_dim, **kwargs))
    layers.append(layer(k, inp_dim, out_dim, **kwargs))
    return nn.Sequential(*layers)

class MergeUp(nn.Module):
    def forward(self, up1, up2):
        return up1 + up2

class MergeCat(nn.Module):
    def forward(self, up1, up2):
        return torch.cat((up1, up2), 1)

def make_merge_layer(dim):
    return MergeCat()

# def make_pool_layer(dim):
#     return nn.MaxPool2d(kernel_size=2, stride=2)

def make_pool_layer(dim):
    return nn.Sequential()

def make_unpool_layer(dim):
    return nn.Upsample(scale_factor=2)

def make_kp_layer(cnv_dim, curr_dim, out_dim):
    return nn.Sequential(
        residual(3, cnv_dim, curr_dim, with_gn=True),
        nn.Conv3d(curr_dim, out_dim, (1, 1, 1))
    )

def make_hg_layer(kernel, dim0, dim1, mod, layer=convolution, **kwargs):
    layers  = [layer(kernel, dim0, dim1, stride=2)]
    layers += [layer(kernel, dim1, dim1, **kwargs) for _ in range(mod - 1)]
    return nn.Sequential(*layers)

def make_hm_layer(cnv_dim, curr_dim, out_dim):
    return nn.Sequential(
        residual(3, cnv_dim, curr_dim, with_gn=True),
        nn.Conv3d(curr_dim, out_dim, (1, 1, 1)),
        nn.Sigmoid()
    )

def make_inter_layer(dim):
    return residual(3, dim, dim)

def make_cnv_layer(inp_dim, out_dim):
    return convolution(3, inp_dim, out_dim)

class kp_module(nn.Module):
    def __init__(
        self, n, dims, modules, layer=residual,
        make_up_layer=make_layer, make_low_layer=make_layer,
        make_hg_layer=make_layer, make_hg_layer_revr=make_layer_revr,
        make_pool_layer=make_pool_layer, make_unpool_layer=make_unpool_layer,
        make_merge_layer=make_merge_layer, debug=False,**kwargs
    ):
        super(kp_module, self).__init__()

        self.n   = n

        if debug:
            print('Current depth:', n, 'Modules:', modules)

        curr_mod = modules[0]
        next_mod = modules[1]

        curr_dim = dims[0]
        next_dim = dims[1]
            
        self.up1  = make_up_layer(
            3, curr_dim, curr_dim, curr_mod, 
            layer=residual, **kwargs
        )  
        self.max1 = make_pool_layer(curr_dim)
        self.low1 = make_hg_layer(
            3, curr_dim, next_dim, curr_mod,
            layer=residual, **kwargs
        )
        if self.n > 1:
            self.low2 = kp_module(
                n - 1, dims[1:], modules[1:], layer=layer, 
                make_up_layer=make_up_layer, 
                make_low_layer=make_low_layer,
                make_hg_layer=make_hg_layer,
                make_hg_layer_revr=make_hg_layer_revr,
                make_pool_layer=make_pool_layer,
                make_unpool_layer=make_unpool_layer,
                make_merge_layer=make_merge_layer,
                debug=debug,
                **kwargs
            )
            self.low3 = make_hg_layer_revr(
                3, next_dim*2, curr_dim, next_mod,
                layer=layer, **kwargs
            )
        else:
            self.low2 = make_low_layer(
                3, next_dim, next_dim, next_mod,
                layer=layer, **kwargs
            )
            self.low3 = make_hg_layer_revr(
                3, next_dim, curr_dim, next_mod,
                layer=layer, **kwargs
            )

        self.up2  = make_unpool_layer(curr_dim)

        self.merge = make_merge_layer(curr_dim)

    def forward(self, x):
        up1  = self.up1(x)
        max1 = self.max1(x)
        low1 = self.low1(max1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2  = self.up2(low3)
        return self.merge(up1, up2)

class exkp(BasicModule):
    def __init__(
        self, n, nstack, dims, modules, heads, pre=None, cnv_dim=256, 
        make_cnv_layer=make_cnv_layer,
        make_heat_layer=make_hm_layer,
        make_regr_layer=make_kp_layer,
        make_up_layer=make_layer, make_low_layer=make_layer, 
        make_hg_layer=make_layer, make_hg_layer_revr=make_layer_revr,
        make_pool_layer=make_pool_layer, make_unpool_layer=make_unpool_layer,
        make_merge_layer=make_merge_layer, make_inter_layer=make_inter_layer, 
        kp_layer=residual, debug=False
    ):
        super(exkp, self).__init__()
        self.debug     = debug

        self.nstack    = nstack
        self.heads     = heads

        curr_dim = dims[0]

        self.pre = nn.Sequential(
            asym_convolution(7, 1, 16, stride=2),
            residual_2D(3, 16, 32, stride=2)
        ) if pre is None else pre

        self.kps  = nn.ModuleList([
            kp_module(
                n, dims, modules, layer=kp_layer,
                make_up_layer=make_up_layer,
                make_low_layer=make_low_layer,
                make_hg_layer=make_hg_layer,
                make_hg_layer_revr=make_hg_layer_revr,
                make_pool_layer=make_pool_layer,
                make_unpool_layer=make_unpool_layer,
                make_merge_layer=make_merge_layer,
                debug=self.debug
            ) for _ in range(nstack)
        ])
        self.cnvs = nn.ModuleList([
            make_cnv_layer(curr_dim*2, cnv_dim) for _ in range(nstack)
        ])

        self.inters = nn.ModuleList([
            make_inter_layer(curr_dim) for _ in range(nstack - 1)
        ])

        self.inters_ = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(curr_dim, curr_dim, (1, 1, 1), bias=False),
                nn.GroupNorm(16, curr_dim)
            ) for _ in range(nstack - 1)
        ])
        self.cnvs_   = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(cnv_dim, curr_dim, (1, 1, 1), bias=False),
                nn.GroupNorm(16, curr_dim)
            ) for _ in range(nstack - 1)
        ])

        ## Output heads: hm = Heat map, wh = Width-Height, off = Offset
        for head in heads.keys():
            if 'hm' in head:
                module =  nn.ModuleList([
                    make_heat_layer(
                        cnv_dim, curr_dim, heads[head]) for _ in range(nstack)
                ])
                self.__setattr__(head, module)
                for heat in self.__getattr__(head):
                    heat[-2].bias.data.fill_(-2.19)
            else:
                module = nn.ModuleList([
                    make_regr_layer(
                        cnv_dim, curr_dim, heads[head]) for _ in range(nstack)
                ])
                self.__setattr__(head, module)


        self.relu = nn.ReLU(inplace=True)

    def forward(self, image):
        if self.debug:
            print('Original image:', image.shape)

        inter = self.pre(image)
        outs  = []

        if self.debug:
            print('Preprocessed image:', inter.shape)
        for i in range(self.nstack):
            kp_, cnv_  = self.kps[i], self.cnvs[i]
            kp  = kp_(inter)
            cnv = cnv_(kp)

            if self.debug:
                print('Before output head:', cnv.shape)
            out = {}
            for head in self.heads:
                layer = self.__getattr__(head)[i]
                y = layer(cnv)
                out[head] = y
            
            outs.append(out)
            if i < self.nstack - 1:
                inter = self.inters_[i](inter) + self.cnvs_[i](cnv)
                inter = self.relu(inter)
                inter = self.inters[i](inter)
        return outs


class HourglassNet(exkp):
    def __init__(self, heads, num_stacks=1, debug=False):
        # How deep do you wanna go? (# of Connections between layers)
        n       = 2
        # Number of channel
        dims    = [32, 64, 128]
        # Number of layers of convolution
        modules = [2, 2, 2]

        super(HourglassNet, self).__init__(
            n, num_stacks, dims, modules, heads,
            make_pool_layer=make_pool_layer,
            make_hg_layer=make_hg_layer,
            kp_layer=residual, cnv_dim=32, debug=debug
        )

def get_large_hourglass_net(heads, n_stacks=1, debug=False):
  model = HourglassNet(heads, n_stacks, debug=debug)
  return model
