import copy
import math
import torch
import os
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from einops import rearrange
# from .RTMPose_GAU import RTMCCBlock,ScaleNorm

import logging
logger = logging.getLogger(__name__)

def build_act_layer(act_type):
    """Build activation layer."""
    if act_type is None:
        return nn.Identity()
    assert act_type in ['GELU', 'ReLU', 'SiLU']
    if act_type == 'SiLU':
        return nn.SiLU()
    elif act_type == 'ReLU':
        return nn.ReLU()
    else:
        return nn.GELU()


def build_norm_layer(norm_type, embed_dims):
    """Build normalization layer."""
    assert norm_type in ['BN', 'GN', 'LN2d', 'SyncBN']
    if norm_type == 'GN':
        return nn.GroupNorm(embed_dims, embed_dims, eps=1e-5)
    if norm_type == 'LN2d':
        return LayerNorm2d(embed_dims, eps=1e-6)
    if norm_type == 'SyncBN':
        return nn.SyncBatchNorm(embed_dims, eps=1e-5)
    else:
        return nn.BatchNorm2d(embed_dims, eps=1e-5)


class LayerNorm2d(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self,
                 normalized_shape,
                 eps=1e-6,
                 data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        assert self.data_format in ["channels_last", "channels_first"]
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ElementScale(nn.Module):
    """A learnable element-wise scaler."""

    def __init__(self, embed_dims, init_value=0., requires_grad=True):
        super(ElementScale, self).__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)),
            requires_grad=requires_grad
        )

    def forward(self, x):
        return x * self.scale


class ChannelAggregationFFN(nn.Module):
    """An implementation of FFN with Channel Aggregation.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`.
        feedforward_channels (int): The hidden dimension of FFNs.
        kernel_size (int): The depth-wise conv kernel size as the
            depth-wise convolution. Defaults to 3.
        act_type (str): The type of activation. Defaults to 'GELU'.
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,  # mlp隐藏层数，等于通道数(embed_dims) * r
                 kernel_size=3,
                 act_type='GELU',
                 ffn_drop=0.):
        super(ChannelAggregationFFN, self).__init__()

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels

        self.fc1 = nn.Conv2d(
            in_channels=embed_dims,
            out_channels=self.feedforward_channels,
            kernel_size=1)
        self.dwconv = nn.Conv2d(
            in_channels=self.feedforward_channels,
            out_channels=self.feedforward_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=True,
            groups=self.feedforward_channels)
        self.act = build_act_layer(act_type)
        self.fc2 = nn.Conv2d(
            in_channels=feedforward_channels,
            out_channels=embed_dims,
            kernel_size=1)
        self.drop = nn.Dropout(ffn_drop)

        self.decompose = nn.Conv2d(
            in_channels=self.feedforward_channels,  # C -> 1
            out_channels=1, kernel_size=1,
        )
        self.sigma = ElementScale(
            self.feedforward_channels, init_value=1e-5, requires_grad=True)
        self.decompose_act = build_act_layer(act_type)

    def feat_decompose(self, x):
        # x_d: [B, C, H, W] -> [B, 1, H, W]
        x = x + self.sigma(x - self.decompose_act(self.decompose(x)))
        return x

    def forward(self, x):
        # proj 1
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        # proj 2
        x = self.feat_decompose(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MultiOrderDWConv(nn.Module):
    """Multi-order Features with Dilated DWConv Kernel.

    Args:
        embed_dims (int): Number of input channels.
        dw_dilation (list): Dilations of three DWConv layers.
        channel_split (list): The raletive ratio of three splited channels.
    """

    def __init__(self,
                 embed_dims,
                 dw_dilation=[1, 2, 3, ],
                 channel_split=[1, 3, 4, ],
                 ):
        super(MultiOrderDWConv, self).__init__()

        self.split_ratio = [i / sum(channel_split) for i in channel_split]
        self.embed_dims_1 = int(self.split_ratio[1] * embed_dims)
        self.embed_dims_2 = int(self.split_ratio[2] * embed_dims)
        self.embed_dims_0 = embed_dims - self.embed_dims_1 - self.embed_dims_2
        self.embed_dims = embed_dims
        assert len(dw_dilation) == len(channel_split) == 3
        assert 1 <= min(dw_dilation) and max(dw_dilation) <= 3
        assert embed_dims % sum(channel_split) == 0

        # basic DW conv
        self.DW_conv0 = nn.Conv2d(
            in_channels=self.embed_dims,
            out_channels=self.embed_dims,
            kernel_size=5,
            padding=(1 + 4 * dw_dilation[0]) // 2,
            groups=self.embed_dims,
            stride=1, dilation=dw_dilation[0],
        )
        # DW conv 1
        self.DW_conv1 = nn.Conv2d(
            in_channels=self.embed_dims_1,
            out_channels=self.embed_dims_1,
            kernel_size=5,
            padding=(1 + 4 * dw_dilation[1]) // 2,
            groups=self.embed_dims_1,
            stride=1, dilation=dw_dilation[1],
        )
        # DW conv 2
        self.DW_conv2 = nn.Conv2d(
            in_channels=self.embed_dims_2,
            out_channels=self.embed_dims_2,
            kernel_size=7,
            padding=(1 + 6 * dw_dilation[2]) // 2,
            groups=self.embed_dims_2,
            stride=1, dilation=dw_dilation[2],
        )
        # a channel convolution
        self.PW_conv = nn.Conv2d(  # point-wise convolution
            in_channels=embed_dims,
            out_channels=embed_dims,
            kernel_size=1)

    def forward(self, x):
        x_0 = self.DW_conv0(x)
        x_1 = self.DW_conv1(
            x_0[:, self.embed_dims_0: self.embed_dims_0 + self.embed_dims_1, ...])
        x_2 = self.DW_conv2(
            x_0[:, self.embed_dims - self.embed_dims_2:, ...])
        x = torch.cat([
            x_0[:, :self.embed_dims_0, ...], x_1, x_2], dim=1)
        x = self.PW_conv(x)
        return x


class MultiOrderGatedAggregation(nn.Module):
    """Spatial Block with Multi-order Gated Aggregation.

    Args:
        embed_dims (int): Number of input channels.
        attn_dw_dilation (list): Dilations of three DWConv layers.
        attn_channel_split (list): The raletive ratio of splited channels.
        attn_act_type (str): The activation type for Spatial Block.
            Defaults to 'SiLU'.
    """

    def __init__(self,
                 embed_dims,
                 attn_dw_dilation=[1, 2, 3],
                 attn_channel_split=[1, 3, 4],
                 attn_act_type='SiLU',
                 attn_force_fp32=False,
                 ):
        super(MultiOrderGatedAggregation, self).__init__()

        self.embed_dims = embed_dims
        self.attn_force_fp32 = attn_force_fp32
        self.proj_1 = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)
        self.gate = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)
        self.value = MultiOrderDWConv(
            embed_dims=embed_dims,
            dw_dilation=attn_dw_dilation,
            channel_split=attn_channel_split,
        )
        self.proj_2 = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)

        # activation for gating and value
        self.act_value = build_act_layer(attn_act_type)
        self.act_gate = build_act_layer(attn_act_type)

        # decompose
        self.sigma = ElementScale(
            embed_dims, init_value=1e-5, requires_grad=True)

    # 这一块是论文中的FD
    def feat_decompose(self, x):
        x = self.proj_1(x)
        # x_d: [B, C, H, W] -> [B, C, 1, 1]
        x_d = F.adaptive_avg_pool2d(x, output_size=1)
        x = x + self.sigma(x - x_d)
        x = self.act_value(x)
        return x

    # 这是论文中Moga块中的Multi-Order Gated Aggregation
    def forward_gating(self, g, v):
        with torch.autocast(device_type='cuda', enabled=False):
            g = g.to(torch.float32)
            v = v.to(torch.float32)
            return self.proj_2(self.act_gate(g) * self.act_gate(v))

    def forward(self, x):
        shortcut = x.clone()
        # proj 1x1
        x = self.feat_decompose(x)
        # gating and value branch
        g = self.gate(x)
        v = self.value(x)
        # aggregation
        if not self.attn_force_fp32:
            x = self.proj_2(self.act_gate(g) * self.act_gate(v))
        else:
            x = self.forward_gating(self.act_gate(g), self.act_gate(v))
        x = x + shortcut
        return x


class MogaBlock(nn.Module):
    """A block of MogaNet.

    Args:
        embed_dims (int): Number of input channels.
        ffn_ratio (float): The expansion ratio of feedforward network hidden
            layer channels. Defaults to 4.
        drop_rate (float): Dropout rate after embedding. Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.1.
        act_type (str): The activation type for projections and FFNs.
            Defaults to 'GELU'.
        norm_cfg (str): The type of normalization layer. Defaults to 'BN'.
        init_value (float): Init value for Layer Scale. Defaults to 1e-5.
        attn_dw_dilation (list): Dilations of three DWConv layers.
        attn_channel_split (list): The raletive ratio of splited channels.
        attn_act_type (str): The activation type for the gating branch.
            Defaults to 'SiLU'.
    """

    def __init__(self,
                 embed_dims,
                 ffn_ratio=4.,  # 论文中的通道扩展系数r
                 drop_rate=0.,
                 drop_path_rate=0.,
                 act_type='GELU',
                 norm_type='BN',
                 init_value=1e-5,
                 attn_dw_dilation=[1, 2, 3],
                 attn_channel_split=[1, 3, 4],
                 attn_act_type='SiLU',
                 attn_force_fp32=False,
                 ):
        super(MogaBlock, self).__init__()
        self.out_channels = embed_dims

        self.norm1 = build_norm_layer(norm_type, embed_dims)

        # spatial attention
        self.attn = MultiOrderGatedAggregation(
            embed_dims,
            attn_dw_dilation=attn_dw_dilation,
            attn_channel_split=attn_channel_split,
            attn_act_type=attn_act_type,
            attn_force_fp32=attn_force_fp32,
        )
        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        self.norm2 = build_norm_layer(norm_type, embed_dims)

        # channel MLP
        mlp_hidden_dim = int(embed_dims * ffn_ratio)
        self.mlp = ChannelAggregationFFN(  # DWConv + Channel Aggregation FFN
            embed_dims=embed_dims,
            feedforward_channels=mlp_hidden_dim,
            act_type=act_type,
            ffn_drop=drop_rate,
        )

        # init layer scale
        self.layer_scale_1 = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)), requires_grad=True)

    def forward(self, x):
        # spatial
        identity = x
        x = self.layer_scale_1 * self.attn(self.norm1(x))
        x = identity + self.drop_path(x)
        # channel
        identity = x
        x = self.layer_scale_2 * self.mlp(self.norm2(x))
        x = identity + self.drop_path(x)
        return x


class ConvPatchEmbed(nn.Module):
    """An implementation of Conv patch embedding layer.

    Args:
        in_features (int): The feature dimension.
        embed_dims (int): The output dimension of PatchEmbed.
        kernel_size (int): The conv kernel size of PatchEmbed.
            Defaults to 3.
        stride (int): The conv stride of PatchEmbed. Defaults to 2.
        norm_type (str): The type of normalization layer. Defaults to 'BN'.
    """

    def __init__(self,
                 in_channels,
                 embed_dims,
                 kernel_size=3,
                 stride=2,
                 norm_type='BN'):
        super(ConvPatchEmbed, self).__init__()

        self.projection = nn.Conv2d(
            in_channels, embed_dims, kernel_size=kernel_size,
            stride=stride, padding=kernel_size // 2)
        self.norm = build_norm_layer(norm_type, embed_dims)

    def forward(self, x):
        x = self.projection(x)
        x = self.norm(x)
        out_size = (x.shape[2], x.shape[3])
        return x, out_size


class StackConvPatchEmbed(nn.Module):
    """An implementation of Stack Conv patch embedding layer.

    Args:
        in_features (int): The feature dimension.
        embed_dims (int): The output dimension of PatchEmbed.
        kernel_size (int): The conv kernel size of stack patch embedding.
            Defaults to 3.
        stride (int): The conv stride of stack patch embedding.
            Defaults to 2.
        act_type (str): The activation in PatchEmbed. Defaults to 'GELU'.
        norm_type (str): The type of normalization layer. Defaults to 'BN'.
    """

    def __init__(self,
                 in_channels,
                 embed_dims,
                 kernel_size=3,
                 stride=2,
                 act_type='GELU',
                 norm_type='BN'):
        super(StackConvPatchEmbed, self).__init__()

        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, embed_dims // 2, kernel_size=kernel_size,
                      stride=stride, padding=kernel_size // 2),
            build_norm_layer(norm_type, embed_dims // 2),
            build_act_layer(act_type),
            nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=kernel_size,
                      stride=stride, padding=kernel_size // 2),
            build_norm_layer(norm_type, embed_dims),
        )

    def forward(self, x):
        x = self.projection(x)
        out_size = (x.shape[2], x.shape[3])
        return x, out_size


class MogaNet(nn.Module):
    r""" MogaNet
        A PyTorch implement of : `Efficient Multi-order Gated Aggregation
        Network <https://arxiv.org/abs/2211.03295>`_

    Args:
        arch (str): MogaNet architecture choosing from 'tiny', 'small',
            'base' and 'large'. Defaults to 'tiny'.
        in_channels (int): The num of input channels. Defaults to 3.
        num_classes (int): The number of classes for linear classifier.
            Defaults to 1000.
        drop_rate (float): Dropout rate after embedding. Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.1.
        init_value (float): Init value for Layer Scale. Defaults to 1e-5.
        head_init_scale (float): Rescale init of classifier for high-resolution
            fine-tuning. Defaults to 1.
        patch_sizes (List[int | tuple]): The patch size in patch embeddings.
            Defaults to [3, 3, 3, 3].
        stem_norm_type (str): The type for normalization layer for stems.
            Defaults to 'BN'.
        conv_norm_type (str): The type for convolution normalization layer.
            Defaults to 'BN'.
        patchembed_types (list): The type of PatchEmbedding in each stage.
            Defaults to ``['ConvEmbed', 'Conv', 'Conv', 'Conv',]``.
        attn_dw_dilation (list): The dilate rate of depth-wise convolutions in
            Moga Blocks. Defaults to ``[1, 2, 3]``.
        attn_channel_split (list): The channel split rate of three depth-wise
            convolutions in Moga Blocks. Defaults to ``[1, 3, 4]``, i.e.,
            divided into ``[1/8, 3/8, 4/8]``.
        attn_act_cfg (dict): Config dict for activation of gating in Moga
            Blocks. Defaults to ``dict(type='SiLU')``.
        attn_final_dilation (bool): Whether to adopt dilated depth-wise
        attn_force_fp32 (bool): Whether to force the gating running with fp32.
            Warning: If you use `attn_force_fp32=False` during training, you
            should also keep it false during evaluation, because the output results
            of whether to use `attn_force_fp32` are different. We set it to false
            in this repo to facilitate code migration. Defaults to False.
        fork_feat (bool): Whether to output features of the 4 stages for dense
            prediction tasks in mmdetection and mmsegmentation. Defaults to False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Defaults to -1.
        init_cfg (dict): Init config dict for mmdetection and mmsegmentation
            to load pretrained weights. Defaults to None.
        pretrained (str): Pretrained path for mmdetection and mmsegmentation
            to load pretrained weights (old version). Defaults to None.
    """
    arch_zoo = {
        **dict.fromkeys(['xt', 'x-tiny', 'xtiny'],
                        {'embed_dims': [32, 64, 96, 192],
                         'depths': [3, 3, 10, 2],
                         'ffn_ratios': [8, 8, 4, 4]}),
        **dict.fromkeys(['t', 'tiny'],
                        {'embed_dims': [32, 64, 128, 256],
                         'depths': [3, 3, 12, 2],
                         'ffn_ratios': [8, 8, 4, 4]}),
        **dict.fromkeys(['s', 'small'],
                        {'embed_dims': [64, 128, 320, 512],
                         'depths': [2, 3, 12, 2],
                         'ffn_ratios': [8, 8, 4, 4]}),
        **dict.fromkeys(['b', 'base'],
                        {'embed_dims': [64, 160, 320, 512],
                         'depths': [4, 6, 22, 3],
                         'ffn_ratios': [8, 8, 4, 4]}),
        **dict.fromkeys(['l', 'large'],
                        {'embed_dims': [64, 160, 320, 640],
                         'depths': [4, 6, 44, 4],
                         'ffn_ratios': [8, 8, 4, 4]}),
        **dict.fromkeys(['xl', 'x-large', 'xlarge'],
                        {'embed_dims': [96, 192, 480, 960],
                         'depths': [6, 6, 44, 4],
                         'ffn_ratios': [8, 8, 4, 4]}),
    }  # yapf: disable

    def __init__(self,
                 arch='tiny',
                 in_channels=3,
                 num_classes=1000,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 init_value=1e-5,
                 head_init_scale=1.,
                 patch_sizes=[3, 3, 3, 3],
                 stem_norm_type='BN',
                 conv_norm_type='BN',
                 patchembed_types=['ConvEmbed', 'Conv', 'Conv', 'Conv', ],
                 attn_dw_dilation=[1, 2, 3],
                 attn_channel_split=[1, 3, 4],
                 attn_act_type='SiLU',
                 attn_final_dilation=True,
                 attn_force_fp32=False,
                 fork_feat=False,
                 frozen_stages=-1,
                 init_cfg=None,
                 pretrained=None,
                 **kwargs):
        super().__init__()

        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:
            essential_keys = {'embed_dims', 'depths', 'ffn_ratios'}
            assert isinstance(arch, dict) and set(arch) == essential_keys, \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch

        self.embed_dims = self.arch_settings['embed_dims']
        self.depths = self.arch_settings['depths']
        self.ffn_ratios = self.arch_settings['ffn_ratios']
        self.num_stages = len(self.depths)
        self.attn_force_fp32 = attn_force_fp32
        self.use_layer_norm = stem_norm_type == 'LN'
        assert len(patchembed_types) == self.num_stages
        self.fork_feat = fork_feat
        self.frozen_stages = frozen_stages

        total_depth = sum(self.depths)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]  # stochastic depth decay rule

        cur_block_idx = 0
        for i, depth in enumerate(self.depths):
            if i == 0 and patchembed_types[i] == "ConvEmbed":
                assert patch_sizes[i] <= 3
                patch_embed = StackConvPatchEmbed(
                    in_channels=in_channels,
                    embed_dims=self.embed_dims[i],
                    kernel_size=patch_sizes[i],
                    stride=patch_sizes[i] // 2 + 1,
                    act_type='GELU',
                    norm_type=conv_norm_type,
                )
            else:
                patch_embed = ConvPatchEmbed(
                    in_channels=in_channels if i == 0 else self.embed_dims[i - 1],
                    embed_dims=self.embed_dims[i],
                    kernel_size=patch_sizes[i],
                    stride=patch_sizes[i] // 2 + 1,
                    norm_type=conv_norm_type)

            if i == self.num_stages - 1 and not attn_final_dilation:
                attn_dw_dilation = [1, 2, 1]
            blocks = nn.ModuleList([
                MogaBlock(
                    embed_dims=self.embed_dims[i],
                    ffn_ratio=self.ffn_ratios[i],
                    drop_rate=drop_rate,
                    drop_path_rate=dpr[cur_block_idx + j],
                    norm_type=conv_norm_type,
                    init_value=init_value,
                    attn_dw_dilation=attn_dw_dilation,
                    attn_channel_split=attn_channel_split,
                    attn_act_type=attn_act_type,
                    attn_force_fp32=attn_force_fp32,
                ) for j in range(depth)
            ])
            cur_block_idx += depth
            norm = build_norm_layer(stem_norm_type, self.embed_dims[i])

            self.add_module(f'patch_embed{i + 1}', patch_embed)
            self.add_module(f'blocks{i + 1}', blocks)
            self.add_module(f'norm{i + 1}', norm)

        if self.fork_feat:
            self.head = nn.Identity()
        else:
            # Classifier head
            self.num_classes = num_classes
            self.head = nn.Linear(self.embed_dims[-1], num_classes) \
                if num_classes > 0 else nn.Identity()

            # init for classification
            self.apply(self._init_weights)
            self.head.weight.data.mul_(head_init_scale)
            self.head.bias.data.mul_(head_init_scale)

        self.init_cfg = copy.deepcopy(init_cfg)
        # load pre-trained model
        if self.fork_feat and (
                self.init_cfg is not None or pretrained is not None):
            self.init_weights(pretrained)

    def _init_weights(self, m):
        """ Init for timm image classification """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _freeze_stages(self):
        for i in range(0, self.frozen_stages + 1):
            # freeze patch embed
            m = getattr(self, f'patch_embed{i + 1}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
            # freeze blocks
            m = getattr(self, f'blocks{i + 1}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
            # freeze norm
            m = getattr(self, f'norm{i + 1}')
            m.eval()

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return dict()

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        outs = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f'patch_embed{i + 1}')
            blocks = getattr(self, f'blocks{i + 1}')
            norm = getattr(self, f'norm{i + 1}')

            x, hw_shape = patch_embed(x)
            for block in blocks:
                x = block(x)
            if self.use_layer_norm:
                x = x.flatten(2).transpose(1, 2)
                x = norm(x)
                x = x.reshape(-1, *hw_shape,
                              blocks.out_channels).permute(0, 3, 1, 2).contiguous()
            else:
                x = norm(x)
            if self.fork_feat:
                outs.append(x)

        if self.fork_feat:
            # output the features of four stages for dense prediction
            return outs
        else:
            # output only the last layer for image classification
            return x

    def forward_head(self, x):
        return self.head(x.mean(dim=[2, 3]))

    def forward(self, x):
        x = self.forward_features(x)
        if self.fork_feat:
            # for dense prediction
            return x
        else:
            # for image classification
            return self.forward_head(x)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224),
        'crop_pct': 0.90, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'moganet_xt': _cfg(crop_pct=0.9),
    'moganet_t': _cfg(crop_pct=0.9),
    'moganet_s': _cfg(crop_pct=0.9),
    'moganet_b': _cfg(crop_pct=0.9),
    'moganet_l': _cfg(crop_pct=0.9),
    'moganet_xl': _cfg(crop_pct=0.9),
}

model_urls = {
    "moganet_xtiny_1k": "https://github.com/Westlake-AI/MogaNet/releases/download/moganet-in1k-weights/moganet_xtiny_sz224_8xbs128_ep300.pth.tar",
    "moganet_xtiny_1k_sz256": "https://github.com/Westlake-AI/MogaNet/releases/download/moganet-in1k-weights/moganet_xtiny_sz256_8xbs128_ep300.pth.tar",
    "moganet_tiny_1k": "https://github.com/Westlake-AI/MogaNet/releases/download/moganet-in1k-weights/moganet_tiny_sz224_8xbs128_ep300.pth.tar",
    "moganet_tiny_1k_sz256": "https://github.com/Westlake-AI/MogaNet/releases/download/moganet-in1k-weights/moganet_tiny_sz256_8xbs128_ep300.pth.tar",
    "moganet_small_1k": "https://github.com/Westlake-AI/MogaNet/releases/download/moganet-in1k-weights/moganet_small_sz224_8xbs128_ep300.pth.tar",
    "moganet_base_1k": "https://github.com/Westlake-AI/MogaNet/releases/download/moganet-in1k-weights/moganet_base_sz224_8xbs128_ep300.pth.tar",
    "moganet_large_1k": "https://github.com/Westlake-AI/MogaNet/releases/download/moganet-in1k-weights/moganet_large_sz224_8xbs64_ep300.pth.tar",
    "moganet_xlarge_1k": "https://github.com/Westlake-AI/MogaNet/releases/download/moganet-in1k-weights/moganet_xlarge_sz224_8xbs64_ep300.pth.tar",
}


@register_model
def moganet_xtiny(pretrained=False, **kwargs):
    model = MogaNet(arch='x-tiny', **kwargs)
    model.default_cfg = default_cfgs['moganet_xt']
    if pretrained:
        url = model_urls['moganet_xtiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["state_dict"])
    return model


@register_model
def moganet_tiny(pretrained=False, **kwargs):
    model = MogaNet(arch='tiny', **kwargs)
    model.default_cfg = default_cfgs['moganet_t']
    if pretrained:
        url = model_urls['moganet_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        # 创建一个新的 state_dict，移除多余的前缀
        new_state_dict = {}
        # print(checkpoint)  #####################################
        for key in checkpoint["state_dict"]:
            # 移除 'backbone.' 前缀
            if key.startswith('backbone.'):
                new_key = key.replace('backbone.', '')
                new_state_dict[new_key] = checkpoint["state_dict"][key]
            elif key.startswith('head.'):  # 移除后面两个用于分类的全连接层
                continue
            else:
                new_state_dict[key] = checkpoint["state_dict"][key]
        # 加载新的 state_dict 到模型中
        model.load_state_dict(new_state_dict)
    return model


@register_model
def moganet_tiny_sz256(pretrained=False, **kwargs):
    model = MogaNet(arch='tiny', **kwargs)
    model.default_cfg = default_cfgs['moganet_t']
    if pretrained:
        url = model_urls['moganet_tiny_1k_sz256']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["state_dict"])
    return model


@register_model
def moganet_small(pretrained=False, **kwargs):
    model = MogaNet(arch='small', **kwargs)
    model.default_cfg = default_cfgs['moganet_s']
    if pretrained:
        url = model_urls['moganet_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["state_dict"])
    return model


@register_model
def moganet_base(pretrained=False, **kwargs):
    model = MogaNet(arch='base', **kwargs)
    model.default_cfg = default_cfgs['moganet_b']
    if pretrained:
        url = model_urls['moganet_base_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["state_dict"])
    return model


@register_model
def moganet_large(pretrained=False, **kwargs):
    model = MogaNet(arch='large', **kwargs)
    model.default_cfg = default_cfgs['moganet_l']
    if pretrained:
        url = model_urls['moganet_large_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["state_dict"])
    return model


@register_model
def moganet_xlarge(pretrained=False, **kwargs):
    model = MogaNet(arch='x-large', **kwargs)
    model.default_cfg = default_cfgs['moganet_xl']
    if pretrained:
        url = model_urls['moganet_xlarge_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["state_dict"])
    return model

class globalNet(nn.Module):
    def __init__(self, channel_settings, output_shape, num_class):
        super(globalNet, self).__init__()
        self.channel_settings = channel_settings
        laterals, upsamples, predict = [], [], []
        for i in range(len(channel_settings)):
            laterals.append(self._lateral(channel_settings[i]))
            predict.append(self._predict(output_shape, num_class))
            if i != len(channel_settings) - 1:
                upsamples.append(self._upsample())
        self.laterals = nn.ModuleList(laterals)
        self.upsamples = nn.ModuleList(upsamples)
        self.predict = nn.ModuleList(predict)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _lateral(self, input_size):
        layers = []
        layers.append(nn.Conv2d(input_size, 256,
            kernel_size=1, stride=1, bias=False))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def _upsample(self):
        layers = []
        layers.append(torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        layers.append(torch.nn.Conv2d(256, 256,
            kernel_size=1, stride=1, bias=False))
        layers.append(nn.BatchNorm2d(256))

        return nn.Sequential(*layers)

    def _predict(self, output_shape, num_class):
        layers = []
        layers.append(nn.Conv2d(256, 256,
            kernel_size=1, stride=1, bias=False))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU(inplace=True))

        # layers.append(nn.Conv2d(256, num_class,
        #     kernel_size=3, stride=1, padding=1, bias=False))
        # 使用深度可分离卷积
        layers.append(DepthwiseSeparableConv(in_channels=256, out_channels=num_class,kernel_size=3,padding=1))

        layers.append(nn.Upsample(size=output_shape, mode='bilinear', align_corners=True))
        layers.append(nn.BatchNorm2d(num_class))

        return nn.Sequential(*layers)

    def forward(self, x):
        global_fms, global_outs = [], []
        for i in range(len(self.channel_settings)):
            if i == 0:
                feature = self.laterals[i](x[i])
            else:
                feature = self.laterals[i](x[i]) + up
            global_fms.append(feature)
            if i != len(self.channel_settings) - 1:
                up = self.upsamples[i](feature)
            feature = self.predict[i](feature)
            global_outs.append(feature)

        return global_fms, global_outs


class DepthwiseConv(nn.Module):
    def __init__(self, in_channels, kernel_size, stride=1, padding=0, dilation=1):
        super(DepthwiseConv, self).__init__()
        self.depthwise_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,  # groups == in_channels 表示深度卷积
            bias=False
        )

    def forward(self, x):
        return self.depthwise_conv(x)


class PointwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PointwiseConv, self).__init__()
        self.pointwise_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False
        )

    def forward(self, x):
        return self.pointwise_conv(x)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = DepthwiseConv(
            in_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation
        )
        self.pointwise = PointwiseConv(
            in_channels=in_channels,
            out_channels=out_channels
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class cascade_globalNet(nn.Module):
    def __init__(self, num_cascade, channel_settings):
        super(cascade_globalNet, self).__init__()
        nets = []
        for i in range(num_cascade):
            if i != 0:
                channel_settings = [256, 256, 256, 256]
            nets.append(self.getGlobalNet(channel_settings))
        self.globalNet = nn.ModuleList(nets)

    def getGlobalNet(self, channel_settings, output_shape=(64, 48), num_class=17):
        return globalNet(channel_settings=channel_settings, output_shape=output_shape, num_class=num_class)

    def forward(self, x):
        global_fms, global_outs = [], []
        for i in range(len(self.globalNet)):
            fms, outs = self.globalNet[i](x)
            global_fms.append(fms)
            global_outs.append(outs)
            x = fms

        return global_fms, global_outs

class PoseMoganetUpsample(nn.Module):
    def __init__(self, cfg ,num_cascade=1, **kwargs):
        super().__init__()
        extra = cfg.MODEL.EXTRA
        self.num_joints = cfg.MODEL.NUM_JOINTS
        self.simcc_split_ratio = cfg.MODEL.SIMDR_SPLIT_RATIO
        self.moganet = moganet_tiny(pretrained=False,fork_feat=True,drop_path_rate=0.15,stem_norm_type='BN',conv_norm_type='BN')
        self.in_channel = 256  # globalNet输出的通道数(不使用predict层)
        self.input_size = cfg.MODEL.IMAGE_SIZE
        # weight_path= './moganet_t_coco_256x192.pth'
        # checkpoint = torch.load(weight_path,map_location='cpu')
        # # 创建一个新的 state_dict，移除多余的前缀
        # new_state_dict = {}
        # for key in checkpoint["state_dict"]:
        #     # 移除 'backbone.' 前缀
        #     if key.startswith('backbone.'):
        #         new_key = key.replace('backbone.', '')
        #         new_state_dict[new_key] = checkpoint["state_dict"][key]
        #     else:
        #         new_state_dict[key] = checkpoint["state_dict"][key]
        # # 加载新的 state_dict 到模型中
        # self.moganet.load_state_dict(new_state_dict,strict=False)

        channel_settings = [256, 128, 64, 32]
        self.globalNet = cascade_globalNet(num_cascade=num_cascade,channel_settings=channel_settings)

        self.final_layer = nn.Conv2d(
            in_channels=self.in_channel,
            out_channels=self.num_joints,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=extra.FINAL_CONV_KERNEL // 2
        )

        flatten_dims = cfg.MODEL.FEATURE_OUTPUT_SIZE[0] * cfg.MODEL.FEATURE_OUTPUT_SIZE[1]

        W = int(self.input_size[0] * self.simcc_split_ratio)
        H = int(self.input_size[1] * self.simcc_split_ratio)

        self.mlp_head_x = nn.Linear(flatten_dims, W, bias=False)
        self.mlp_head_y = nn.Linear(flatten_dims, H, bias=False)

    def forward(self, x):
        x = self.moganet(x)
        x.reverse()
        global_fms, _ = self.globalNet(x)
        x = global_fms[-1][-1]

        x = self.final_layer(x) # (bs,17,64,48)
        x = rearrange(x, 'b c h w -> b c (h w)')

        pred_x = self.mlp_head_x(x)
        pred_y = self.mlp_head_y(x)

        return pred_x, pred_y

    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            logger.info('=> init final conv weights from normal distribution')
            for m in self.final_layer.modules():
                if isinstance(m, nn.Conv2d):
                    logger.info('=> init final_layer.weight as normal(0, 0.001)')
                    logger.info('=> init final_layer.bias as 0')
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.constant_(m.bias, 0)
            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            self.load_state_dict(pretrained_state_dict, strict=False)
        else: # 因为加载了moganet的预训练训练模型所以就不随机初始化了
            logger.info('=> init weights from normal distribution')
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=.02)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)
                elif isinstance(m, nn.Conv2d):  # upsample int mina void
                    fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    fan_out //= m.groups
                    m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                    if m.bias is not None:
                        m.bias.data.zero_()

def get_pose_net(cfg, is_train, **kwargs):
    model = PoseMoganetUpsample(cfg, **kwargs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)
    return model



