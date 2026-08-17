# Copyright (c) OpenMMLab. All rights reserved.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks import DropPath
from mmengine.utils import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION

class ScaleNorm(nn.Module):
    """Scale Norm.
    Args:
        dim (int): The dimension of the scale vector.
        eps (float, optional): The minimum value in clamp. Defaults to 1e-5.
    Reference:
        `Transformers without Tears: Improving the Normalization
        of Self-Attention <https://arxiv.org/abs/1910.05895>`_
    """

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.scale = dim**-0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        """Forward function.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: The tensor after applying scale norm.
        """

        if torch.onnx.is_in_onnx_export() and \
                digit_version(TORCH_VERSION) >= digit_version('1.12'):

            norm = torch.linalg.norm(x, dim=-1, keepdim=True)

        else:
            norm = torch.norm(x, dim=-1, keepdim=True)
        norm = norm * self.scale
        return x / norm.clamp(min=self.eps) * self.g


def rope(x, dim):
    """Applies Rotary Position Embedding to input tensor.
    Args:
        x (torch.Tensor): Input tensor.
        dim (int | list[int]): The spatial dimension(s) to apply
            rotary position embedding.
    Returns:
        torch.Tensor: The tensor after applying rotary position
            embedding.
    Reference:
        `RoFormer: Enhanced Transformer with Rotary
        Position Embedding <https://arxiv.org/abs/2104.09864>`_
    """
    shape = x.shape
    if isinstance(dim, int):
        dim = [dim]

    spatial_shape = [shape[i] for i in dim]
    total_len = 1
    for i in spatial_shape:
        total_len *= i

    position = torch.reshape(
        torch.arange(total_len, dtype=torch.int, device=x.device),
        spatial_shape)

    for i in range(dim[-1] + 1, len(shape) - 1, 1):
        position = torch.unsqueeze(position, dim=-1)

    half_size = shape[-1] // 2
    freq_seq = -torch.arange(
        half_size, dtype=torch.int, device=x.device) / float(half_size)
    inv_freq = 10000**-freq_seq

    sinusoid = position[..., None] * inv_freq[None, None, :]

    sin = torch.sin(sinusoid)
    cos = torch.cos(sinusoid)
    x1, x2 = torch.chunk(x, 2, dim=-1)

    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class Scale(nn.Module):
    """Scale vector by element multiplications.
    Args:
        dim (int): The dimension of the scale vector.
        init_value (float, optional): The initial value of the scale vector.
            Defaults to 1.0.
        trainable (bool, optional): Whether the scale vector is trainable.
            Defaults to True.
    """

    def __init__(self, dim, init_value=1., trainable=True):
        super().__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        """Forward function."""

        return x * self.scale

# gau的配置(rtmpose-m)
gau_cfg = dict(
    hidden_dims=256,
    s=128,
    expansion_factor=2,
    dropout_rate=0.,
    drop_path=0.,
    act_fn='ReLU',
    use_rel_bias=False,
    pos_enc=False)

class RTMCCBlock(nn.Module):
    """Gated Attention Unit (GAU) in RTMBlock.
    Args:
        num_token (int): The number of tokens.
        in_token_dims (int): The input token dimension.
        out_token_dims (int): The output token dimension.
        expansion_factor (int, optional): The expansion factor of the
            intermediate token dimension. Defaults to 2.
        s (int, optional): The self-attention feature dimension.
            Defaults to 128.
        eps (float, optional): The minimum value in clamp. Defaults to 1e-5.
        dropout_rate (float, optional): The dropout rate. Defaults to 0.0.
        drop_path (float, optional): The drop path rate. Defaults to 0.0.
        attn_type (str, optional): Type of attention which should be one of
            the following options:

            - 'self-attn': Self-attention.
            - 'cross-attn': Cross-attention.

            Defaults to 'self-attn'.
        act_fn (str, optional): The activation function which should be one
            of the following options:

            - 'ReLU': ReLU activation.
            - 'SiLU': SiLU activation.

            Defaults to 'SiLU'.
        bias (bool, optional): Whether to use bias in linear layers.
            Defaults to False.
        use_rel_bias (bool, optional): Whether to use relative bias.
            Defaults to True.
        pos_enc (bool, optional): Whether to use rotary position
            embedding. Defaults to False.
    Reference:
        `Transformer Quality in Linear Time
        <https://arxiv.org/abs/2202.10447>`_
    """

    def __init__(self,
                 num_token,
                 in_token_dims,
                 out_token_dims,
                 expansion_factor=2,
                 s=128,
                 eps=1e-5,
                 dropout_rate=0.,
                 drop_path=0.,
                 attn_type='self-attn',
                 act_fn='SiLU',
                 bias=False,
                 use_rel_bias=True,
                 pos_enc=False):
        super(RTMCCBlock, self).__init__()
        # s: 注意力头的维度。
        self.s = s
        # num_token: 输入序列中的token数量。
        self.num_token = num_token
        # use_rel_bias: 是否使用相对位置偏差
        self.use_rel_bias = use_rel_bias
        # attn_type: 注意力机制的类型（'self-attn'或'cross-attn'）。
        self.attn_type = attn_type
        # pos_enc: 是否使用位置编码。
        self.pos_enc = pos_enc
        # drop_path: DropPath层，用于随机丢弃网络中的路径，以提高泛化能力。
        self.drop_path = DropPath(drop_path) \
            if drop_path > 0. else nn.Identity()

        # e: 扩展后token的中间维度。
        self.e = int(in_token_dims * expansion_factor)
        # 定义相对位置偏差参数
        if use_rel_bias:
            if attn_type == 'self-attn':
                # w: self-attention的相对位置偏差参数。
                self.w = nn.Parameter(
                    torch.rand([2 * num_token - 1], dtype=torch.float))
            else:
                # a, b: cross-attention的相对位置偏差参数。
                self.a = nn.Parameter(torch.rand([1, s], dtype=torch.float))
                self.b = nn.Parameter(torch.rand([1, s], dtype=torch.float))
        # o: 线性层，用于将注意力机制的输出投影到输出维度。
        self.o = nn.Linear(self.e, out_token_dims, bias=bias)

        # 定义注意力机制参数
        if attn_type == 'self-attn':
            # uv: 线性层，用于将输入投影到query, value和base向量。
            self.uv = nn.Linear(in_token_dims, 2 * self.e + self.s, bias=bias)
            # gamma, beta: 用于旋转位置编码的参数。
            self.gamma = nn.Parameter(torch.rand((2, self.s)))
            self.beta = nn.Parameter(torch.rand((2, self.s)))
        else:
            # uv: 线性层，用于将输入投影到query和value向量。
            self.uv = nn.Linear(in_token_dims, self.e + self.s, bias=bias)
            # k_fc, v_fc: 线性层，用于投影key和value向量。
            self.k_fc = nn.Linear(in_token_dims, self.s, bias=bias)
            self.v_fc = nn.Linear(in_token_dims, self.e, bias=bias)
            nn.init.xavier_uniform_(self.k_fc.weight)
            nn.init.xavier_uniform_(self.v_fc.weight)

        # ln: Layer Normalization层。
        self.ln = ScaleNorm(in_token_dims, eps=eps)

        nn.init.xavier_uniform_(self.uv.weight)

        # 定义激活函数
        if act_fn == 'SiLU' or act_fn == nn.SiLU:
            assert digit_version(TORCH_VERSION) >= digit_version('1.7.0'), \
                'SiLU activation requires PyTorch version >= 1.7'
            self.act_fn = nn.SiLU(True)
        elif act_fn == 'ReLU' or act_fn == nn.ReLU:
            self.act_fn = nn.ReLU(True)
        else:
            raise NotImplementedError

        # shortcut: 是否使用shortcut连接。
        if in_token_dims == out_token_dims:
            self.shortcut = True
            self.res_scale = Scale(in_token_dims)
        else:
            self.shortcut = False

        # sqrt_s: 注意力头维度的平方根。
        self.sqrt_s = math.sqrt(s)

        self.dropout_rate = dropout_rate
        if dropout_rate > 0.:
            self.dropout = nn.Dropout(dropout_rate)

    def rel_pos_bias(self, seq_len, k_len=None):
        """Add relative position bias.

        Args:
            seq_len (int): query的序列长度。
            k_len (int, optional): key的序列长度。如果为None，则使用seq_len。默认为None。

        Returns:
            Tensor: 相对位置偏差张量。
        """
        if self.attn_type == 'self-attn':
            # t: self-attention的相对位置偏差张量。
            t = F.pad(self.w[:2 * seq_len - 1], [0, seq_len]).repeat(seq_len)
            t = t[..., :-seq_len].reshape(-1, seq_len, 3 * seq_len - 2)
            r = (2 * seq_len - 1) // 2
            t = t[..., r:-r]
        else:
            # a, b: cross-attention的旋转位置编码。
            a = rope(self.a.repeat(seq_len, 1), dim=0)
            b = rope(self.b.repeat(k_len, 1), dim=0)
            # t: cross-attention的相对位置偏差张量。
            t = torch.bmm(a, b.permute(0, 2, 1))
        return t

    def _forward(self, inputs):
        """GAU Forward function.

        Args:
            inputs (Tensor or Tuple[Tensor]): 输入张量或输入张量的元组。
                如果``attn_type``是'self-attn'，则输入是形状为
                [B, K, in_token_dims]的单个张量。
                如果``attn_type``是'cross-attn'，则输入是三个张量的元组：
                query, key和value，每个的形状为[B, K, in_token_dims]。

        Returns:
            Tensor: 形状为[B, K, out_token_dims]的输出张量。
        """
        if self.attn_type == 'self-attn':
            x = inputs
        else:
            x, k, v = inputs

        x = self.ln(x)

        # uv: 输入的线性投影。
        # [B, K, in_token_dims] -> [B, K, e + e + s] 或 [B, K, e + s]
        uv = self.uv(x)
        uv = self.act_fn(uv)

        if self.attn_type == 'self-attn':
            # u, v, base: 将投影后的输入分成query, value和base向量。
            # [B, K, e + e + s] -> [B, K, e], [B, K, e], [B, K, s]
            u, v, base = torch.split(uv, [self.e, self.e, self.s], dim=2)
            # base: 对base向量应用旋转位置编码。
            # [B, K, 1, s] * [1, 1, 2, s] + [2, s] -> [B, K, 2, s]
            base = base.unsqueeze(2) * self.gamma[None, None, :] + self.beta

            if self.pos_enc:
                base = rope(base, dim=1)
            # q, k: 将base向量分成query和key向量。
            # [B, K, 2, s] -> [B, K, s], [B, K, s]
            q, k = torch.unbind(base, dim=2)
        else:
            # u, q: 将投影后的输入分成value和query向量。
            # [B, K, e + s] -> [B, K, e], [B, K, s]
            u, q = torch.split(uv, [self.e, self.s], dim=2)
            # k, v: 投影key和value向量。
            k = self.k_fc(k)  # -> [B, K, s]
            v = self.v_fc(v)  # -> [B, K, e]

            if self.pos_enc:
                q = rope(q, 1)
                k = rope(k, 1)

        # qk: 注意力矩阵。
        # [B, K, s].permute() -> [B, s, K]
        # [B, K, s] x [B, s, K] -> [B, K, K]
        qk = torch.bmm(q, k.permute(0, 2, 1))

        if self.use_rel_bias:
            # bias: 相对位置偏差。
            bias = self.rel_pos_bias(q.size(1), k_len=k.size(1)) if self.attn_type == 'cross-attn' else self.rel_pos_bias(q.size(1))
            qk += bias[:, :q.size(1), :k.size(1)]
        # kernel: Gated attention kernel.
        # [B, K, K]
        kernel = torch.square(F.relu(qk / self.sqrt_s))

        if self.dropout_rate > 0.:
            kernel = self.dropout(kernel)
        # x: 应用注意力和门控机制到value。
        # [B, K, K] x [B, K, e] -> [B, K, e]
        x = u * torch.bmm(kernel, v)
        # x: 投影输出。
        # [B, K, e] -> [B, K, out_token_dims]
        x = self.o(x)

        return x

    def forward(self, x):
        """Forward function."""
        if self.shortcut:
            # res_shortcut: shortcut连接。
            res_shortcut = x[0] if self.attn_type == 'cross-attn' else x
            # main_branch: main branch的输出。
            main_branch = self.drop_path(self._forward(x))
            return self.res_scale(res_shortcut) + main_branch
        else:
            return self.drop_path(self._forward(x))
