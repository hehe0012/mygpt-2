import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HParams:
    n_vocab: int = 0
    n_ctx: int = 1024
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12
    n_window: int = 512
    moe_num_experts: int = 8
    moe_top_k: int = 1
    moe_capacity_factor: float = 1.25
    moe_layers: list = None

    def override_from_dict(self, values):
        for key, value in values.items():
            if hasattr(self, key):
                setattr(self, key, value)


def default_hparams():
    '''
    选择MoE层的默认设置,如果HParams.moe_layers为None,则默认在第4-9层使用MoE
    '''
    if HParams.moe_layers is None:
        default_layers = [4, 5, 6, 7, 8, 9]
    else:
        default_layers = HParams.moe_layers
    return HParams(moe_layers=default_layers)


def shape_list(x):
    return list(x.shape)


def softmax(x, axis=-1):
    return F.softmax(x, dim=axis)


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def split_states(x, n):
    '''
    将x的最后一个维度m重塑为[n, m / n]
    '''
    *start, m = shape_list(x)
    return x.view(*start, n, m // n)


def merge_states(x):
    '''
    将x的最后两个维度a和b合并为一个维度a * b
    '''
    *start, a, b = shape_list(x)
    return x.contiguous().view(*start, a * b)


def attention_mask(nd, ns, *, device, window_size):
    '''
    注意力机制的因果滑动窗口掩码
    生成一个形状为[nd, ns]的布尔张量
    仅允许每个查询位置关注其左侧窗口内的token
    '''
    if window_size is None or window_size <= 0:
        window_size = ns
    q_pos = torch.arange(ns - nd, ns, device=device)[:, None]
    k_pos = torch.arange(ns, device=device)[None, :]
    return (k_pos <= q_pos) and (k_pos >= (q_pos - window_size + 1))


def past_shape(*, hparams, batch_size=None, sequence=None):
    '''
    计算KV cache的形状
    '''
    return [batch_size, hparams.n_layer, 2, hparams.n_head, sequence, hparams.n_embd // hparams.n_head]


def expand_tile(value, size):
    '''
    将value扩展为一个新的维度，并在该维度上重复size次
    '''
    if not torch.is_tensor(value):
        value = torch.tensor(value)
    return value.unsqueeze(0).repeat(size, *([1] * value.dim()))


def positions_for(tokens, past_length):
    '''
    计算tokens的位置信息
    '''
    batch_size = tokens.shape[0]
    nsteps = tokens.shape[1]
    return expand_tile(past_length + torch.arange(nsteps, device=tokens.device), batch_size)


def _parse_layer_list(value):
    '''
    解析MoE层列表的字符串表示
    例如 "4-9,12" 表示第4到9层和第12层使用MoE, "4,6,8" 表示第4、6、8层使用MoE
    '''
    if value is None:
        return []
    if isinstance(value, list):
        return [int(v) for v in value]
    if isinstance(value, int):
        return [value]
    if not isinstance(value, str):
        return []
    layers = []
    for part in value.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            start, end = part.split('-', 1)
            layers.extend(list(range(int(start), int(end) + 1)))
        else:
            layers.append(int(part))
    return layers


class Conv1D(nn.Module):
    '''
    1D卷积层,用于将最后一个维度从nx转换为nf
    '''
    def __init__(self, nx, nf, w_init_stdev=0.02):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(nx, nf))
        self.bias = nn.Parameter(torch.zeros(nf))
        nn.init.normal_(self.weight, std=w_init_stdev)

    def forward(self, x):
        size_out = x.size()[:-1] + (self.weight.size(1),)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        return x.view(*size_out)


class Attention(nn.Module):
    '''
    多头自注意力机制
    '''
    def __init__(self, n_state, hparams):
        '''
        n_state: 注意力输出的维度，必须是hparams.n_embd的倍数
        hparams.n_head: 注意力头数
        '''
        super().__init__()
        if n_state % hparams.n_head != 0:
            raise ValueError('n_state must be divisible by n_head')
        self.n_head = hparams.n_head
        self.split_size = n_state
        self.scale = 1.0 / math.sqrt(n_state // hparams.n_head)
        self.window_size = hparams.n_window
        self.c_attn = Conv1D(n_state, n_state * 3)
        self.c_proj = Conv1D(n_state, n_state)

    def split_heads(self, x):
        '''
        将x的最后一个维度分割为n_head个头，并将头的维度放在前面
        '''
        return split_states(x, self.n_head).permute(0, 2, 1, 3)

    def merge_heads(self, x):
        '''
        将x的头维度和最后一个维度合并回原来的形状
        '''
        return merge_states(x.permute(0, 2, 1, 3))

    def multihead_attn(self, q, k, v):
        '''
        多头注意力的计算
        q, k, v的形状为[batch_size, n_head, sequence_length, head_dim]
        '''
        w = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # w是注意力权重Q*K^T/sqrt(d_k)
        nd, ns = w.shape[-2], w.shape[-1]
        # nd,ns分别是查询和键的序列长度,并计算注意力掩码
        b = attention_mask(nd, ns, device=w.device, window_size=self.window_size).view(1, 1, nd, ns)
        # 将注意力权重w中的无效位置设置为一个非常小的值，以便在softmax中被忽略
        w = w.masked_fill(~b, torch.finfo(w.dtype).min)
        w = softmax(w, axis=-1)
        return torch.matmul(w, v)

    def forward(self, x, past=None):
        '''
        多头注意力的前向过程
        '''
        # 将最后一维度从n_embd转换为3*n_embd,并分割为q,k,v
        c = self.c_attn(x)
        q, k, v = c.split(self.split_size, dim=2)
        q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)
        present = torch.stack([k, v], dim=1)

        # 如果有KV cache,将当前的k和v与之前的k和v连接起来
        if past is not None:
            pk, pv = past[:, 0], past[:, 1]
            k = torch.cat([pk, k], dim=-2)
            v = torch.cat([pv, v], dim=-2)

        # 计算多头注意力输出
        a = self.multihead_attn(q, k, v)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        return a, present


class MLP(nn.Module):
    '''
    前馈神经网络,包含一个隐藏层和一个输出层
    '''
    def __init__(self, n_state, hparams):
        super().__init__()
        nx = hparams.n_embd
        self.c_fc = Conv1D(nx, n_state)
        self.c_proj = Conv1D(n_state, nx)

    def forward(self, x):
        return self.c_proj(gelu(self.c_fc(x)))


class MoE(nn.Module):
    '''
    MoE层（Top-1 路由）
    '''
    def __init__(self, n_state, hparams):
        super().__init__()
        self.num_experts = hparams.moe_num_experts
        self.top_k = hparams.moe_top_k
        self.capacity_factor = hparams.moe_capacity_factor
        self.gate = nn.Linear(hparams.n_embd, self.num_experts, bias=False)
        self.experts = nn.ModuleList([MLP(n_state, hparams) for _ in range(self.num_experts)])

    def forward(self, x):
        if self.top_k != 1:
            raise ValueError('Only top-1 routing is supported in this MoE implementation.')
        batch, seq_len, hidden = x.shape
        x_flat = x.view(batch * seq_len, hidden)
        scores = F.softmax(self.gate(x_flat), dim=-1)
        expert_idx = torch.argmax(scores, dim=-1)
        expert_score = scores.gather(1, expert_idx.unsqueeze(1)).squeeze(1)

        tokens = x_flat.shape[0]
        capacity = int(math.ceil(self.capacity_factor * tokens / self.num_experts))
        output = torch.zeros_like(x_flat)

        for expert_id, expert in enumerate(self.experts):
            mask = expert_idx == expert_id
            indices = mask.nonzero(as_tuple=False).squeeze(1)
            if indices.numel() == 0:
                continue
            if indices.numel() > capacity:
                indices = indices[:capacity]
            expert_out = expert(x_flat[indices])
            output[indices] = expert_out * expert_score[indices].unsqueeze(1)

        return output.view(batch, seq_len, hidden)


class Block(nn.Module):
    '''
    Transformer块,包含一个注意力层和一个前馈神经网络层
    '''
    def __init__(self, hparams, layer_idx):
        super().__init__()
        nx = hparams.n_embd
        # 标准化层
        self.ln_1 = nn.LayerNorm(nx, eps=1e-5)
        # 多头注意力层
        self.attn = Attention(nx, hparams)
        # 标准化层
        self.ln_2 = nn.LayerNorm(nx, eps=1e-5)
        # 前馈神经网络层
        moe_layers = set(_parse_layer_list(hparams.moe_layers))
        if layer_idx in moe_layers:
            self.mlp = MoE(nx * 4, hparams)
        else:
            self.mlp = MLP(nx * 4, hparams)

    def forward(self, x, past=None):
        a, present = self.attn(self.ln_1(x), past=past)
        # 将注意力输出a与输入x相加
        x = x + a
        # 将前馈神经网络的输出与当前的x相加
        x = x + self.mlp(self.ln_2(x))
        return x, present


class GPT2Model(nn.Module):
    '''
    GPT-2模型的实现
    '''
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.wpe = nn.Embedding(hparams.n_ctx, hparams.n_embd)
        self.wte = nn.Embedding(hparams.n_vocab, hparams.n_embd)
        self.h = nn.ModuleList([Block(hparams, layer_idx) for layer_idx in range(hparams.n_layer)])
        self.ln_f = nn.LayerNorm(hparams.n_embd, eps=1e-5)
        nn.init.normal_(self.wpe.weight, std=0.01)
        nn.init.normal_(self.wte.weight, std=0.02)

    def forward(self, X, past=None):
        '''
        GPT-2模型的前向过程
        '''
        # 获取输入的批量大小和序列长度
        batch, sequence = shape_list(X)
        past_length = 0 if past is None else past.shape[-2]
        h = self.wte(X) + self.wpe(positions_for(X, past_length))

        # 收集每层的KV cache
        presents = []
        pasts = list(torch.unbind(past, dim=1)) if past is not None else [None] * self.hparams.n_layer
        if len(pasts) != self.hparams.n_layer:
            raise ValueError('past layers do not match hparams.n_layer')
        
        # 遍历每个Transformer块,并将当前的h和对应的KV cache传入块中
        for layer, layer_past in enumerate(pasts):
            h, present = self.h[layer](h, past=layer_past)
            presents.append(present)

        # 将每层的KV cache堆叠成一个张量,然后标准化
        present = torch.stack(presents, dim=1)
        h = self.ln_f(h)

        # 将h展平为二维张量,并与词嵌入矩阵的转置相乘,得到logits
        h_flat = h.view(batch * sequence, self.hparams.n_embd)
        logits = torch.matmul(h_flat, self.wte.weight.t()).view(batch, sequence, self.hparams.n_vocab)
        return {
            'present': present,
            'logits': logits,
        }


def model(hparams, X, past=None, scope='model', reuse=False, module=None):
    del scope, reuse
    if module is None:
        module = GPT2Model(hparams)
    return module(X, past=past)
