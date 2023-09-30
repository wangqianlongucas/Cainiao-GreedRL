import math
import torch
import torch.nn.functional as F

from torch import nn
from torch.utils.checkpoint import checkpoint


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, hidden_dim):
        super(MultiHeadAttention, self).__init__()

        assert hidden_dim % heads == 0

        self.heads = heads
        head_dim = hidden_dim // heads
        self.alpha = 1 / math.sqrt(head_dim)

        self.nn_Q = nn.Parameter(torch.Tensor(heads, hidden_dim, head_dim))
        self.nn_O = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, K, V, mask):
        batch_size, query_num, hidden_dim = q.size()

        size = (self.heads, batch_size, query_num, -1)

        q = q.reshape(-1, hidden_dim)
        Q = torch.matmul(q, self.nn_Q).view(size)

        value_num = V.size(2)
        heads_batch = self.heads * batch_size
        Q = Q.view(heads_batch, query_num, -1)
        K = K.view(heads_batch, value_num, -1).transpose(1, 2)

        S = masked_tensor(mask, self.heads)
        S = S.view(heads_batch, query_num, value_num)
        S.baddbmm_(Q, K, alpha=self.alpha)
        S = S.view(self.heads, batch_size, query_num, value_num)

        S = F.softmax(S, dim=-1)

        x = torch.matmul(S, V).permute(1, 2, 0, 3)
        x = x.reshape(batch_size, query_num, -1)
        x = torch.matmul(x, self.nn_O)
        return x


class Decode(nn.Module):

    def __init__(self, nn_args):
        super(Decode, self).__init__()

        self.nn_args = nn_args

        heads = nn_args['decode_atten_heads']
        hidden_dim = nn_args['decode_hidden_dim']

        self.heads = heads
        self.alpha = 1 / math.sqrt(hidden_dim)

        if heads > 0:
            assert hidden_dim % heads == 0
            head_dim = hidden_dim // heads
            self.nn_K = nn.Parameter(torch.Tensor(heads, hidden_dim, head_dim))
            self.nn_V = nn.Parameter(torch.Tensor(heads, hidden_dim, head_dim))
            self.nn_mha = MultiHeadAttention(heads, hidden_dim)

        decode_rnn = nn_args.setdefault('decode_rnn', 'LSTM')
        assert decode_rnn in ('GRU', 'LSTM', 'NONE')
        if decode_rnn == 'GRU':
            self.nn_rnn_cell = nn.GRUCell(hidden_dim, hidden_dim)
        elif decode_rnn == 'LSTM':
            self.nn_rnn_cell = nn.LSTMCell(hidden_dim, hidden_dim)
        else:
            self.nn_rnn_cell = None

        self.vars_dim = sum(nn_args['variable_dim'].values())
        if self.vars_dim > 0:
            atten_type = nn_args.setdefault('decode_atten_type', 'add')
            assert atten_type == 'add', "must be addition attention when vars_dim > 0, {}".format(atten_type)
            self.nn_A = nn.Parameter(torch.Tensor(self.vars_dim, hidden_dim))
            self.nn_B = nn.Parameter(torch.Tensor(hidden_dim))
        else:
            atten_type = nn_args.setdefault('decode_atten_type', 'prod')

        if atten_type == 'add':
            self.nn_W = nn.Parameter(torch.Tensor(hidden_dim))
        else:
            self.nn_W = None

        for param in self.parameters():
            stdv = 1 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, X, K, V, query, state1, state2, varfeat, mask, chosen, sample_p, clip, mode, memopt=0):
        if self.training and memopt > 2:
            state1, state2 = checkpoint(self.rnn_step, query, state1, state2)
        else:
            state1, state2 = self.rnn_step(query, state1, state2)

        query = state1
        NP = X.size(0)
        NR = query.size(0) // NP
        batch_size = query.size(0)
        if self.heads > 0:
            query = query.view(NP, NR, -1)
            if self.training and memopt > 1:
                query = checkpoint(self.nn_mha, query, K, V, mask)
            else:
                query = self.nn_mha(query, K, V, mask)

            query = query.view(batch_size, -1)

        if self.nn_W is None:
            query = query.view(NP, NR, -1)
            logit = masked_tensor(mask, 1)
            logit = logit.view(NP, NR, -1)
            X = X.permute(0, 2, 1)
            logit.baddbmm_(query, X, alpha=self.alpha)
            logit = logit.view(batch_size, -1)
        else:
            if self.training and self.vars_dim > 0 and memopt > 0:
                logit = checkpoint(self.atten, query, X, varfeat, mask)
            else:
                logit = self.atten(query, X, varfeat, mask)

        chosen_p = choose(logit, chosen, sample_p, clip, mode)
        return state1, state2, chosen_p

    def rnn_step(self, query, state1, state2):
        if isinstance(self.nn_rnn_cell, nn.GRUCell):
            state1 = self.nn_rnn_cell(query, state1)
        elif isinstance(self.nn_rnn_cell, nn.LSTMCell):
            state1, state2 = self.nn_rnn_cell(query, (state1, state2))
        return state1, state2

    def atten(self, query, keyvalue, varfeat, mask):
        if self.vars_dim > 0:
            varfeat = vfaddmm(varfeat, mask, self.nn_A, self.nn_B)
        return atten(query, keyvalue, varfeat, mask, self.nn_W)


def choose(logit, chosen, sample_p, clip, mode):
    mask = logit == -math.inf
    logit = torch.tanh(logit) * clip
    logit[mask] = -math.inf

    if mode == 0:
        pass
    elif mode == 1:
        chosen[:] = logit.argmax(1)
    elif mode == 2:
        p = logit.exp()
        chosen[:] = torch.multinomial(p, 1).squeeze(1)
    else:
        raise Exception()

    logp = logit.log_softmax(1)
    logp = logp.gather(1, chosen[:, None])
    logp = logp.squeeze(1)
    return logp


def atten(query, keyvalue, varfeat, mask, weight):
    batch_size = query.size(0)
    NP, NK, ND = keyvalue.size()

    query = query.view(NP, -1, 1, ND)
    varfeat = varfeat.view(NP, -1, NK, ND)
    keyvalue = keyvalue[:, None, :, :]
    keyvalue = keyvalue + varfeat + query
    keyvalue = torch.tanh(keyvalue)
    keyvalue = keyvalue.view(-1, ND)

    logit = masked_tensor(mask, 1).view(-1)
    logit.addmv_(keyvalue, weight)
    return logit.view(batch_size, -1)


def masked_tensor(mask, heads):
    size = list(mask.size())
    size.insert(0, heads)
    mask = mask[None].expand(size)
    result = mask.new_zeros(size, dtype=torch.float32)
    result[mask] = -math.inf
    return result


def vfaddmm(varfeat, mask, A, B):
    varfeat = varfeat.permute(0, 2, 1)
    return F.linear(varfeat, A.permute(1, 0), B)

