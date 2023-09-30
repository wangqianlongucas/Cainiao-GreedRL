import math
import torch
import torch.nn.functional as F

from torch import nn
from torch.utils.checkpoint import checkpoint
from .norm import Norm1D, Norm2D
from .dense import Dense
from .utils import repeat
from .feature import *


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, hidden_dim):
        super(MultiHeadAttention, self).__init__()
        
        assert hidden_dim % heads == 0

        self.heads = heads
        head_dim = hidden_dim // heads
        self.alpha = 1 / math.sqrt(head_dim)

        self.nn_Q = nn.Parameter(torch.Tensor(heads, hidden_dim, head_dim))
        self.nn_K = nn.Parameter(torch.Tensor(heads, hidden_dim, head_dim))
        self.nn_V = nn.Parameter(torch.Tensor(heads, hidden_dim, head_dim))
        self.nn_O = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, x, edge):
        batch_size, item_num, hidden_dim = x.size()
        size = (self.heads, batch_size, item_num, -1)
        
        x = x.reshape(-1, hidden_dim)
        Q = torch.matmul(x, self.nn_Q).view(size)
        K = torch.matmul(x, self.nn_K).view(size)
        V = torch.matmul(x, self.nn_V).view(size)

        heads_batch = self.heads * batch_size
        Q = Q.view(heads_batch, item_num, -1)
        K = K.view(heads_batch, item_num, -1).transpose(1, 2)
        
        if edge is not None:
            S = edge.view(heads_batch, item_num, item_num)
            S = S.baddbmm(Q, K, alpha=self.alpha)
        else:
            S = Q.new_zeros(heads_batch, item_num, item_num)
            S = S.baddbmm_(Q, K, alpha=self.alpha)

        S = S.view(self.heads, batch_size, item_num, item_num)

        S = F.softmax(S, dim=-1)

        x = torch.matmul(S, V).permute(1, 2, 0, 3)
        x = x.reshape(batch_size, item_num, -1)
        x = torch.matmul(x, self.nn_O)
        return x


class Encode(nn.Module):
    def __init__(self, nn_args):
        super(Encode, self).__init__()

        self.nn_args = nn_args
        self.worker_dim = nn_args['worker_dim']
        self.task_dim = nn_args['task_dim']
        self.edge_dim = nn_args['edge_dim']

        self.embed_dict = nn_args['embed_dict']
        self.feature_dict = nn_args['feature_dict']

        layers = nn_args.setdefault('encode_layers', 3)
        heads = nn_args.setdefault('encode_atten_heads', 8)
        norm = nn_args.setdefault('encode_norm', 'instance')
        hidden_dim = nn_args.setdefault('encode_hidden_dim', 128)
        output_dim = nn_args.setdefault('decode_hidden_dim', 128)
        output_heads = nn_args.setdefault('decode_atten_heads', 0)
        
        self.heads = heads
        self.layers = layers

        worker_dim = max(1, sum(self.worker_dim.values()))
        task_dim = max(1, sum(self.task_dim.values()))

        self.nn_dense_worker_start = Dense(worker_dim, hidden_dim)
        self.nn_dense_worker_end = Dense(worker_dim, hidden_dim)
        self.nn_dense_task = Dense(task_dim, hidden_dim)
        
        self.nn_norm_worker_task = Norm1D(hidden_dim, norm, True)

        if len(self.edge_dim) > 0:
            edge_dim = sum(self.edge_dim.values())
            self.nn_dense_edge = Dense(edge_dim, heads)
            self.nn_norm_edge = Norm2D(heads, norm, True)

        nn_embed_dict = {}
        for k, v in self.embed_dict.items():
            nn_embed_dict[k] = nn.Embedding(v, hidden_dim)
        self.nn_embed_dict = nn.ModuleDict(nn_embed_dict)

        self.nn_attens = nn.ModuleList()
        self.nn_denses = nn.ModuleList()
        self.nn_norms1 = nn.ModuleList()
        self.nn_norms2 = nn.ModuleList()
        for i in range(layers):
            self.nn_attens.append(MultiHeadAttention(heads, hidden_dim))
            self.nn_denses.append(nn.Sequential(
                                    Dense(hidden_dim, hidden_dim * 4),
                                    Dense(hidden_dim * 4, hidden_dim, act='relu'),
                                    ))
            self.nn_norms1.append(Norm1D(hidden_dim, norm, True))
            self.nn_norms2.append(Norm1D(hidden_dim, norm, True))

        self.nn_finish = nn.Parameter(torch.Tensor(1, 1, hidden_dim))        
        
        if output_dim != hidden_dim:
            self.nn_X = nn.Parameter(torch.Tensor(hidden_dim, output_dim))
        else:
            self.nn_X = None

        if output_heads > 0:
            assert output_dim % output_heads == 0
            head_dim = output_dim // output_heads
            self.nn_K = nn.Parameter(torch.Tensor(heads, hidden_dim, head_dim))
            self.nn_V = nn.Parameter(torch.Tensor(heads, hidden_dim, head_dim))
        else:
            self.nn_K = None
            self.nn_V = None

        for param in self.parameters():
            stdv = 1 / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, problem, batch_size, worker_num, task_num, memopt=0):
        worker_start, worker_end = self.encode_worker(problem, batch_size, worker_num)
        task = self.encode_task(problem, batch_size, task_num)
        X = torch.cat([worker_start, worker_end, task], 1)
        X = self.nn_norm_worker_task(X)

        if len(self.edge_dim) > 0:
            edge = self.encode_edge(problem, batch_size, worker_num, task_num)
            edge = self.nn_norm_edge(edge)
            edge = edge.permute(3, 0, 1, 2).contiguous()
        else:
            edge = None
        
        #transformer encoding
        for i in range(self.layers):
            X = self.encode_layer(X, edge, i, memopt)

        finish = repeat(self.nn_finish, X.size(0))
        X = torch.cat([X, finish], 1)
        if self.nn_X is not None:
            X = torch.matmul(X, self.nn_X)
        
        if self.nn_K is not None:
            batch_size, item_num, hidden_dim = X.size()
            size = (self.heads, batch_size, item_num, -1)
            X2 = X.reshape(-1, hidden_dim)
            K = torch.matmul(X2, self.nn_K).view(size)
            V = torch.matmul(X2, self.nn_V).view(size)
        else:
            K = torch.ones(0)
            V = torch.ones(0)
        return X, K, V
    
    def encode_layer(self, X, edge, i, memopt):
        run_fn = self.encode_layer_fn(i, memopt)
        if self.training and memopt > 6:
            return checkpoint(run_fn, X, edge)
        else:
            return run_fn(X, edge)

    def encode_layer_fn(self, i, memopt):
        def run_fn(X, edge):
            if self.training and memopt == 6:
                X = X + checkpoint(self.nn_attens[i], X, edge)
            else:
                X = X + self.nn_attens[i](X, edge)
            X = self.nn_norms1[i](X)

            X = X + self.nn_denses[i](X)
            X = self.nn_norms2[i](X)
            return X

        return run_fn
    
    def encode_worker(self, problem, batch_size, worker_num):
        feature_list = []
        for k, dim in self.worker_dim.items():
            f = self.feature_dict.get(k)
            if isinstance(f, GlobalCategory):
                v = problem[f.name]
                v = self.nn_embed_dict[k](v.long())
            elif isinstance(f, ContinuousFeature):
                v = problem[f.name]
            else:
                raise Exception("unsupported feature type: {}".format(type(f)))

            if v.dim() == 2:
                v = v[:, :, None]

            assert dim == v.size(-1), \
                "feature dim error, feature: {}, expected: {}, actual: {}".format(k, dim, v.size(-1))

            feature_list.append(v.float())

        if feature_list:
            x = torch.cat(feature_list, 2)
        else:
            x = self.nn_finish.new_ones(batch_size, worker_num, 1)
        return self.nn_dense_worker_start(x), self.nn_dense_worker_end(x)

    def encode_task(self, problem, batch_size, task_num):
        feature_list = []
        for k, dim in self.task_dim.items():
            f = self.feature_dict.get(k)
            if isinstance(f, SparseLocalFeature):
                v = problem[f.value]
                assert v.dim() == 3, \
                    "sparse local feature's dimension must 2, feature:{}".format(k)
                v = v.clamp(0, 1).sum(2, dtype=v.dtype)
            elif isinstance(f, GlobalCategory):
                v = problem[f.name]
                v = self.nn_embed_dict[k](v.long())
            elif isinstance(f, LocalFeature):
                v = problem[f.name]
                assert v.dim() == 3, \
                    "local feature's dimension must 2, feature:{}".format(k)
                v = v.clamp(0, 1).sum(2, dtype=v.dtype)
            elif isinstance(f, ContinuousFeature):
                v = problem[f.name]
            else:
                raise Exception("unsupported feature type: {}".format(type(f)))

            if v.dim() == 2:
                v = v[:, :, None]

            assert dim == v.size(-1), \
                "feature dim error, feature: {}, expected: {}, actual: {}".format(k, dim, v.size(-1))

            feature_list.append(v.float())

        if feature_list:
            x = torch.cat(feature_list, 2)
        else:
            x = self.nn_finish.new_ones(batch_size, task_num, 1)
        return self.nn_dense_task(x)

    def encode_edge(self, problem, batch_size, worker_num, task_num):
        NP = batch_size
        NW = worker_num
        NT = task_num
        NWW = NW + NW
        feature_list = []
        for k, dim in self.edge_dim.items():
            f = self.feature_dict.get(k)
            if isinstance(f, LocalCategory):
                assert f.name.startswith("task_")

                v = problem[k]
                v1 = v[:, :, None]
                v2 = v[:, None, :]

                v = torch.zeros(NP, NWW + NT, NWW + NT,
                                dtype=v.dtype, device=v.device)
                v[:, NWW:, NWW:] = ((v1 == v2) & (v1 >= 0))
            elif isinstance(f, LocalFeature):
                assert f.name.startswith("task_")

                v = problem[k].float()
                dot_product = torch.matmul(v, v.transpose(-1, -2))
                v_norm = v.norm(dim=2) + 1e-10
                v1_norm = v_norm[:, :, None]
                v2_norm = v_norm[:, None, :]

                v = torch.zeros(NP, NWW + NT, NWW + NT,
                                dtype=v.dtype, device=v.device)
                v[:, NWW:, NWW:] = dot_product / v1_norm / v2_norm
            elif isinstance(f, SparseLocalFeature):
                assert NP == 1
                assert f.index.startswith("task_")
                assert f.value.startswith("task_")

                index = problem[f.index]
                value = problem[f.value].float()
                
                NV = index.max().item() + 1
                spv = value.reshape(-1).tolist()
                spi = index.reshape(-1).tolist()

                device = value.device
                spj = torch.arange(NT, device=device)
                spj = spj[:, None].expand_as(index)
                spj = spj.reshape(-1).tolist()
                
                value1 = torch.sparse_coo_tensor([spj, spi], spv, (NT, NV), device=device)
                value2 = torch.sparse_coo_tensor([spi, spj], spv, (NV, NT), device=device)
                
                value1 = value1.coalesce()
                value2 = value2.coalesce()
                cosine = torch.sparse.mm(value1, value2).to_dense()
                
                norm = value.norm(dim=-1).reshape(-1)
                norm1 = norm[:, None].expand(-1, NT)
                norm2 = norm[None, :].expand(NT, -1)
                cosine = cosine / (norm1 * norm2 + 1e-10)

                v = torch.zeros(NP, NWW + NT, NWW + NT,
                                dtype=value.dtype, device=value.device)
                v[:, NWW:, NWW:] = cosine

            elif isinstance(f, ContinuousFeature):
                if f.name.endswith("_matrix"):
                    v = problem[k]
                elif f.name.startswith("worker_task_"):
                    v = problem[k]
                    if v.dim() == 3:
                        new_v = torch.zeros(NP, NWW + NT, NWW + NT,
                                            dtype=v.dtype, device=v.device)
                    else:
                        new_v = torch.zeros(NP, NWW + NT, NWW + NT, v.size(3),
                                            dtype=v.dtype, device=v.device)
                    problem_index = torch.arange(NP, device=v.device)[:, None, None]
                    worker_index = torch.arange(NW, device=v.device)[None, :, None]
                    task_index = torch.arange(NT, device=v.device)[None, None, :] + NW + NW
                    new_v[problem_index, worker_index, task_index] = v
                    new_v[problem_index, task_index, worker_index] = v
                    new_v[problem_index, worker_index + NW, task_index] = v
                    new_v[problem_index, task_index, worker_index + NW] = v
                    v = new_v
                else:
                    raise Exception("feature: {}".format(f.name))
            else:
                raise Exception("feature: {}, type: {}".format(k, type(f)))

            if v.dim() == 3:
                v = v[:, :, :, None]

            assert dim == v.size(-1), \
                "feature: {}, expected: {}, actual: {}".format(k, dim, v.size(-1))

            feature_list.append(v.float())

        x = torch.cat(feature_list, 3)
        return self.nn_dense_edge(x)

