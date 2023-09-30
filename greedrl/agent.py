import torch

from torch import nn
from collections import OrderedDict
from torch.utils.checkpoint import checkpoint
from .feature import *
from .pyenv import PyEnv
from .encode import Encode
from .decode import Decode


class Agent(nn.Module):

    def __init__(self, nn_args):
        super(Agent, self).__init__()

        self.nn_args = nn_args
        self.vars_dim = sum(nn_args['variable_dim'].values())
        self.steps_ratio = nn_args.setdefault('decode_steps_ratio', 1.0);

        logit_clips = nn_args.setdefault('decode_logit_clips', 10.0);
        if isinstance(logit_clips, str):
            self.logit_clips = [float(v) for v in logit_clips.split(',')]
        else:
            self.logit_clips = [float(logit_clips)]

        self.nn_encode = Encode(nn_args)
        self.nn_decode = Decode(nn_args)

    def nn_args_dict(self):
        return self.nn_args

    def forward(self, problem, batch_size, greedy=False, solution=None, memopt=0):
        X, K, V = self.nn_encode(problem.feats, problem.batch_size,
                                 problem.worker_num, problem.task_num, memopt)

        return self.interact(problem, X, K, V, batch_size, greedy, solution, memopt)

    def interact(self, problem, X, K, V, batch_size, greedy, solution, memopt):
        NP = problem.batch_size
        NW = problem.worker_num
        NT = problem.task_num

        sample_num = batch_size // NP
        assert sample_num > 0 and batch_size % NP == 0

        MyEnv = problem.environment
        if MyEnv is None:
            env = PyEnv(problem, batch_size, sample_num, self.nn_args)
        else:
            env = MyEnv(str(problem.device), problem.feats, batch_size,
                        sample_num, problem.worker_num, problem.task_num)

        query = X.new_zeros(batch_size, X.size(-1))
        state1 = X.new_zeros(batch_size, X.size(-1))
        state2 = X.new_zeros(batch_size, X.size(-1))

        p_list = []
        NULL = X.new_ones(0)
        p_index = torch.div(torch.arange(batch_size, device=X.device), sample_num, rounding_mode='trunc') # torch.arange(batch_size, device=X.device) // sample_num
        if solution is not None:
            solution = solution[:, :, 0:2].to(torch.int64).permute(1, 0, 2)
            assert torch.all(solution >= 0) and solution.size(1) == batch_size
            offset = torch.tensor([0, NW, NW + NW, NW + NW + NT], device=X.device)
            chosen_list = solution[:, :, 1] + offset[solution[:, :, 0]]

            mode = 0
            sample_p = torch.rand(batch_size, device=X.device)
            for chosen in chosen_list:
                env_time = env.time()
                clip = self.logit_clips[min(env_time, len(self.logit_clips) - 1)]
                varfeat = env.make_feat() if self.vars_dim > 0 else NULL
                state1, state2, chosen_p = self.decode(X, K, V, query, state1, state2,
                                                       varfeat, env.mask(), chosen, sample_p, clip, mode, memopt)
                query = X[p_index, chosen]
                p_list.append(chosen_p)
                env.step(chosen)

            assert env.all_finished(), 'not all finished!'
        else:
            mode = 1 if greedy else 2
            min_env_time = int(self.steps_ratio * NT)
            R = torch.rand(NT * 2, batch_size, device=X.device)
            while True:
                env_time = env.time()
                if env_time > min_env_time and env_time % 3 == 0 and env.all_finished():
                    break

                clip = self.logit_clips[min(env_time, len(self.logit_clips) - 1)]
                sample_p = R[env_time % R.size(0)]
                chosen = X.new_empty(batch_size, dtype=torch.int64)
                varfeat = env.make_feat() if self.vars_dim > 0 else NULL
                state1, state2, chosen_p = self.decode(X, K, V, query, state1, state2,
                                                       varfeat, env.mask(), chosen, sample_p, clip, mode, memopt)
                query = X[p_index, chosen]
                p_list.append(chosen_p)
                env.step(chosen)

        env.finalize()
        return env, torch.stack(p_list, 1)

    def decode(self, X, K, V, query, state1, state2, varfeat, mask, chosen, sample_p, clip, mode, memopt):
        run_fn = self.decode_fn(clip, mode, memopt)
        if self.training and memopt > 3:
            return checkpoint(run_fn, X, K, V, query, state1, state2, varfeat, mask, chosen, sample_p)
        else:
            return run_fn(X, K, V, query, state1, state2, varfeat, mask, chosen, sample_p)

    def decode_fn(self, clip, mode, memopt):
        memopt = 0 if memopt > 3 else memopt

        def run_fn(X, K, V, query, state1, state2, varfeat, mask, chosen, sample_p):
            return self.nn_decode(X, K, V, query, state1, state2,
                                  varfeat, mask, chosen, sample_p, clip, mode, memopt)

        return run_fn


def parse_nn_args(problem, nn_args):
    worker_dim = OrderedDict()
    task_dim = OrderedDict()
    edge_dim = OrderedDict()
    variable_dim = OrderedDict()
    embed_dict = OrderedDict()

    def set_dim_by_name(name, k, dim):
        if name.startswith("worker_task_"):
            edge_dim[k] = dim
        elif name.startswith("worker_"):
            worker_dim[k] = dim
        elif name.startswith("task_"):
            task_dim[k] = dim
        elif name.endswith("_matrix"):
            edge_dim[k] = dim
        else:
            raise Exception("attribute can't be feature: {}".format(k))

    feature_dict = make_feat_dict(problem)
    variables = [var(problem, problem.batch_size, 1) for var in problem.variables]
    variable_dict = dict([(var.name, var) for var in variables])
    for k, f in feature_dict.items():
        if isinstance(f, VariableFeature):
            var = variable_dict[f.name]
            assert hasattr(var, 'make_feat'), \
                "{} cann't be variable feature, name:{}".format(type(var).__name__, k)
            v = var.make_feat()
            if v.dim() == 2:
                variable_dim[k] = 1
            else:
                variable_dim[k] = v.size(-1)
        elif isinstance(f, SparseLocalFeature):
            edge_dim[k] = 1
            set_dim_by_name(f.value, k, 1)
        elif isinstance(f, LocalFeature):
            edge_dim[k] = 1
            set_dim_by_name(f.name, k, 1)
        elif isinstance(f, LocalCategory):
            edge_dim[k] = 1
        elif isinstance(f, GlobalCategory):
            set_dim_by_name(f.name, k, nn_args.setdefault('encode_hidden_dim', 128))
            embed_dict[k] = f.size
        elif isinstance(f, ContinuousFeature):
            v = problem.feats[k]
            if k.startswith("worker_task_") or k.endswith("_matrix"):
                simple_dim = 3
            else:
                simple_dim = 2

            if v.dim() == simple_dim:
                set_dim_by_name(f.name, k, 1)
            else:
                set_dim_by_name(f.name, k, v.size(-1))
        else:
            raise Exception("unsupported feature type: {}".format(type(f)))

    nn_args['worker_dim'] = worker_dim
    nn_args['task_dim'] = task_dim
    nn_args['edge_dim'] = edge_dim
    nn_args['variable_dim'] = variable_dim
    nn_args['embed_dict'] = embed_dict
    nn_args['feature_dict'] = feature_dict
    return nn_args


def make_feat_dict(problem):
    feature_dict = OrderedDict()

    def add(k, f):
        _f = feature_dict.get(k)
        if _f is None or _f == f:
            feature_dict[k] = f
        else:
            "duplicated feature, name: {}, feature1: {}, feature2: {}".format(k, _f, f)

    for f in problem.features:
        if isinstance(f, VariableFeature):
            add(':'.join(['var', f.name]), f)
        elif isinstance(f, SparseLocalFeature):
            add(':'.join([f.index, f.value]), f)
        else:
            add(f.name, f)

    return feature_dict
