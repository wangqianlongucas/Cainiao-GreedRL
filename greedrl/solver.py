import os
import sys
import math
import copy
import time
import queue
import inspect
import torch
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist

from .agent import Agent, parse_nn_args
from .utils import repeat, get_default_device, cutime_stats
from .variable import TaskDemandNow

from torch.nn.utils import clip_grad_norm_, parameters_to_vector, vector_to_parameters
from torch.utils.data import Dataset, IterableDataset, DataLoader
from torch.optim.lr_scheduler import MultiStepLR


class Problem(object):
    def __init__(self, isbatch=False):
        self.isbatch = isbatch
        self.features = []
        self.environment = None

    def pin_memory(self):
        for k, v in self.feats.items():
            self.feats[k] = v.pin_memory()
        return self

    def __getattr__(self, name):
        if name not in ('solution'):
            raise AttributeError()
        return self.feats.get(name)


class Solution(object):
    def __init__(self, cost=None):
        self.cost = cost
        self.worker_task_sequence = None


class WrapDataset(Dataset):
    def __init__(self, dataset, solver):
        self._dataset = [solver.to_batch(p) for p in dataset]

    def __getitem__(self, index):
        return self._dataset[index]

    def __len__(self):
        return len(self._dataset)


class WrapIterator:
    def __init__(self, iterator, solver):
        self._iterator = iterator
        self._solver = solver

    def __next__(self):
        p = next(self._iterator)
        p = self._solver.to_batch(p, False)
        return p


class WrapIterableDataset(IterableDataset):
    def __init__(self, dataset, solver):
        self._dataset = dataset
        self._solver = solver

    def __iter__(self):
        return WrapIterator(iter(self._dataset), self._solver)


class CyclicIterator:
    def __init__(self, iterable):
        self._iterable = iterable
        self._iterator = iter(iterable)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = iter(self._iterable)
            return next(self._iterator)


class BufferedIterator:
    def __init__(self, iterator, size, reuse):
        self._iterator = iterator
        self._reuse = reuse
        self._queue = queue.Queue(size)
        self._buffer = []
        self._iter_step = 0

    def __next__(self):
        if not self._queue.full() or self._iter_step % self._reuse == 0:
            problem = next(self._iterator)
            if self._queue.full():
                index = self._queue.get()
                self._buffer[index] = problem
            else:
                index = len(self._buffer)
                self._buffer.append(problem)
            self._queue.put(index)
        self._iter_step += 1
        index = torch.randint(0, len(self._buffer), (1,)).item()
        return self._buffer[index]


class Solver(object):
    def __init__(self, device=None, nn_args=None):

        if device is None:
            self.device = get_default_device()
        elif device == 'cuda':
            self.device = get_default_device()
            assert self.device.type == 'cuda', 'no cuda device available!'
        else:
            self.device = torch.device(device)

        if nn_args is None:
            nn_args = {}
        self.nn_args = nn_args

        self.agent = None

    def parse_nn_args(self, problem):
        parse_nn_args(problem, self.nn_args)

    def new_agent(self):
        return Agent(self.nn_args)

    def train(self, agent_filename, train_dataset, valid_dataset, **kwargs):
        if dist.is_initialized():
            torch.manual_seed(torch.initial_seed() + dist.get_rank() * 20000)

        train_dataset_workers = kwargs.pop('train_dataset_workers', 1)
        train_dataset_buffers = kwargs.pop('train_dataset_buffers', 2)
        valid_dataset_workers = kwargs.pop('valid_dataset_workers', 1)
        valid_dataset_buffers = kwargs.pop('valid_dataset_buffers', 2)

        train_dataset = self.wrap_dataset(train_dataset, train_dataset_workers,
                                          train_dataset_buffers, torch.initial_seed() + 1)
        valid_dataset = self.wrap_dataset(valid_dataset, valid_dataset_workers,
                                          valid_dataset_buffers, torch.initial_seed() + 10001)

        if self.device.type == 'cuda':
            with torch.cuda.device(cuda_or_none(self.device)):
                self.do_train(agent_filename, train_dataset, valid_dataset, **kwargs)
        else:
            self.do_train(agent_filename, train_dataset, valid_dataset, **kwargs)

    def do_train(self, agent_filename, train_dataset, valid_dataset, reuse_buffer=0, reuse_times=1, on_policy=True,
                 advpow=1, batch_size=512, topk_size=1, init_lr=0.0001, sched_lr=(int(1e10),), gamma_lr=0.5,
                 warmup_steps=100, log_steps=-1, optim_steps=1, valid_steps=100, max_steps=int(1e10), memopt=1):

        for arg in inspect.getfullargspec(self.do_train)[0][1:]:
            if arg not in ('train_dataset', 'valid_dataset'):
                print("train_args: {} = {}".format(arg, locals()[arg]))

        if log_steps < 0:
            log_steps = valid_steps

        train_dataset = CyclicIterator(train_dataset)
        if reuse_buffer > 0:
            train_dataset = BufferedIterator(train_dataset, reuse_buffer, reuse_times)

        valid_dataset = list(valid_dataset)

        if dist.is_initialized() and dist.get_rank() != 0:
            dist.barrier()

        if agent_filename is not None and os.path.exists(agent_filename):
            saved_state = torch.load(agent_filename, map_location='cpu')
            self.nn_args = saved_state['nn_args']
        else:
            saved_state = None
            self.parse_nn_args(valid_dataset[0])

        step = 0
        start_step = 0
        self.agent = self.new_agent().train()
        self.agent.to(self.device)
        self.print_nn_args()

        best_agent = copy.deepcopy(self.agent).eval()
        min_valid_cost = math.inf

        optimizer = torch.optim.Adam(self.agent.parameters(), lr=init_lr)
        scheduler = MultiStepLR(optimizer, milestones=sched_lr, gamma=gamma_lr)

        def do_save_state(rng_state, cuda_rng_state):
            if agent_filename is not None:
                save_data = {'step': step, 'rng_state': rng_state}
                if cuda_rng_state is not None:
                    save_data['cuda_rng_state'] = cuda_rng_state
                save_data['nn_args'] = self.agent.nn_args_dict()
                save_data['agent_state'] = self.agent.state_dict()
                save_data['best_agent_state'] = best_agent.state_dict()
                save_data['optimizer_state'] = optimizer.state_dict()
                save_data['scheduler_state'] = scheduler.state_dict()
                torch.save(save_data, agent_filename)

        def valid_sched_save(step):
            if dist.is_initialized():
                params = parameters_to_vector(self.agent.parameters())
                params_clone = params.clone()
                dist.broadcast(params_clone, 0)
                assert torch.all(params == params_clone)

            rng_state = torch.get_rng_state()
            cuda_rng_state = None
            if self.device.type == 'cuda':
                cuda_rng_state = torch.cuda.get_rng_state(self.device)

            print("{} - step={}, validate...".format(time.strftime("%Y-%m-%d %H:%M:%S"), step))
            sys.stdout.flush()

            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            start_time = time.time()
            valid_result = self.validate(valid_dataset, batch_size)
            avg_cost1, avg_cost2, avg_feasible = valid_result
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)

            duration = time.time() - start_time

            if step > 0:
                scheduler.step()

            if not dist.is_initialized() or dist.get_rank() == 0:
                do_save_state(rng_state, cuda_rng_state)

            strftime = time.strftime("%Y-%m-%d %H:%M:%S")
            print("{} - step={}, cost=[{:.6g}, {:.6g}], feasible={:.0%}".format(
                strftime, step, avg_cost1, avg_cost2, avg_feasible))
            print("{} - step={}, min_valid_cost={:.6g}, time={:.3f}s".format(
                strftime, step, min(min_valid_cost, avg_cost2), duration))
            print("---------------------------------------------------------------------------------------")
            sys.stdout.flush()
            return avg_cost2

        if saved_state is not None:
            start_step = saved_state['step']

            if not dist.is_initialized() or dist.get_rank() == 0:
                torch.set_rng_state(saved_state['rng_state'])
                if torch.cuda.is_available():
                    torch.cuda.set_rng_state(saved_state['cuda_rng_state'], self.device)

            best_agent.load_state_dict(saved_state['best_agent_state'])
            self.agent.load_state_dict(saved_state['best_agent_state'])

            # if 'agent_state' in saved_state:
            #    self.agent.load_state_dict(saved_state['agent_state'])
            # else:
            #    self.agent.load_state_dict(saved_state['best_agent_state'])

            if 'optimizer_state' in saved_state:
                optimizer.load_state_dict(saved_state['optimizer_state'])
            if 'scheduler_state' in saved_state:
                scheduler.load_state_dict(saved_state['scheduler_state'])
        else:
            if dist.is_initialized() and dist.get_rank() == 0:
                rng_state = torch.get_rng_state()
                cuda_rng_state = None
                if self.device.type == 'cuda':
                    cuda_rng_state = torch.cuda.get_rng_state(self.device)
                do_save_state(rng_state, cuda_rng_state)

        if dist.is_initialized() and dist.get_rank() == 0:
            dist.barrier()

        for step in range(start_step, max_steps):
            if step % valid_steps == 0:
                valid_cost = valid_sched_save(step)
                if valid_cost < min_valid_cost:
                    best_agent.load_state_dict(self.agent.state_dict())
                    min_valid_cost = valid_cost

            start_time = time.time()

            # problem
            with torch.no_grad():
                problem = next(train_dataset)
                if step < warmup_steps:
                    batch_size_now = batch_size // 2
                else:
                    batch_size_now = batch_size
                problem = self.to_device(problem)

            if not on_policy:
                data_agent = best_agent
            else:
                data_agent = self.agent

            data_agent.eval()

            # solution
            if topk_size > 1:
                with torch.no_grad():
                    batch_size_topk = batch_size_now * topk_size
                    env, logp = data_agent(problem, batch_size_topk)
                    cost = env.cost().sum(1).float()
                    solution = env.worker_task_sequence()

                    NP = problem.batch_size
                    NK = batch_size_now // NP
                    NS = solution.size(1)

                    cost = cost.view(NP, -1)
                    cost, kidx = cost.topk(NK, 1, False, False)
                    cost = cost.view(-1)
                    kidx = kidx[:, :, None, None].expand(-1, -1, NS, 3)
                    solution = solution.view(NP, -1, NS, 3)
                    solution = solution.gather(1, kidx).view(-1, NS, 3)

            elif not on_policy:
                with torch.no_grad():
                    env, logp = data_agent(problem, batch_size_now)
                    cost = env.cost().sum(1).float()
                    solution = env.worker_task_sequence()
            else:
                self.agent.train()
                env, logp = self.agent(problem, batch_size_now, memopt=memopt)
                cost = env.cost().sum(1).float()
                solution = env.worker_task_sequence()

            self.agent.train()

            # advantage
            with torch.no_grad():
                NP = problem.batch_size
                if topk_size > 1:
                    baseline = cost.view(NP, -1).max(1)[0]
                else:
                    baseline = cost.view(NP, -1).mean(1)
                baseline = repeat(baseline, cost.size(0) // NP)
                adv = (cost - baseline)[:, None]
                adv_norm = adv.norm()
                if adv_norm > 0:
                    adv = adv / adv.norm() * adv.size(0)
                    adv = adv.sign() * adv.abs().pow(advpow)

            # backward
            if topk_size > 1 or not on_policy:
                env, logp = self.agent(problem, batch_size_now, solution=solution, memopt=memopt)

            loss = adv * logp
            loss = loss.mean()
            loss.backward()

            if step % optim_steps == 0:
                if dist.is_initialized():
                    params = filter(lambda a: a.grad is not None, self.agent.parameters())
                    grad_list = [param.grad for param in params]
                    grad_vector = parameters_to_vector(grad_list)
                    dist.all_reduce(grad_vector, op=dist.ReduceOp.SUM)
                    vector_to_parameters(grad_vector, grad_list)

                grad_norm = clip_grad_norm_(self.agent.parameters(), 1)
                optimizer.step()
                optimizer.zero_grad()

            if step % log_steps == 0:
                strftime = time.strftime("%Y-%m-%d %H:%M:%S")
                lr = optimizer.param_groups[0]['lr']
                duration = time.time() - start_time
                with torch.no_grad():
                    p = logp.to(torch.float64).sum(1).exp().mean()
                print("{} - step={}, grad={:.6g}, lr={:.6g}, p={:.6g}".format(
                    strftime, step, grad_norm, lr, p))

                print("{} - step={}, cost={:.6g}, time={:.3f}s".format(strftime, step, cost.mean(), duration))
                print("---------------------------------------------------------------------------------------")
                sys.stdout.flush()

        valid_sched_save(step)

    def solve(self, problem, greedy=False, batch_size=512):
        if self.device.type == 'cuda':
            with torch.cuda.device(cuda_or_none(self.device)):
                return self.do_solve(problem, greedy, batch_size)
        else:
            return self.do_solve(problem, greedy, batch_size)

    def do_solve(self, problem, greedy, batch_size):
        isbatch = problem.isbatch
        problem = self.to_batch(problem)
        problem = self.to_device(problem)

        if self.agent is None:
            self.parse_nn_args(problem)
            self.agent = self.new_agent()
            self.agent.to(self.device)

        self.agent.eval()

        with torch.no_grad():
            env, prob = self.agent(problem, batch_size, greedy, problem.solution)

        NP = problem.batch_size
        NR = prob.size(0) // NP

        prob = prob.view(NP, NR, -1)
        cost = env.cost().sum(1).view(NP, NR)
        feasible = env.feasible().view(NP, NR)
        size = list(env.worker_task_sequence().size())
        size = [NP, NR] + size[1:]
        worker_task_sequence = env.worker_task_sequence().view(size)

        p_index = torch.arange(NP)
        base_cost = cost.max() + 1
        cost[~feasible] += base_cost
        cost, s_index = cost.min(1)
        feasible = feasible[p_index, s_index]
        cost[~feasible] -= base_cost
        probability = prob[p_index, s_index].exp()
        worker_task_sequence = worker_task_sequence[p_index, s_index]

        if isbatch:
            solution = Solution(cost)
            solution.feasible = feasible
            solution.probability = probability
            solution.worker_task_sequence = worker_task_sequence
        else:
            solution = Solution(cost.item())
            solution.feasible = feasible.item()
            solution.probability = probability.squeeze(0)
            solution.worker_task_sequence = worker_task_sequence.squeeze(0)

        return solution

    def load_agent(self, filename, strict=True):
        if self.device.type == 'cuda':
            with torch.cuda.device(cuda_or_none(self.device)):
                self.do_load_agent(filename, strict)
        else:
            self.do_load_agent(filename, strict)

    def do_load_agent(self, filename, strict=True):
        saved_state = torch.load(filename, map_location='cpu')
        self.nn_args = saved_state['nn_args']

        self.agent = self.new_agent()
        self.agent.to(self.device)
        self.agent.load_state_dict(saved_state['best_agent_state'], strict)
        self.print_nn_args()

    def to_batch(self, problem, pin_memory=True):
        assert not hasattr(problem, 'feats')

        NW = 1
        NT = 1
        NP = 1
        isbatch = problem.isbatch
        for k, v in problem.__dict__.items():
            if k.startswith("worker_"):
                NW = len(v[0]) if isbatch else len(v)
            elif k.startswith("task_"):
                NP = len(v) if isbatch else 1
                NT = len(v[0]) if isbatch else len(v)
        NWW = NW * 2

        new_problem = Problem(True)
        new_problem.feats = {}
        new_problem.device = 'cpu'

        new_problem.batch_size = NP
        new_problem.worker_num = NW
        new_problem.task_num = NT

        new_problem.features = problem.features

        if type(self) == Solver:
            new_problem.variables = problem.variables
            new_problem.constraint = problem.constraint
            new_problem.objective = problem.objective
            new_problem.environment = problem.environment
        else:
            new_problem.variables = []
            new_problem.constraints = problem.constraints
            new_problem.oa_estimate_tasks = problem.oa_estimate_tasks
            new_problem.oa_multiple_steps = problem.oa_multiple_steps

        edge_size_list = ((NWW + NT, NWW + NT), (NW + NT, NW + NT))

        def check_size(f, k, v):
            assert f, "size error, feature: {}, size: {}".format(k, tuple(v.size()))

        for k, v in problem.__dict__.items():
            if k == 'solution' and v is not None:
                v = to_tensor(k, v, isbatch)
                check_size(v.dim() == 3 and v.size(-1) == 3, k, v)
            elif k.startswith("worker_task_"):
                v = to_tensor(k, v, isbatch)
                check_size(v.dim() in (3, 4) and v.size()[1:3] == (NW, NT), k, v)
            elif k.startswith("worker_"):
                v = to_tensor(k, v, isbatch)
                check_size(v.dim() in (2, 3) and v.size(1) == NW, k, v)
            elif k.startswith("task_"):
                v = to_tensor(k, v, isbatch)
                check_size(v.dim() in (2, 3) and v.size(1) == NT, k, v)
            elif k.endswith("_matrix"):
                v = to_tensor(k, v, isbatch)
                check_size(v.dim() in (3, 4) and v.size()[1:3] in edge_size_list, k, v)
                if v.size()[1:3] == (NW + NT, NW + NT):
                    worker_index = torch.arange(NW)
                    task_index = torch.arange(NT) + NW
                    index = torch.cat([worker_index, worker_index, task_index])
                    index1 = index[:, None]
                    index2 = index[None, :]
                    v = v[:, index1, index2]
            elif isinstance(v, np.ndarray):
                v = torch.tensor(v)

            if isinstance(v, torch.Tensor):
                new_problem.feats[k] = v

        if pin_memory and self.device.type == 'cuda':
            new_problem.pin_memory()

        return new_problem

    def to_device(self, problem):

        assert hasattr(problem, 'feats')

        new_problem = copy.copy(problem)
        new_problem.device = self.device
        new_problem.feats = {}

        non_blocking = self.device.type == 'cuda'
        for k, v in problem.feats.items():
            v = v.to(self.device, non_blocking=non_blocking)
            new_problem.feats[k] = v

        return new_problem

    def validate(self, problem_list, batch_size):
        self.agent.eval()
        with torch.no_grad():
            valid_result = self.do_validate(problem_list, batch_size)

        self.agent.train()
        return valid_result

    def do_validate(self, problem_list, batch_size):
        total_cost1 = 0
        total_cost2 = 0
        total_feasible = 0
        total_problem = 0
        start_time = time.time()
        for problem in problem_list:
            problem = self.to_device(problem)
            env, _, = self.agent(problem, batch_size)

            NP = problem.batch_size
            cost = env.cost().sum(1).view(NP, -1)
            cost1, _ = cost.min(1)
            cost2 = cost.mean(1)
            feasible = env.feasible().view(NP, -1)
            feasible = torch.any(feasible, 1)

            total_cost1 += cost1.sum().item()
            total_cost2 += cost2.sum().item()
            total_feasible += feasible.int().sum().item()
            total_problem += NP

        if dist.is_initialized():
            data = [total_cost1, total_cost2, total_feasible, total_problem]
            data = torch.tensor(data, device=self.device)
            dist.all_reduce(data, op=dist.ReduceOp.SUM)
            total_cost1, total_cost2, total_feasible, total_problem = data.tolist()

        avg_cost1 = total_cost1 / total_problem
        avg_cost2 = total_cost2 / total_problem
        avg_feasible = total_feasible / total_problem

        return avg_cost1, avg_cost2, avg_feasible

    def wrap_dataset(self, dataset, workers, buffers, seed):
        if isinstance(dataset, IterableDataset):
            dataset = WrapIterableDataset(dataset, self)
            dataset = DataLoader(dataset, batch_size=None, pin_memory=True,
                                 num_workers=workers, prefetch_factor=buffers,
                                 worker_init_fn=lambda worker_id: torch.manual_seed(seed + worker_id))
        else:
            if self.device.type == 'cuda':
                with torch.cuda.device(cuda_or_none(self.device)):
                    dataset = WrapDataset(dataset, self)
                    dataset = DataLoader(dataset, batch_size=None, pin_memory=True, shuffle=True)
            else:
                dataset = WrapDataset(dataset, self)
                dataset = DataLoader(dataset, batch_size=None, pin_memory=True, shuffle=True)

        return dataset

    def print_nn_args(self):
        for key, value in self.nn_args.items():
            if type(value) in [int, float, str, bool]:
                print("nn_args: {} = {}".format(key, value))
        sys.stdout.flush()


def to_tensor(key, value, isbatch):
    if isinstance(value, torch.Tensor):
        tensor = value.to('cpu')
    else:
        tensor = torch.tensor(value, device='cpu')

    if not isbatch:
        tensor = tensor[None]

    return tensor


def cuda_or_none(device):
    return device if device.type == 'cuda' else None
