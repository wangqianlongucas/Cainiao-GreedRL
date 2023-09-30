import torch
import json
import math

from collections import OrderedDict
from .const import *
from .utils import to_list
from .norm import Norm1D, Norm2D
from .variable import AttributeVariable, WorkerTaskSequence


class PyEnv(object):

    def __init__(self, problem, batch_size, sample_num, nn_args):
        super(PyEnv, self).__init__()

        self._problem = problem
        self._batch_size = batch_size
        self._sample_num = sample_num
        self._debug = -1

        self._NW = problem.worker_num
        self._NWW = problem.worker_num * 2
        self._NT = problem.task_num
        self._NWWT = self._NWW + self._NT

        self._feats_dict = nn_args['feature_dict']
        self._vars_dim = nn_args['variable_dim']

        self._vars_dict = {}
        self._vars = [var(problem, batch_size, sample_num) for var in problem.variables]
        for variable in self._vars:
            save_variable_version(variable)
            assert variable.name not in self._vars_dict, \
                "duplicated variable, name: {}".format(variable.name)
            self._vars_dict[variable.name] = variable

        self._constraint = problem.constraint()
        self._objective = problem.objective()

        self._worker_index = torch.full((self._batch_size,), -1,
                                        dtype=torch.int64,
                                        device=problem.device)

        self._batch_index = torch.arange(self._batch_size,
                                         dtype=torch.int64,
                                         device=problem.device)

        self._problem_index = torch.div(self._batch_index, sample_num, rounding_mode='trunc') #  self._batch_index // sample_num

        self._feasible = torch.ones(self._batch_size,
                                    dtype=torch.bool,
                                    device=problem.device)

        self._cost = torch.zeros(self._batch_size, self._NT * 2,
                                 dtype=torch.float32,
                                 device=problem.device)

        self._mask = torch.zeros(self._batch_size,
                                 self._NWWT + 1,
                                 dtype=torch.bool,
                                 device=problem.device)

        self._worker_task_sequence = torch.full((self._batch_size, self._NT * 2, 3), -1,
                                                dtype=torch.int64,
                                                device=problem.device)
        self._step = 0
        self.register_variables(self._constraint)
        self._finished = self._constraint.finished()

        if hasattr(self._constraint, 'mask_worker_start'):
            self.register_variables(self._constraint)
            mask_start = self._constraint.mask_worker_start()
        else:
            mask_start = False

        self._mask[:, :self._NW] = mask_start
        self._mask[:, self._NW:] = True

        if self._debug >= 0:
            print("\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            print("new env")
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")

    def time(self):
        return self._step

    def step(self, chosen):
        with torch.no_grad():
            self._do_step(chosen)

    def _do_step(self, chosen):
        if self._debug >= 0:
            print("----------------------------------------------------------------------")
            feasible = self._feasible & ~self._mask[self._problem_index, chosen]
            print("feasible={}".format(feasible[self._debug].tolist()))

        is_start = (chosen >= 0) & (chosen < self._NW)
        if torch.any(is_start):
            b_index = self._batch_index[is_start]
            p_index = self._problem_index[is_start]
            w_index = chosen[is_start]
            self.step_worker_start(b_index, p_index, w_index)

        is_end = (chosen >= self._NW) & (chosen < self._NWW)
        if torch.any(is_end):
            b_index = self._batch_index[is_end]
            p_index = self._problem_index[is_end]
            w_index = chosen[is_end] - self._NW
            self.step_worker_end(b_index, p_index, w_index)

        is_task = (chosen >= self._NWW) & (chosen < self._NWWT)
        if torch.any(is_task):
            b_index = self._batch_index[is_task]
            p_index = self._problem_index[is_task]
            t_index = chosen[is_task] - self._NWW
            step_task_b_index = b_index
            self.step_task(b_index, p_index, t_index)
        else:
            step_task_b_index = None

        is_finish = chosen == self._NWWT
        if torch.any(is_finish):
            b_index = self._batch_index[is_finish]
            self._worker_task_sequence[b_index, self._step, 0] = GRL_FINISH
            self._worker_task_sequence[b_index, self._step, 1] = 0
            self._worker_task_sequence[b_index, self._step, 2] = -1

        self.update_mask(step_task_b_index)

        for var in self._vars:
            check_variable_version(var)

        if self._debug >= 0:
            print("worker_task_sequence[{}]={}".format(self._step,
                                                       self._worker_task_sequence[self._debug, self._step].tolist()))
            for var in self._vars:
                if var.value is None:
                    print("{}={}".format(var.name, None))
                elif isinstance(var, AttributeVariable):
                    print("{}={}".format(var.name, to_list(var.value)))
                else:
                    print("{}={}".format(var.name, to_list(var.value[self._debug])))

        self._step += 1
        if self._step >= self._cost.size(1):
            cost = torch.zeros(self._batch_size, self._step + self._NT,
                               dtype=torch.float32,
                               device=chosen.device)
            cost[:, 0:self._step] = self._cost;
            self._cost = cost

            worker_task_sequence = torch.full((self._batch_size, self._step + self._NT, 3), -1,
                                              dtype=torch.int64,
                                              device=chosen.device)
            worker_task_sequence[:, 0:self._step, :] = self._worker_task_sequence
            self._worker_task_sequence = worker_task_sequence

    def step_worker_start(self, b_index, p_index, w_index):
        self._worker_task_sequence[b_index, self._step, 0] = GRL_WORKER_START
        self._worker_task_sequence[b_index, self._step, 1] = w_index
        self._worker_task_sequence[b_index, self._step, 2] = -1
        for var in self._vars:
            if hasattr(var, 'step_worker_start'):
                var.step_worker_start(b_index, p_index, w_index)
                save_variable_version(var)

        if hasattr(self._objective, 'step_worker_start'):
            self.register_variables(self._objective, b_index)
            self.update_cost(self._objective.step_worker_start(), b_index)

        self._worker_index[b_index] = w_index
        self._mask[b_index, :self._NWW] = True
        self._mask[b_index, self._NWW:] = False

    def step_worker_end(self, b_index, p_index, w_index):
        self._worker_task_sequence[b_index, self._step, 0] = GRL_WORKER_END
        self._worker_task_sequence[b_index, self._step, 1] = w_index
        self._worker_task_sequence[b_index, self._step, 2] = -1;

        for var in self._vars:
            if hasattr(var, 'step_worker_end'):
                var.step_worker_end(b_index, p_index, w_index)
                save_variable_version(var)

        if hasattr(self._objective, 'step_worker_end'):
            self.register_variables(self._objective, b_index)
            self.update_cost(self._objective.step_worker_end(), b_index)

        self._worker_index[b_index] = -1

        self.register_variables(self._constraint, b_index)
        self._finished[b_index] |= self._constraint.finished()
        if hasattr(self._constraint, 'mask_worker_start'):
            mask_start = self._constraint.mask_worker_start()
        else:
            mask_start = False

        self._mask[b_index, :self._NW] = mask_start
        self._mask[b_index, self._NW:] = True

    def step_task(self, b_index, p_index, t_index):
        self._worker_task_sequence[b_index, self._step, 0] = GRL_TASK
        self._worker_task_sequence[b_index, self._step, 1] = t_index

        for var in self._vars:
            if not hasattr(var, 'step_task'):
                continue
            elif var.step_task.__code__.co_argcount == 4:
                var.step_task(b_index, p_index, t_index)
            else:
                var.step_task(b_index, p_index, t_index, None)
            save_variable_version(var)

        if hasattr(self._constraint, 'do_task'):
            self.register_variables(self._constraint, b_index)
            done = self._constraint.do_task()
            self._worker_task_sequence[b_index, self._step, 2] = done.long()

            for var in self._vars:
                if not hasattr(var, 'step_task'):
                    continue
                elif var.step_task.__code__.co_argcount == 4:
                    pass
                else:
                    check_variable_version(var)
                    var.step_task(b_index, p_index, t_index, done)
                    save_variable_version(var)
        else:
            done = None

        if hasattr(self._objective, 'step_task'):
            self.register_variables(self._objective, b_index)
            self.update_cost(self._objective.step_task(), b_index)

        if hasattr(self._constraint, 'mask_worker_end'):
            self.register_variables(self._constraint, b_index)
            mask_end = self._constraint.mask_worker_end()
        else:
            mask_end = False

        w_index = self._NW + self._worker_index[b_index]
        self._mask[b_index, w_index] = mask_end
        self._mask[b_index, self._NWW:] = False
        return done

    def update_cost(self, cost, b_index=None):
        if isinstance(cost, tuple):
            cost, feasible = cost
            if b_index is None:
                self._feasible &= feasible
            else:
                self._feasible[b_index] &= feasible

        if isinstance(cost, torch.Tensor):
            cost = cost.float()
        else:
            assert type(cost) in (int, float), "unexpected cost's type: {}".format(type(cost))

        if b_index is None:
            self._cost[:, self._step] = cost
        else:
            self._cost[b_index, self._step] = cost

    def update_mask(self, step_task_b_index):
        self._mask |= self._finished[:, None]
        self._mask[:, -1] = ~self._finished
        self.register_variables(self._constraint)
        self._mask[:, self._NWW:self._NWWT] |= self._constraint.mask_task()

        if step_task_b_index is not None:
            b_index = step_task_b_index
            w_index = self._NW + self._worker_index[b_index]
            task_mask = self._mask[b_index, self._NWW:self._NWWT]
            self._mask[b_index, w_index] &= ~torch.all(task_mask, 1)

    def batch_size():
        return self._batch_size

    def sample_num():
        return self._sample_num

    def mask(self):
        return self._mask.clone()

    def cost(self):
        return self._cost[:, 0:self._step]

    def feasible(self):
        return self._feasible

    def worker_task_sequence(self):
        return self._worker_task_sequence[:, 0:self._step]

    def var(self, name):
        return self._vars_dict[name].value

    def register_variables(self, obj, b_index=None, finished=False):
        for var in self._vars:
            if var.value is None or b_index is None \
                    or isinstance(var, AttributeVariable):
                value = var.value
            else:
                value = var.value[b_index]
            obj.__dict__[var.name] = value

            if not hasattr(var, 'ext_values'):
                continue

            for k, v in var.ext_values.items():
                k = var.name + '_' + k
                obj.__dict__[k] = v[b_index]

    def finished(self):
        return self._finished

    def all_finished(self):
        return torch.all(self.finished())

    def finalize(self):
        self._worker_task_sequence[:, self._step, 0] = GRL_FINISH
        self._worker_task_sequence[:, self._step, 1] = 0
        self._worker_task_sequence[:, self._step, 2] = -1

        for var in self._vars:
            if hasattr(var, 'step_finish'):
                var.step_finish(self.worker_task_sequence())

        if hasattr(self._objective, 'step_finish'):
            self.register_variables(self._objective, finished=True)
            self.update_cost(self._objective.step_finish())

        self._step += 1

    def make_feat(self):
        with torch.no_grad():
            return self.do_make_feat()

    def do_make_feat(self):
        if not self._vars_dim:
            return None

        feature_list = []
        for k, dim in self._vars_dim.items():
            f = self._feats_dict[k]
            var = self._vars_dict[f.name]
            v = var.make_feat()
            if v.dim() == 2:
                v = v[:, :, None]

            assert dim == v.size(-1), \
                "feature dim error, feature: {}, expected: {}, actual: {}".format(k, dim, v.size(-1))
            feature_list.append(v.float())

        v = torch.cat(feature_list, 2)
        u = v.new_zeros(v.size(0), self._NWW, v.size(2))
        f = v.new_zeros(v.size(0), 1, v.size(2))
        v = torch.cat([u, v, f], 1).permute(0, 2, 1)

        v[self._mask[:, None, :].expand(v.size())] = 0

        norm = v.new_ones(self._mask.size())
        norm[self._mask] = 0
        norm = norm.sum(1) + 1e-10
        norm = norm[:, None, None]

        avg = v.sum(-1, keepdim=True) / norm
        v = v - avg

        std = v.norm(dim=-1, keepdim=True) / norm + 1e-10
        v = v / std
        return v.contiguous()


def save_variable_version(var):
    if isinstance(var.value, torch.Tensor):
        var.__version__ = var.value._version


def check_variable_version(var):
    if isinstance(var.value, torch.Tensor):
        assert var.__version__ == var.value._version, \
            "variable's value is modified, name: {}".format(var.name)
