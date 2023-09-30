import torch
import functools

from .utils import repeat


class VarMeta(object):
    def __init__(self, clazz, **kwargs):
        self.clazz = clazz
        self._kwargs = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __call__(self, problem, batch_size, sample_num):
        kwargs = self._kwargs.copy()
        kwargs['problem'] = problem.feats
        kwargs['batch_size'] = batch_size
        kwargs['sample_num'] = sample_num
        kwargs['worker_num'] = problem.worker_num
        kwargs['task_num'] = problem.task_num
        return self.clazz(**kwargs)


def attribute_variable(name, attribute=None):
    return VarMeta(AttributeVariable, name=name, attribute=attribute)


class AttributeVariable:
    def __init__(self, name, attribute, problem, batch_size, sample_num, worker_num, task_num):
        if attribute is None:
            attribute = name;

        self.name = name
        self.value = problem[attribute]


def feature_variable(name, feature=None):
    return VarMeta(FeatureVariable, name=name, feature=feature)


class FeatureVariable:
    def __init__(self, name, feature, problem, batch_size, sample_num, worker_num, task_num):
        if feature is None:
            feature = name

        assert feature == 'id' or feature.startswith("worker_") or feature.startswith("task_")

        self.name = name
        self.feature = problem[feature]
        self.value = repeat(self.feature, sample_num)


def task_variable(name, feature=None):
    return VarMeta(TaskVariable, name=name, feature=feature)


class TaskVariable:
    def __init__(self, name, feature, problem, batch_size, sample_num, worker_num, task_num):
        if feature is None:
            feature = name

        assert feature.startswith("task_")

        self.name = name
        self.feature = problem[feature]

        size = list(self.feature.size())
        size[0] = batch_size
        del size[1]
        self.value = self.feature.new_zeros(size)

    def step_task(self, b_index, p_index, t_index):
        self.value[b_index] = self.feature[p_index, t_index]


def worker_variable(name, feature=None):
    return VarMeta(WorkerVariable, name=name, feature=feature)


class WorkerVariable:
    def __init__(self, name, feature, problem, batch_size, sample_num, worker_num, task_num):
        if feature is None:
            feature = name

        assert feature.startswith("worker_")

        self.name = name
        self.feature = problem[feature]

        size = list(self.feature.size())
        size[0] = batch_size
        del size[1]
        self.value = self.feature.new_zeros(size)

    def step_worker_start(self, b_index, p_index, w_index):
        self.value[b_index] = self.feature[p_index, w_index]


def worker_task_variable(name, feature=None):
    return VarMeta(WorkerTaskVariable, name=name, feature=feature)


class WorkerTaskVariable:
    def __init__(self, name, feature, problem, batch_size, sample_num, worker_num, task_num):
        if feature is None:
            feature = name

        assert feature.startswith("worker_task_")

        self.name = name
        self.feature = problem[feature]

        size = list(self.feature.size())
        size[0] = batch_size

        del size[1]
        self._feature = self.feature.new_zeros(size)

        del size[2]
        self.value = self.feature.new_zeros(size)

    def step_worker_start(self, b_index, p_index, w_index):
        self._feature[b_index] = self.feature[p_index, w_index]

    def step_task(self, b_index, p_index, t_index):
        self.value[b_index] = self._feature[b_index, t_index]


def worker_task_group(name, feature=None):
    return VarMeta(WorkerTaskGroup, name=name, feature=feature)


class WorkerTaskGroup:
    def __init__(self, name, feature, problem, batch_size, sample_num, worker_num, task_num):
        if feature is None:
            feature = name

        assert feature.startswith("task_")

        self.name = name
        self.feature = problem[feature].long()

        NG = self.feature.max() + 1
        assert torch.all(self.feature >= 0)

        self.value = self.feature.new_zeros(batch_size, NG)

    def step_worker_start(self, b_index, p_index, w_index):
        self.value[b_index] = 0

    def step_task(self, b_index, p_index, t_index):
        group = self.feature[p_index, t_index]
        self.value[b_index, group] += 1;


def worker_task_item(name, item_id, item_num):
    return VarMeta(WorkerTaskItem, name=name, item_id=item_id, item_num=item_num)


class WorkerTaskItem:
    def __init__(self, name, item_id, item_num, problem, batch_size, sample_num, worker_num, task_num):
        assert item_id.startswith('task_')
        assert item_num.startswith('task_')

        self.name = name
        self.item_id = repeat(problem[item_id], sample_num).long()
        self.item_num = repeat(problem[item_num], sample_num)

        assert torch.all(self.item_id >= 0)

        size = [0, 0]
        size[0] = self.item_id.size(0)
        size[1] = self.item_id.max() + 1
        self.value = self.item_num.new_zeros(size)

    def step_worker_start(self, b_index, p_index, w_index):
        self.value[b_index] = 0

    def step_task(self, b_index, p_index, t_index):
        item_id = self.item_id[b_index, t_index]
        item_num = self.item_num[b_index, t_index]
        self.value[b_index[:, None], item_id] += item_num

    def make_feat(self):
        NT = self.item_id.size(1)
        v = self.value[:, None, :]
        v = v.expand(-1, NT, -1)

        v = v.gather(2, self.item_id).clamp(0, 1)
        v = self.item_num.clamp(0, 1) - v
        return v.clamp(0, 1).sum(2)


def task_demand_now(name, feature=None, only_this=False):
    return VarMeta(TaskDemandNow, name=name, feature=feature, only_this=only_this)


class TaskDemandNow:
    def __init__(self, name, feature, only_this, problem, batch_size, sample_num, worker_num, task_num):

        if feature is None:
            feature = name

        assert feature.startswith("task_")

        self.name = name
        self.only_this = only_this
        self._value = repeat(problem[feature], sample_num)

        assert self._value.dtype in \
               (torch.int8, torch.int16, torch.int32, torch.int64)
        assert torch.all(self._value >= 0)

        if only_this:
            size = self._value.size(0)
            self.value = self._value.new_zeros(size)
        else:
            self.value = self._value

    def step_task(self, b_index, p_index, t_index, done):
        if done is not None:
            self._value[b_index, t_index] -= done

        if self.only_this:
            self.value[b_index] = self._value[b_index, t_index]
        else:
            self.value = self._value


def worker_count_now(name, feature=None):
    return VarMeta(WorkerCountNow, name=name, feature=feature)


class WorkerCountNow:
    def __init__(self, name, feature, problem, batch_size, sample_num, worker_num, task_num):
        if feature is None:
            feature = name

        assert feature.startswith("worker_")

        self.name = name
        self.value = repeat(problem[feature], sample_num)

        assert self.value.dtype in \
               (torch.int8, torch.int16, torch.int32, torch.int64)
        assert torch.all(self.value >= 0)

    def step_worker_start(self, b_index, p_index, w_index):
        self.value[b_index, w_index] -= 1


def edge_variable(name, feature, last_to_this=False,
                  this_to_task=False, task_to_end=False, last_to_loop=False):
    return VarMeta(EdgeVariable, name=name, feature=feature,
                   last_to_this=last_to_this, this_to_task=this_to_task, task_to_end=task_to_end,
                   last_to_loop=last_to_loop)


class EdgeVariable:
    def __init__(self, name, feature, last_to_this, this_to_task, task_to_end, last_to_loop,
                 problem, batch_size, sample_num, worker_num, task_num):

        assert feature.endswith("_matrix")

        flags = [last_to_this, this_to_task, task_to_end, last_to_loop]
        assert flags.count(True) == 1 and flags.count(False) == 3

        if feature is None:
            feature = name

        self.name = name
        self.last_to_this = last_to_this
        self.this_to_task = this_to_task
        self.task_to_end = task_to_end
        self.last_to_loop = last_to_loop

        self.worker_num = worker_num
        self.task_num = task_num

        self.feature = problem[feature]

        size = list(self.feature.size())
        size[0] = batch_size
        del size[1:3]

        if self.this_to_task or self.task_to_end:
            size.insert(1, task_num)
            self.value = self.feature.new_zeros(size)
        else:
            self.value = self.feature.new_zeros(size)

        self.end_index = self.feature.new_zeros(size[0], dtype=torch.int64)
        self.loop_index = self.feature.new_zeros(size[0], dtype=torch.int64)
        self.last_index = self.feature.new_zeros(size[0], dtype=torch.int64)
        self.task_index = (torch.arange(task_num) + worker_num * 2)[None, :]

    def step_worker_start(self, b_index, p_index, w_index):
        if self.last_to_this:
            self.value[b_index] = 0
            self.last_index[b_index] = w_index
        elif self.this_to_task:
            self.do_this_to_task(b_index, p_index, w_index)
        elif self.task_to_end:
            self.end_index[b_index] = w_index + self.worker_num
            self.do_task_to_end(b_index, p_index)
        elif self.last_to_loop:
            self.value[b_index] = 0
            self.last_index[b_index] = w_index

    def step_worker_end(self, b_index, p_index, w_index):
        this_index = w_index + self.worker_num
        if self.last_to_this:
            self.do_last_to_this(b_index, p_index, this_index)
        elif self.this_to_task:
            self.do_this_to_task(b_index, p_index, this_index)
        elif self.task_to_end:
            pass
        elif self.last_to_loop:
            self.do_last_to_loop(b_index, p_index)

    def step_task(self, b_index, p_index, t_index):
        this_index = t_index + self.worker_num * 2
        if self.last_to_this:
            self.do_last_to_this(b_index, p_index, this_index)
            self.last_index[b_index] = this_index
        elif self.this_to_task:
            self.do_this_to_task(b_index, p_index, this_index)
        elif self.task_to_end:
            pass
        elif self.last_to_loop:
            last_index = self.last_index[b_index]
            loop_index = self.loop_index[b_index]
            self.loop_index[b_index] = torch.where(last_index < self.worker_num, this_index, loop_index)
            self.last_index[b_index] = this_index

    def do_last_to_this(self, b_index, p_index, this_index):
        last_index = self.last_index[b_index]
        self.value[b_index] = self.feature[p_index, last_index, this_index]

    def do_this_to_task(self, b_index, p_index, this_index):
        p_index2 = p_index[:, None]
        this_index2 = this_index[:, None]
        task_index2 = self.task_index
        self.value[b_index] = self.feature[p_index2, this_index2, task_index2]

    def do_task_to_end(self, b_index, p_index):
        p_index2 = p_index[:, None]
        task_index2 = self.task_index
        end_index = self.end_index[b_index]
        end_index2 = end_index[:, None]
        self.value[b_index] = self.feature[p_index2, task_index2, end_index2]

    def do_last_to_loop(self, b_index, p_index):
        loop_index = self.loop_index[b_index]
        last_index = self.last_index[b_index]
        self.value[b_index] = self.feature[p_index, last_index, loop_index]

    def make_feat(self):
        assert self.this_to_task or self.task_to_end, \
            "one of [this_to_task, task_to_end] must be true"
        return self.value.clone()


def worker_used_resource(name, edge_require=None, task_require=None, task_ready=None, worker_ready=None, task_due=None):
    return VarMeta(WorkerUsedResource, name=name, edge_require=edge_require, task_require=task_require,
                   task_ready=task_ready, worker_ready=worker_ready, task_due=task_due)


class WorkerUsedResource:
    def __init__(self, name, edge_require, task_require, task_ready, worker_ready, task_due,
                 problem, batch_size, sample_num, worker_num, task_num):

        assert edge_require is None or edge_require.endswith("_matrix"), "unsupported edge: {}".format(edge_require)
        assert task_require is None or task_require.startswith("task_"), "unsupported task_require: {}".format(
            task_require)
        assert task_ready is None or task_ready.startswith("task_"), "unsupported task_service: {}".format(task_ready)
        assert worker_ready is None or worker_ready.startswith("worker_") and not worker_ready.startswith(
            "worker_task_")
        assert task_due is None or task_due.startswith("task_"), "unsupported task_due: {}".format(task_due)

        self.name = name

        self.worker_num = worker_num
        self.task_num = task_num

        if edge_require is None:
            self.edge_require = None
        else:
            self.edge_require = problem[edge_require]
            self.last_index = self.edge_require.new_zeros(batch_size, dtype=torch.int64)

        if task_require is None:
            self.task_require = None
        else:
            self.task_require = problem[task_require]
            self.task_require2 = repeat(self.task_require, sample_num)

        if task_ready is None:
            self.task_ready = None
        else:
            self.task_ready = problem[task_ready]

        if worker_ready is None:
            self.worker_ready = None
        else:
            self.worker_ready = problem[worker_ready]

        if task_due is None:
            self.task_due = None
        else:
            self.task_due = problem[task_due]

        tenors = [self.edge_require, self.task_require, self.task_ready, self.worker_ready]
        tenors = list(filter(lambda x: x is not None, tenors))
        assert tenors, "at least one of edge_require, task_require, task_ready, worker_ready is required!"

        size = list(tenors[0].size())
        size[0] = batch_size
        if self.edge_require is None:
            del size[1]
        else:
            del size[1:3]

        self.value = tenors[0].new_zeros(size)

    def step_worker_start(self, b_index, p_index, w_index):
        if self.worker_ready is None:
            self.value[b_index] = 0
        else:
            self.value[b_index] = self.worker_ready[p_index, w_index]

        if self.edge_require is not None:
            self.last_index[b_index] = w_index

    def step_worker_end(self, b_index, p_index, w_index):
        if self.edge_require is not None:
            last_index = self.last_index[b_index]
            this_index = w_index + self.worker_num
            self.value[b_index] += self.edge_require[p_index, last_index, this_index]
            self.last_index[b_index] = this_index;

    def step_task(self, b_index, p_index, t_index, done):
        if done is None:
            if self.edge_require is not None:
                last_index = self.last_index[b_index]
                this_index = t_index + (self.worker_num * 2)
                self.value[b_index] += self.edge_require[p_index, last_index, this_index]
                self.last_index[b_index] = this_index

            if self.task_ready is not None:
                self.value[b_index] = torch.max(self.value[b_index], self.task_ready[p_index, t_index])

        else:
            if self.task_require is not None:
                if self.value.dim() == 2:
                    done = done[:, None]
                self.value[b_index] += self.task_require[p_index, t_index] * done

    def make_feat(self):
        assert self.value.dim() == 2, \
            "value's dim must be 2, actual: {}".format(self.value.dim())
        assert self.task_require is not None, "task_require is required"

        v = self.value[:, None, :] + self.task_require2
        return v.clamp(0, 1).sum(2, dtype=v.dtype)


def worker_task_sequence(name):
    return VarMeta(WorkerTaskSequence, name=name)


class WorkerTaskSequence:
    def __init__(self, name, problem, batch_size, sample_num, worker_num, task_num):
        self.name = name
        self.value = None

    def step_finish(self, worker_task_seq):
        self.value = worker_task_seq
