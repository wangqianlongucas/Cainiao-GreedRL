import json

from greedrl.feature import *
from greedrl.variable import *
from greedrl.function import *
from greedrl import Problem

features = [local_category('task_order'),
            global_category('task_type', 2),
            global_category('task_new_order', 2),
            variable_feature('time_this_to_task'),
            continuous_feature('x_time_matrix'),
            continuous_feature('task_due_time_x'),
            continuous_feature('worker_task_mask')]

variables = [task_demand_now('task_demand_now', feature='task_demand'),
             task_demand_now('task_demand_this', feature='task_demand', only_this=True),
             task_variable('task_pickup_this', feature='task_pickup'),
             task_variable('task_due_time_this', feature='task_due_time'),
             feature_variable('task_order', feature='task_order'),
             feature_variable('task_type', feature='task_type'),
             feature_variable('task_new_pickup', feature='task_new_pickup'),
             feature_variable('worker_task_mask', feature='worker_task_mask'),
             worker_count_now('worker_count_now', feature='worker_count'),
             worker_variable('worker_min_old_task_this', feature='worker_min_old_task'),
             worker_variable('worker_max_new_order_this', feature='worker_max_new_order'),
             worker_variable('worker_task_mask_this', feature='worker_task_mask'),
             worker_used_resource('worker_used_old_task', task_require='task_old'),
             worker_used_resource('worker_used_new_order', task_require='task_new_pickup'),
             worker_used_resource('worker_used_time', edge_require='time_matrix'),
             edge_variable('time_this_to_task', feature='x_time_matrix', this_to_task=True)]


class Constraint:

    def do_task(self):
        return self.task_demand_this

    def mask_worker_start(self):
        mask = self.worker_count_now <= 0

        finished = self.task_demand_now <= 0
        worker_task_mask = self.worker_task_mask | finished[:, None, :]
        mask |= torch.all(worker_task_mask, 2)

        return mask

    def mask_worker_end(self):
        mask = self.worker_used_old_task < self.worker_min_old_task_this
        mask |= task_group_split(self.task_order, self.task_demand_now <= 0)
        return mask

    def mask_task(self):
        mask = self.task_demand_now <= 0

        mask |= task_group_priority(self.task_order, self.task_type, mask)

        worker_max_new_order = self.worker_max_new_order_this - self.worker_used_new_order
        mask |= self.task_new_pickup > worker_max_new_order[:, None]

        mask |= self.worker_task_mask_this

        return mask

    def finished(self):
        worker_mask = self.worker_count_now <= 0
        task_mask = self.task_demand_now <= 0
        worker_task_mask = worker_mask[:, :, None] | task_mask[:, None, :]

        worker_task_mask |= self.worker_task_mask
        batch_size = worker_task_mask.size(0)
        worker_task_mask = worker_task_mask.view(batch_size, -1)
        return worker_task_mask.all(1)


class Objective:

    def step_task(self):
        over_time = (self.worker_used_time - self.task_due_time_this).clamp(min=0)
        pickup_time = self.worker_used_time * self.task_pickup_this
        return self.worker_used_time + over_time + pickup_time

    def step_finish(self):
        return self.task_demand_now.sum(1) * 1000


def preprocess(problem):
    NW, NT = problem.worker_task_mask.size()

    worker_task_old = torch.ones(NW, NT, dtype=torch.int32)
    new_task_mask = problem.task_new_order[None, :].expand(NW, NT)
    worker_task_old[new_task_mask] = 0
    worker_task_old[problem.worker_task_mask] = 0
    assert torch.all(worker_task_old.sum(0) <= 1)
    problem.worker_min_old_task = worker_task_old.sum(1)

    problem.worker_count = torch.ones(NW, dtype=torch.int32)
    problem.task_demand = torch.ones(NT, dtype=torch.int32)
    problem.task_pickup = (problem.task_type == 0).to(torch.int32)

    task_old = torch.ones(NT, dtype=torch.int32)
    task_old[problem.task_new_order] = 0
    problem.task_old = task_old

    task_new_pickup = torch.ones(NT, dtype=torch.int32)
    task_new_pickup[problem.task_type >= 1] = 0
    task_new_pickup[~problem.task_new_order] = 0
    problem.task_new_pickup = task_new_pickup

    problem.task_due_time_x = problem.task_due_time.float() / 900
    problem.x_time_matrix = problem.time_matrix.float() / 900

    problem.features = features
    problem.variables = variables
    problem.constraint = Constraint
    problem.objective = Objective

    return problem


def make_problem_from_json(data):
    data = json.loads(data)

    problem = Problem()

    problem.id = data['id']
    problem.task_order = torch.tensor(data['task_order'], dtype=torch.int32)
    problem.task_type = torch.tensor(data['task_type'], dtype=torch.int32)
    problem.task_new_order = torch.tensor(data['task_new_order'], dtype=torch.bool)
    problem.task_due_time = torch.tensor(data['task_due_time'], dtype=torch.int32)

    problem.worker_max_new_order = torch.tensor(data['worker_max_new_order'], dtype=torch.int32)
    problem.worker_task_mask = torch.tensor(data['worker_task_mask'], dtype=torch.bool)
    problem.time_matrix = torch.tensor(data['time_matrix'], dtype=torch.int32)

    NW, NT = problem.worker_task_mask.size()

    assert problem.task_order.size() == (NT,), "task_order size error"
    assert problem.task_type.size() == (NT,), "task_type size error"
    assert problem.task_new_order.size() == (NT,), "task_new_order size error"
    assert problem.task_due_time.size() == (NT,), "task_due_time size error"
    assert problem.worker_max_new_order.size() == (NW,), "worker_max_new_order size error"
    assert problem.time_matrix.size() == (NW + NT, NW + NT), "time_matrix size error"

    return preprocess(problem)


def make_problem(batch_count, batch_size=1, task_count=100):
    assert batch_size == 1
    assert task_count == 100

    NW = 100
    NT = task_count
    NO = NT // 2  # 订单数, 一个订单有pickup， delivery两个任务
    problem_list = []
    for i in range(batch_count):
        problem = Problem()

        # user-provided data
        problem.worker_max_new_order = torch.full((NW,), 2, dtype=torch.int32)

        task_order = torch.arange(NO, dtype=torch.int32)
        problem.task_order = torch.cat([task_order, task_order], 0)

        task_type = torch.zeros(NO, dtype=torch.int32)
        problem.task_type = torch.cat([task_type, task_type + 1], 0)

        problem.task_new_order = torch.ones(NT, dtype=torch.bool)

        task_due_time = torch.randint(1000, 1800, (NO,), dtype=torch.int32)
        problem.task_due_time = torch.cat([task_due_time, task_due_time + 1800], 0)

        worker_task_mask = torch.rand(NW, NO) < 0.9
        problem.worker_task_mask = torch.cat([worker_task_mask, worker_task_mask], 1)

        loc = torch.rand(NW + NT, 2, dtype=torch.float32)
        time_matrix = torch.norm(loc[:, None, :] - loc[None, :, :], dim=2) * 1000
        problem.time_matrix = time_matrix.to(torch.int32)

        problem_list.append(preprocess(problem))

    return problem_list


if __name__ == '__main__':
    import sys
    import os.path as osp
    sys.path.append(osp.join(osp.dirname(__file__), '../'))
    import runner

    runner.run(make_problem)
