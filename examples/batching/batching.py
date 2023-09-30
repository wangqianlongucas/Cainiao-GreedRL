import json

from greedrl import Problem, Solver
from greedrl.feature import *
from greedrl.variable import *

features = [local_feature('task_area'),
            local_feature('task_roadway'),
            local_feature('task_area_group'),
            sparse_local_feature('task_item_id', 'task_item_num'),
            sparse_local_feature('task_item_owner_id', 'task_item_num'),
            variable_feature('worker_task_item'),
            variable_feature('worker_used_roadway'),
            variable_feature('worker_used_area')]

variables = [task_demand_now('task_demand_now', feature='task_demand'),
             task_demand_now('task_demand_this', feature='task_demand', only_this=True),
             feature_variable('task_item_id'),
             feature_variable('task_item_num'),
             feature_variable('task_item_owner_id'),
             feature_variable('task_area'),
             feature_variable('task_area_group'),
             feature_variable('task_load'),
             feature_variable('task_group'),
             worker_variable('worker_load_limit'),
             worker_variable('worker_area_limit'),
             worker_variable('worker_area_group_limit'),
             worker_task_item('worker_task_item', item_id='task_item_id', item_num='task_item_num'),
             worker_task_item('worker_task_item_owner', item_id='task_item_owner_id', item_num='task_item_num'),
             worker_used_resource('worker_used_load', task_require='task_load'),
             worker_used_resource('worker_used_area', task_require='task_area'),
             worker_used_resource('worker_used_roadway', task_require='task_roadway'),
             worker_used_resource('worker_used_area_group', task_require='task_area_group')]


class Constraint:

    def do_task(self):
        return self.task_demand_this

    def mask_worker_end(self):
        return self.worker_used_load < self.worker_load_limit

    def mask_task(self):
        # 已经完成的任务
        mask = self.task_demand_now <= 0
        # mask |= task_group_priority(self.task_group, self.task_out_stock_time, mask)

        NT = self.task_item_id.size(1)
        worker_task_item = self.worker_task_item[:, None, :]
        worker_task_item = worker_task_item.expand(-1, NT, -1)
        task_item_in_worker = worker_task_item.gather(2, self.task_item_id.long())
        task_item_in_worker = (task_item_in_worker > 0) & (self.task_item_num > 0)

        worker_task_item_owner = self.worker_task_item_owner[:, None, :]
        worker_task_item_owner = worker_task_item_owner.expand(-1, NT, -1)
        task_item_owner_in_worker = worker_task_item_owner.gather(2, self.task_item_owner_id.long())
        task_item_owner_in_worker = (task_item_owner_in_worker > 0) & (self.task_item_num > 0)

        # 同一个sku，不同货主，不能在一个拣选单
        mask |= torch.any(task_item_in_worker & ~task_item_owner_in_worker, 2)

        worker_load_limit = self.worker_load_limit - self.worker_used_load
        mask |= (self.task_load > worker_load_limit[:, None])

        task_area = self.task_area + self.worker_used_area[:, None, :]
        task_area_num = task_area.clamp(0, 1).sum(2, dtype=torch.int32)
        mask |= (task_area_num > self.worker_area_limit[:, None])

        tak_area_group = self.task_area_group + self.worker_used_area_group[:, None, :]
        tak_area_group_num = tak_area_group.clamp(0, 1).sum(2, dtype=torch.int32)
        mask |= (tak_area_group_num > self.worker_area_group_limit[:, None])

        return mask

    def finished(self):
        return torch.all(self.task_demand_now <= 0, 1)


class Objective:

    def step_worker_end(self):
        area_num = self.worker_used_area.clamp(0, 1).sum(1)
        roadway_num = self.worker_used_roadway.clamp(0, 1).sum(1)
        item_num = self.worker_task_item.clamp(0, 1).sum(1)
        penalty = (self.worker_load_limit - self.worker_used_load) * 10
        return area_num * 100 + roadway_num * 10 + item_num + penalty


def make_problem_from_json(data):
    if isinstance(data, str):
        data = json.loads(data)
    problem = Problem()
    problem.id = data["id"]
    if 'uuid' in data:
        problem.uuid = data["uuid"]

    problem.task_item_id = torch.tensor(data["task_item_id"], dtype=torch.int32)
    problem.task_item_owner_id = torch.tensor(data["task_item_owner_id"], dtype=torch.int32)
    problem.task_item_num = torch.tensor(data["task_item_num"], dtype=torch.int32)
    problem.task_area = torch.tensor(data["task_area"], dtype=torch.int32)
    problem.task_roadway = torch.tensor(data["task_roadway"], dtype=torch.int32)
    problem.task_out_stock_time = torch.tensor(data["task_out_stock_time"], dtype=torch.int32)
    problem.task_area_group = torch.tensor(data["task_area_group"], dtype=torch.int32)

    NT = problem.task_item_id.size(0)
    problem.task_load = torch.ones(NT, dtype=torch.int32)
    problem.task_group = torch.zeros(NT, dtype=torch.int32)
    problem.task_demand = torch.ones(NT, dtype=torch.int32)

    problem.worker_load_limit = torch.tensor(data["worker_load_limit"], dtype=torch.int32)
    problem.worker_area_limit = torch.tensor(data["worker_area_limit"], dtype=torch.int32)
    problem.worker_area_group_limit = torch.tensor(data["worker_area_group_limit"], dtype=torch.int32)

    problem.features = features
    problem.variables = variables
    problem.constraint = Constraint
    problem.objective = Objective

    return problem


def make_problem(batch_count, batch_size=1, task_count=100):
    assert batch_size == 1

    NT = task_count
    problem_list = []
    for i in range(batch_count):
        problem = Problem()
        problem.id = i

        device = Solver().device
        p = torch.ones(NT, 1000, dtype=torch.float32, device=device)
        problem.task_item_id = torch.multinomial(p, 10).to(torch.int32).cpu()
        problem.task_item_owner_id = torch.multinomial(p, 10).to(torch.int32).cpu()
        problem.task_item_num = torch.randint(0, 5, (NT, 10), dtype=torch.int32)
        problem.task_area = torch.randint(0, 5, (NT, 10), dtype=torch.int32).clamp(0, 1)
        problem.task_roadway = torch.randint(0, 5, (NT, 200), dtype=torch.int32).clamp(0, 1)
        problem.task_area_group = torch.randint(0, 5, (NT, 10), dtype=torch.int32).clamp(0, 1)

        problem.task_load = torch.ones(NT, dtype=torch.int32)
        problem.task_group = torch.zeros(NT, dtype=torch.int32)
        problem.task_demand = torch.ones(NT, dtype=torch.int32)

        problem.worker_load_limit = torch.tensor([20], dtype=torch.int32)
        problem.worker_area_limit = torch.tensor([10], dtype=torch.int32)
        problem.worker_area_group_limit = torch.tensor([10], dtype=torch.int32)

        problem.features = features
        problem.variables = variables
        problem.constraint = Constraint
        problem.objective = Objective

        problem_list.append(problem)

    return problem_list


if __name__ == '__main__':
    import sys
    import os.path as osp
    sys.path.append(osp.join(osp.dirname(__file__), '../'))
    import runner

    runner.run(make_problem)
