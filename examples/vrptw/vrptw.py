import json

from greedrl import Problem
from greedrl.feature import *
from greedrl.variable import *

features = [continuous_feature('worker_weight_limit'),
            continuous_feature('worker_ready_time'),
            continuous_feature('worker_due_time'),
            continuous_feature('worker_basic_cost'),
            continuous_feature('worker_distance_cost'),
            continuous_feature('task_demand'),
            continuous_feature('task_weight'),
            continuous_feature('task_ready_time'),
            continuous_feature('task_due_time'),
            continuous_feature('task_service_time'),
            continuous_feature('distance_matrix')]

variables = [task_demand_now('task_demand_now', feature='task_demand'),
             task_demand_now('task_demand_this', feature='task_demand', only_this=True),
             feature_variable('task_weight'),
             feature_variable('task_due_time'),
             feature_variable('task_ready_time'),
             feature_variable('task_service_time'),
             worker_variable('worker_weight_limit'),
             worker_variable('worker_due_time'),
             worker_variable('worker_basic_cost'),
             worker_variable('worker_distance_cost'),
             worker_used_resource('worker_used_weight', task_require='task_weight'),
             worker_used_resource('worker_used_time', 'distance_matrix', 'task_service_time', 'task_ready_time',
                                  'worker_ready_time'),
             edge_variable('distance_last_to_this', feature='distance_matrix', last_to_this=True),
             edge_variable('distance_this_to_task', feature='distance_matrix', this_to_task=True),
             edge_variable('distance_task_to_end', feature='distance_matrix', task_to_end=True)]


class Constraint:

    def do_task(self):
        return self.task_demand_this

    def mask_task(self):
        # 已经完成的任务
        mask = self.task_demand_now <= 0
        # 车辆容量限制
        worker_weight_limit = self.worker_weight_limit - self.worker_used_weight
        mask |= self.task_demand_now * self.task_weight > worker_weight_limit[:, None]

        worker_used_time = self.worker_used_time[:, None] + self.distance_this_to_task
        mask |= worker_used_time > self.task_due_time

        worker_used_time = torch.max(worker_used_time, self.task_ready_time)
        worker_used_time += self.task_service_time
        worker_used_time += self.distance_task_to_end
        mask |= worker_used_time > self.worker_due_time[:, None]

        return mask

    def finished(self):
        return torch.all(self.task_demand_now <= 0, 1)


class Objective:

    def step_worker_start(self):
        return self.worker_basic_cost

    def step_worker_end(self):
        return self.distance_last_to_this * self.worker_distance_cost

    def step_task(self):
        return self.distance_last_to_this * self.worker_distance_cost


def make_problem_from_json(data):
    if isinstance(data, str):
        data = json.loads(data)

    problem = Problem()
    problem.worker_weight_limit = torch.tensor(data['worker_weight_limit'], dtype=torch.float32)
    problem.worker_ready_time = torch.tensor(data['worker_ready_time'], dtype=torch.float32)
    problem.worker_due_time = torch.tensor(data['worker_due_time'], dtype=torch.float32)
    problem.worker_basic_cost = torch.tensor(data['worker_basic_cost'], dtype=torch.float32)
    problem.worker_distance_cost = torch.tensor(data['worker_distance_cost'], dtype=torch.float32)

    problem.task_demand = torch.tensor(data['task_demand'], dtype=torch.int32)
    problem.task_weight = torch.tensor(data['task_weight'], dtype=torch.float32)
    problem.task_ready_time = torch.tensor(data['task_ready_time'], dtype=torch.float32)
    problem.task_due_time = torch.tensor(data['task_due_time'], dtype=torch.float32)
    problem.task_service_time = torch.tensor(data['task_service_time'], dtype=torch.float32)

    problem.distance_matrix = torch.tensor(data['distance_matrix'], dtype=torch.float32);

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

        problem.worker_weight_limit = torch.tensor([50], dtype=torch.float32)
        problem.worker_ready_time = torch.tensor([0], dtype=torch.float32)
        problem.worker_due_time = torch.tensor([1000000], dtype=torch.float32)
        problem.worker_basic_cost = torch.tensor([100], dtype=torch.float32)
        problem.worker_distance_cost = torch.tensor([1], dtype=torch.float32)

        problem.task_demand = torch.randint(1, 10, (NT,), dtype=torch.int32)
        problem.task_weight = torch.ones(NT, dtype=torch.float32)
        problem.task_ready_time = torch.zeros(NT, dtype=torch.float32)
        problem.task_due_time = torch.randint(10000, 100000, (NT,), dtype=torch.float32)
        problem.task_service_time = torch.zeros(NT, dtype=torch.float32)

        loc = torch.rand(NT + 1, 2, dtype=torch.float32)
        problem.distance_matrix = torch.norm(loc[:, None, :] - loc[None, :, :], dim=2) * 1000
        problem_list.append(problem)

        problem.features = features
        problem.variables = variables
        problem.constraint = Constraint
        problem.objective = Objective

    return problem_list


if __name__ == '__main__':
    import sys
    import os.path as osp
    sys.path.append(osp.join(osp.dirname(__file__), '../'))
    import runner

    runner.run(make_problem)
