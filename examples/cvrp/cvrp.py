from greedrl.feature import *
from greedrl.variable import *
from greedrl import Problem

features = [continuous_feature('task_demand'),
            continuous_feature('worker_weight_limit'),
            continuous_feature('distance_matrix'),
            variable_feature('distance_this_to_task'),
            variable_feature('distance_task_to_end')]

variables = [task_demand_now('task_demand_now', feature='task_demand'),
             task_demand_now('task_demand_this', feature='task_demand', only_this=True),
             feature_variable('task_weight'),
             worker_variable('worker_weight_limit'),
             worker_used_resource('worker_used_weight', task_require='task_weight'),
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
        return mask

    def finished(self):
        return torch.all(self.task_demand_now <= 0, 1)


class Objective:

    def step_worker_end(self):
        return self.distance_last_to_this

    def step_task(self):
        return self.distance_last_to_this


def make_problem(batch_count, batch_size=1, task_count=100):
    assert task_count in (100, 1000, 2000, 5000)

    weight_limit = 50
    problem_list = []
    for i in range(batch_count):
        problem = Problem(True)
        problem.id = torch.arange(batch_size) + i * batch_size;

        problem.worker_weight_limit = torch.full((batch_size, 1), weight_limit, dtype=torch.int32)

        N = task_count
        problem.task_demand = torch.randint(1, 10, (batch_size, N), dtype=torch.int32)
        problem.task_demand_x = problem.task_demand.float() / weight_limit

        # 一个单位的task_demand的重量
        problem.task_weight = torch.ones(batch_size, N, dtype=torch.int32)

        loc = torch.rand(batch_size, N + 1, 2, dtype=torch.float32)
        problem.task_location = loc[:, 1:, :]
        problem.worker_location = loc[:, 0:1, :]

        distance_matrix = torch.norm(loc[:, :, None, :] - loc[:, None, :, :], dim=3)
        problem.distance_matrix = distance_matrix

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
