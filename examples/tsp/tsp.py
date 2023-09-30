from greedrl.feature import *
from greedrl.variable import *
from greedrl import Problem

features = [continuous_feature('task_location'),
            variable_feature('distance_this_to_task'),
            variable_feature('distance_task_to_end')]

variables = [task_demand_now('task_demand_now', feature='task_demand'),
             task_demand_now('task_demand_this', feature='task_demand', only_this=True),
             edge_variable('distance_last_to_this', feature='distance_matrix', last_to_this=True),
             edge_variable('distance_this_to_task', feature='distance_matrix', this_to_task=True),
             edge_variable('distance_task_to_end', feature='distance_matrix', task_to_end=True),
             edge_variable('distance_last_to_loop', feature='distance_matrix', last_to_loop=True)]


class Constraint:

    def do_task(self):
        return self.task_demand_this

    def mask_task(self):
        # 已经完成的任务
        mask = self.task_demand_now <= 0
        return mask

    def mask_worker_end(self):
        return torch.any(self.task_demand_now > 0, 1)

    def finished(self):
        return torch.all(self.task_demand_now <= 0, 1)


class Objective:

    def step_worker_end(self):
        return self.distance_last_to_loop

    def step_task(self):
        return self.distance_last_to_this


def make_problem(batch_count, batch_size=1, task_count=100):
    NP = batch_size
    NT = task_count
    problem_list = []
    for i in range(batch_count):
        problem = Problem(True)

        problem.task_demand = torch.ones(NP, NT, dtype=torch.int32)

        loc = torch.rand(NP, NT + 1, 2, dtype=torch.float32)
        problem.distance_matrix = torch.norm(loc[:, :, None, :] - loc[:, None, :, :], dim=3)
        problem.distance_matrix[0, :] = 0
        problem.distance_matrix[:, 0] = 0

        problem.task_location = loc[:, 1:]

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
