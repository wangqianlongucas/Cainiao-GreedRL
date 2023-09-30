from greedrl.feature import *
from greedrl.variable import *
from greedrl.function import *
from greedrl import Problem

features = [local_category('task_group'),
            global_category('task_priority', 2),
            variable_feature('distance_this_to_task'),
            variable_feature('distance_task_to_end')]

variables = [task_demand_now('task_demand_now', feature='task_demand'),
             task_demand_now('task_demand_this', feature='task_demand', only_this=True),
             feature_variable('task_weight'),
             feature_variable('task_group'),
             feature_variable('task_priority'),
             feature_variable('task_due_time2', feature='task_due_time'),
             task_variable('task_due_time'),
             task_variable('task_service_time'),
             task_variable('task_due_time_penalty'),
             worker_variable('worker_basic_cost'),
             worker_variable('worker_distance_cost'),
             worker_variable('worker_due_time'),
             worker_variable('worker_weight_limit'),
             worker_used_resource('worker_used_weight', task_require='task_weight'),
             worker_used_resource('worker_used_time', 'distance_matrix', 'task_service_time', 'task_ready_time',
                                  'worker_ready_time'),
             edge_variable('distance_last_to_this', feature='distance_matrix', last_to_this=True),
             edge_variable('distance_this_to_task', feature='distance_matrix', this_to_task=True),
             edge_variable('distance_task_to_end', feature='distance_matrix', task_to_end=True)]


class Constraint:

    def do_task(self):
        return self.task_demand_this

    def mask_worker_end(self):
        return task_group_split(self.task_group, self.task_demand_now <= 0)

    def mask_task(self):
        mask = self.task_demand_now <= 0
        mask |= task_group_priority(self.task_group, self.task_priority, mask)

        worker_used_time = self.worker_used_time[:, None] + self.distance_this_to_task
        mask |= (worker_used_time > self.task_due_time2) & (self.task_priority == 0)

        # 容量约束
        worker_weight_limit = self.worker_weight_limit - self.worker_used_weight
        mask |= self.task_demand_now * self.task_weight > worker_weight_limit[:, None]
        return mask

    def finished(self):
        return torch.all(self.task_demand_now <= 0, 1)


class Objective:

    def step_worker_start(self):
        return self.worker_basic_cost

    def step_worker_end(self):
        feasible = self.worker_used_time <= self.worker_due_time
        return self.distance_last_to_this * self.worker_distance_cost, feasible

    def step_task(self):
        worker_used_time = self.worker_used_time - self.task_service_time
        feasible = worker_used_time <= self.task_due_time
        feasible &= worker_used_time <= self.worker_due_time
        cost = self.distance_last_to_this * self.worker_distance_cost
        return torch.where(feasible, cost, cost + self.task_due_time_penalty), feasible


def make_problem(batch_count, batch_size=1, task_count=100):
    assert batch_size == 1

    N = task_count // 2  # 订单数, 一个订单有pickup， delivery两个任务
    problem_list = []
    for i in range(batch_count):
        problem = Problem()
        problem.id = i

        problem.worker_weight_limit = torch.tensor([50], dtype=torch.float32)
        problem.worker_ready_time = torch.tensor([0], dtype=torch.float32)
        problem.worker_due_time = torch.tensor([1000000], dtype=torch.float32)
        problem.worker_basic_cost = torch.tensor([100], dtype=torch.float32)
        problem.worker_distance_cost = torch.tensor([1], dtype=torch.float32)

        task_demand = torch.randint(1, 10, (N,), dtype=torch.int32)
        problem.task_demand = torch.cat([task_demand, task_demand], 0)

        task_weight = torch.ones(N, dtype=torch.float32)
        problem.task_weight = torch.cat([task_weight, task_weight * -1], 0)

        task_group = torch.arange(N, dtype=torch.int32)
        problem.task_group = torch.cat([task_group, task_group], 0)

        task_priority = torch.zeros(N, dtype=torch.int32)
        problem.task_priority = torch.cat([task_priority, task_priority + 1], 0)

        task_ready_time = torch.zeros(N, dtype=torch.float32)
        problem.task_ready_time = torch.cat([task_ready_time, task_ready_time], 0)

        task_due_time = torch.randint(10000, 100000, (N,), dtype=torch.float32)
        problem.task_due_time = torch.cat([task_due_time, task_due_time * 2], 0)

        task_service_time = torch.zeros(N, dtype=torch.float32)
        problem.task_service_time = torch.cat([task_service_time, task_service_time])

        task_due_time_penalty = torch.ones(N, dtype=torch.float32)
        problem.task_due_time_penalty = torch.cat([task_due_time_penalty, task_due_time_penalty])

        loc = torch.rand(N + 1, 2, dtype=torch.float32)
        distance_matrix = torch.norm(loc[:, None, :] - loc[None, :, :], dim=2) * 1000
        distance_matrix = distance_matrix.to(torch.float32)
        index = torch.cat([torch.zeros(N + 1, dtype=torch.int64), torch.arange(N, dtype=torch.int64) + 1])
        index1 = index[:, None]
        index2 = index[None, :]
        problem.distance_matrix = distance_matrix[index1, index2]

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
