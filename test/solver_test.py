import sys
import os.path as osp
import torch
import unittest

import basetest
from greedrl import Solver
from greedrl.const import *

sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), "../"))
from examples.cvrp import cvrp


class TestSolver(basetest.TestCase):
    def test(self):
        problem_list = cvrp.make_problem(1)

        nn_args = {}
        nn_args['decode_rnn'] = 'GRU'
        solver = Solver(None, nn_args)

        solver.train(None, problem_list, problem_list,
                     batch_size=32, max_steps=5, memopt=10)

        solver.train(None, problem_list, problem_list,
                     batch_size=32, max_steps=5, memopt=10, topk_size=10)

        solver.train(None, problem_list, problem_list,
                     batch_size=32, max_steps=5, memopt=10, on_policy=False)

        solution = solver.solve(problem_list[0], batch_size=8)
        assert torch.all(solution.worker_task_sequence[:, -1, 0] == GRL_FINISH)
        problem_list[0].solution = solution.worker_task_sequence[:, 0:-1, :]

        solution2 = solver.solve(problem_list[0], batch_size=1)
        assert torch.all(solution.worker_task_sequence == solution2.worker_task_sequence)


if __name__ == '__main__':
    unittest.main()
