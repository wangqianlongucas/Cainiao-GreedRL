import sys
import time
import torch
import unittest
import basetest

from greedrl import Solver
from greedrl.function import *

device = Solver().device


class TestFunction(basetest.TestCase):

    def test_task_group_split(self):
        group = torch.ones((8, 8), dtype=torch.int32)
        group[:, 0:4] = 0
        value = torch.zeros((8, 8), dtype=torch.bool)
        value[:, 0:4] = True
        result = task_group_split(group, value)
        assert not torch.any(result)

        value[:, 0:2] = False
        result = task_group_split(group, value)
        assert torch.all(result)

    def test_task_group_split2(self):
        group = torch.randint(48, (1024, 1000), dtype=torch.int32)
        value = torch.randint(2, (1024, 1000), dtype=torch.int8) <= 0
        self.do_test(task_group_split, group, value)

    def test_task_group_priority(self):
        group = torch.ones((8, 8), dtype=torch.int32)
        group[:, 0:4] = 0
        priority = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.int32)
        priority = priority[None, :].expand(8, -1).clone()
        value = torch.zeros((8, 8), dtype=torch.bool)
        value[:, 4:6] = True

        result = task_group_priority(group, priority, value)
        expected = torch.tensor([False, True, True, True, True, True, False, True])
        expected = expected[None, :].expand(8, -1)
        assert torch.all(result == expected)

    def test_task_group_priority2(self):
        group = torch.randint(48, (1024, 1000), dtype=torch.int32)
        value = torch.randint(2, (1024, 1000), dtype=torch.int8) < 1
        priority = torch.randint(2, (1024, 1000), dtype=torch.int32)
        self.do_test(task_group_priority, group, priority, value)

    def do_test(self, function, *args):
        print("\ntest {} ...".format(function.__name__))
        start = time.time()
        result1 = function(*args)
        print("time: {:.6f}s, device: {}".format(time.time() - start, args[0].device))

        args = [arg.to(device) for arg in args]
        result1 = result1.to(device)

        function(*args)
        self.sync_device(device)

        start = time.time()
        result2 = function(*args)
        self.sync_device(device)
        print("time: {:.6f}s, device: {} ".format(time.time() - start, args[0].device))

        if result1.is_floating_point():
            assert torch.all(torch.abs(result1 - result2) < 1e-6)
        else:
            assert torch.all(result1 == result2)

    def sync_device(self, device):
        if device.type == 'cuda':
            torch.cuda.synchronize(device)


if __name__ == '__main__':
    unittest.main()
