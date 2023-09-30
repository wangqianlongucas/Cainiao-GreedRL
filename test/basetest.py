import torch
import unittest


class TestCase(unittest.TestCase):
    def tearDown(self):
        torch.cuda.empty_cache()

