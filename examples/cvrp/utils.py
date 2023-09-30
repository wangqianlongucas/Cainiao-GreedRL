from greedrl.feature import *
from cvrp import make_problem as make_cvrp_problem
from torch.utils.data import Dataset, IterableDataset, DataLoader


def make_problem(batch_count, batch_size, task_count):
    features = [continuous_feature('task_demand_x'),
                continuous_feature('distance_matrix')]

    problem_list = make_cvrp_problem(batch_count, batch_size, task_count)
    for problem in problem_list:
        problem.features = features

    return problem_list


class Dataset(IterableDataset):
    def __init__(self, batch_count, batch_size, task_count):
        self._batch_size = batch_size
        self._task_count = task_count
        self._batch_count = batch_count
        self._index = 0

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._batch_count is not None \
                and self._index >= self._batch_count:
            raise StopIteration()

        p = make_problem(1, self._batch_size, self._task_count)[0]
        self._index += 1
        return p


def write_vrplib(filename, name, size, demand, capacity, location):
    with open(filename, 'w') as f:
        f.write('\n'.join([
            "{} : {}".format(k, v)
            for k, v in (
                ('NAME', name),
                ('TYPE', 'CVRP'),
                ('COMMENT', 'NONE'),
                ('DIMENSION', size + 1),
                ('EDGE_WEIGHT_TYPE', 'EUC_2D'),
                ('CAPACITY', capacity)
            )
        ]))

        f.write('\n')
        f.write('NODE_COORD_SECTION\n')

        f.write('\n'.join(['{}\t{}\t{}'.format(i + 1, x, y) for i, (x, y) in enumerate(location)]))

        f.write('\n')
        f.write('DEMAND_SECTION\n')
        f.write('\n'.join(['{}\t{}'.format(i + 1, d) for i, d in enumerate([0] + demand)]))

        f.write('\n')
        f.write('DEPOT_SECTION\n')
        f.write('1\n')
        f.write('-1\n')
        f.write('EOF\n')
