import sys
import time
import torch
import argparse
import utils
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2


def solve(problem, i, max_time):
    scale = 100000
    size = problem.task_demand.size(1)
    demand = [0] + problem.task_demand[i].tolist()
    capacity = problem.worker_weight_limit[i].tolist()
    distance = (problem.distance_matrix[i] * scale + 0.5).to(torch.int32).tolist()

    queue = mp.Queue()
    p = mp.Process(target=do_solve, args=(size, demand, capacity, distance, max_time, queue))
    p.start()
    p.join()

    return queue.get() / scale, queue.get()


def do_solve(size, demand, capacity, distance, max_time, queue):
    capacity = capacity * size

    manager = pywrapcp.RoutingIndexManager(size + 1, size, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance[from_node][to_node]

    distance_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(distance_callback_index)

    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return demand[from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(demand_callback_index, 0, capacity, True, 'capacity')

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    params.local_search_metaheuristic = (routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    params.time_limit.seconds = max_time

    start_time = time.time()
    solution = routing.SolveWithParameters(params)
    spent_time = time.time() - start_time

    queue.put(solution.ObjectiveValue())
    queue.put(spent_time)


def run_orts(task, max_time):
    problem, i = task
    return solve(problem, i, max_time)


def main(args):
    print("args: {}".format(vars(args)))
    problem_size = args.problem_size
    problem_count = args.problem_count
    batch_size = args.batch_size

    assert problem_count % batch_size == 0
    batch_count = problem_count // batch_size
    problem_list = utils.make_problem(batch_count, batch_size, problem_size)

    executor = ThreadPoolExecutor(max_workers=args.threads)
    task_list = [(p, i) for p in problem_list for i in range(batch_size)]

    total_cost = 0
    total_time = 0
    for cost, elapse in executor.map(run_orts, task_list, [args.max_time] * problem_count):
        total_cost += cost
        total_time += elapse

    avg_cost = total_cost / problem_count
    avg_time = total_time / problem_count
    print()
    print("-----------------------------------------------------")
    print("avg_cost: {:.4f}".format(avg_cost))
    print("avg_time: {:.6f}s".format(avg_time))
    print("total_count: {}".format(problem_count))
    print("-----------------------------------------------------\n")
    sys.stdout.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--threads', default=20, type=int, help='number of threads')
    parser.add_argument('--max_time', default=60, type=int, help='the time limit for the search in seconds')

    parser.add_argument('--problem_size', default=100, type=int, choices=[100, 1000, 2000, 5000],  help='problem size')
    parser.add_argument('--problem_count', default=128, type=int,  help='total number of generated problem instances')
    parser.add_argument('--batch_size', default=128, type=int,  help='batch size for feedforwarding')

    args = parser.parse_args()
    main(args)
