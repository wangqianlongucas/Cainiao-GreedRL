import os
import sys
import time
import torch
import argparse
import utils
from greedrl import Solver

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def do_solve(args):
    print("args: {}".format(vars(args)))

    problem_size = args.problem_size
    problem_count = args.problem_count
    batch_size = args.batch_size
    assert problem_count % batch_size == 0
    batch_count = problem_count // batch_size

    problem_list = utils.make_problem(batch_count, batch_size, problem_size)

    solver = Solver(device=args.device)

    model_path = os.path.join('./', args.model_name)
    solver.load_agent(model_path)

    total_cost = 0

    if solver.device.type == 'cuda':
        torch.cuda.synchronize()

    start_time = time.time()
    for problem in problem_list:
        solution = solver.solve(problem, greedy=False, batch_size=batch_size)
        total_cost += solution.cost.sum().item()

    if solver.device.type == 'cuda':
        torch.cuda.synchronize()

    total_time = time.time() - start_time

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
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'], help="choose a device")
    parser.add_argument('--model_name', default='cvrp_100.pt', choices=['cvrp_100.pt', 'cvrp_1000.pt', 'cvrp_2000.pt', 'cvrp_5000.pt'], help="choose a model")
    parser.add_argument('--problem_size', default=100, type=int, choices=[100, 1000, 2000, 5000],  help='problem size')
    parser.add_argument('--problem_count', default=128, type=int,  help='total number of generated problem instances')
    parser.add_argument('--batch_size', default=128, type=int,  help='batch size for feedforwarding')

    args = parser.parse_args()
    do_solve(args)

