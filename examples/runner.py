import time
import random
import argparse
import torch

from greedrl import Problem, Solution, Solver


def run(make_problem, mask_task_ratio=0.1):
    random.seed(123)
    torch.manual_seed(123)
    problem_list = make_problem(1)

    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--device', default=None, type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--agent_file', default=None, type=str)
    parser.add_argument('--valid_steps', default=5, type=int)
    parser.add_argument('--max_steps', default=10000000, type=int)

    args, _ = parser.parse_known_args()
    for k, v in args.__dict__.items():
        print("arg: {} = {}".format(k, v))

    # rl train
    solver = Solver(device=args.device)
    solver.train(args.agent_file, problem_list, problem_list,
                 batch_size=args.batch_size, valid_steps=args.valid_steps, max_steps=args.max_steps)
    # predict
    solver = Solver(device=args.device)
    if args.agent_file is not None:
        solver.load_agent(args.agent_file)

    print("solve ...")
    start = time.time()
    for problem in problem_list:
        solver.solve(problem, batch_size=args.batch_size)
    print("time: {}s".format(time.time() - start))
