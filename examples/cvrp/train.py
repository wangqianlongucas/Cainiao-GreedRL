import sys
import math
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
import utils
from greedrl import Solver


def do_train(args, rank):
    world_size = args.world_size
    model_filename = args.model_filename
    problem_size = args.problem_size
    batch_size = args.batch_size

    index = model_filename.rfind('.')
    if world_size > 1:
        stdout_filename = '{}_r{}.log'.format(model_filename[0:index], rank)
    else:
        stdout_filename = '{}.log'.format(model_filename[0:index])

    stdout = open(stdout_filename, 'a')
    sys.stdout = stdout
    sys.stderr = stdout

    print("args: {}".format(vars(args)))
    if world_size > 1:
        dist.init_process_group('NCCL', init_method='tcp://127.0.0.1:29500',
                                rank=rank, world_size=world_size)

    problem_batch_size = 8
    batch_count = 0
    if problem_size == 100:
        batch_count = math.ceil(10000 / problem_batch_size)
    elif problem_size == 1000:
        batch_count = math.ceil(200 / problem_batch_size)
    elif problem_size == 2000:
        batch_count = math.ceil(100 / problem_batch_size)
    elif problem_size == 5000:
        batch_count = math.ceil(10 / problem_batch_size)
    else:
        raise Exception("unsupported problem size: {}".format(problem_size))

    nn_args = {
        'encode_norm': 'instance',
        'encode_layers': 6,
        'decode_rnn': 'LSTM'
    }

    device = None if world_size == 1 else 'cuda:{}'.format(rank)
    solver = Solver(device, nn_args)

    train_dataset = utils.Dataset(None, problem_batch_size, problem_size)
    valid_dataset = utils.Dataset(batch_count, problem_batch_size, problem_size)

    solver.train(model_filename, train_dataset, valid_dataset,
                 train_dataset_workers=5,
                 batch_size=batch_size,
                 memopt=10,
                 topk_size=1,
                 init_lr=1e-4,
                 valid_steps=500,
                 warmup_steps=0)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--model_filename', type=str, help='model file name')
    parser.add_argument('--problem_size', default=100, type=int, choices=[100, 1000, 2000, 5000],  help='problem size')
    parser.add_argument('--batch_size', default=128, type=int,  help='batch size for training')

    args = parser.parse_args()

    processes = []
    for rank in range(args.world_size):
        p = mp.Process(target=do_train, args=(args, rank))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
