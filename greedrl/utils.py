import os
import time
import torch

act_dict = {}
act_dict['none'] = lambda x: x
act_dict['relu'] = torch.relu
act_dict['tanh'] = torch.tanh
act_dict['sigmoid'] = torch.sigmoid


def get_act(act):
    return act_dict[act]


def to_list(var):
    if isinstance(var, dict):
        return {k: to_list(v) for k, v in var.items()}
    elif isinstance(var, list):
        return [to_list(v) for v in var]
    elif isinstance(var, tuple):
        return (to_list(v) for v in var)
    elif isinstance(var, torch.Tensor):
        return var.tolist()
    else:
        return var


def repeat(tensor, size, dim=0):
    return tensor.repeat_interleave(size, dim)


def get_default_device():
    if not torch.cuda.is_available():
        return torch.device("cpu")

    cmd = 'nvidia-smi -q -d Memory | grep -A4 GPU | grep Free'
    with os.popen(cmd) as result:
        max_free_mem = 0
        max_cuda_index = -1
        for i, line in enumerate(result):
            free_mem = int(line.strip().split()[2])
            if free_mem > max_free_mem:
                max_free_mem = free_mem
                max_cuda_index = i

    return torch.device("cuda:{}".format(max_cuda_index))


def cumem_stats(device, msg):
    torch.cuda.empty_cache()
    print("{}, device:{}, memory_allocated: {:.3f}G".format(msg, device,
                                                            torch.cuda.memory_allocated(device) / (1024 * 1024 * 1024)))


cutime_stats_time = None


def cutime_stats(device, msg=''):
    global cutime_stats_time
    torch.cuda.synchronize(device)
    if cutime_stats_time is not None:
        print("{} time: {:.6f}s".format(msg, time.time() - cutime_stats_time))

    cutime_stats_time = time.time()
