#include "task_group_split.h"

__global__ void task_group_split_kernel(
        int* group, bool* value, bool* output,
        const int batch_size, const int task_num, const int group_num)
{
    group += blockIdx.x * task_num;
    value += blockIdx.x * task_num;
    extern __shared__ bool temp[];
    
    __shared__ bool split;
    if(threadIdx.x == 0) split = false;

    for(int i=threadIdx.x; i<group_num; i+=blockDim.x)
    {
        temp[i] = false;
    }

    __syncthreads();

    for(int i=threadIdx.x; i<task_num; i+=blockDim.x)
    {
        int g = group[i];
        if(value[i]) temp[g] = true; 
    }
    
    __syncthreads();
    
    for(int i=threadIdx.x; i<task_num; i+=blockDim.x)
    {
        int g = group[i];
        if(temp[g] && !value[i]) split = true;
    }
    
    __syncthreads();

    if(threadIdx.x == 0) output[blockIdx.x] = split;
};

void task_group_split_cuda(
        int* group, bool* value, bool* output,
        const int batch_size, const int task_num, const int group_num, const int device)
{
    const int shared_mem = group_num * sizeof(bool);

    GRL_CHECK_CUDA(cudaSetDevice(device));

    task_group_split_kernel<<<batch_size, 256, shared_mem>>>(
        group, value, output, batch_size, task_num, group_num);

    GRL_CHECK_CUDA(cudaGetLastError());
};

