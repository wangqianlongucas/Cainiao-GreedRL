#include "task_group_priority.h"

__global__ void task_group_priority_kernel(
        int* group, int* priority, bool* value, bool* output,
        int batch_size, int task_num, int group_num)
{
    group += blockIdx.x * task_num;
    priority += blockIdx.x * task_num;
    value += blockIdx.x * task_num;
    output += blockIdx.x * task_num;

    extern __shared__ int temp[];
    
    for(int i=threadIdx.x; i<group_num; i+=blockDim.x)
    {
        temp[i] = std::numeric_limits<int>::max();
    }
    
    __syncthreads();

    for(int i=threadIdx.x; i<task_num; i+=blockDim.x){
        if(value[i]){
            continue;
        }
        int g = group[i];
        int p = priority[i];
        atomicMin(&temp[g], p);
    }

    __syncthreads();

    for(int i=threadIdx.x; i<task_num; i+=blockDim.x){
        int g = group[i];
        output[i] = priority[i]!=temp[g];
    }
};

template<typename _Tg, typename _Tp>
__global__ void cuda_do_task_group_priority(
        const torch::PackedTensorAccessor<_Tg,2,torch::RestrictPtrTraits> group,
        const torch::PackedTensorAccessor<_Tp,2,torch::RestrictPtrTraits> priority,
        const torch::PackedTensorAccessor<bool,2,torch::RestrictPtrTraits> value,
        torch::PackedTensorAccessor<bool,2,torch::RestrictPtrTraits> result,
        const _Tg NG)
{
    const int NP = group.size(0);
    const int NT = group.size(1);
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if(p < NP)
    {
        extern __shared__ char _temp[];
        auto temp = reinterpret_cast<_Tp*>(_temp);
        temp += (threadIdx.x * NG);
        for(_Tg g=0; g<NG; g++){
            temp[g] = std::numeric_limits<_Tp>::max();
        }

        for(int t=0; t<NT; t++){
            if(value[p][t]){
                continue;
            }
            _Tg g = group[p][t];
            _Tp _p = priority[p][t];
            if(_p < temp[g]){
                temp[g] = _p;
            }
        }

        for(int t=0; t<NT; t++){
            _Tg g = group[p][t];
            if(priority[p][t]==temp[g]){
                result[p][t] = false;
            }
        }
    }
};



void task_group_priority_cuda(
        int* group, int* priority, bool* value, bool* output,
        const int batch_size, const int task_num, const int group_num, const int device)
{
    const int shared_mem = group_num * sizeof(int);

    GRL_CHECK_CUDA(cudaSetDevice(device));

    task_group_priority_kernel<<<batch_size, 256, shared_mem>>>(
        group, priority, value, output, batch_size, task_num, group_num);

    GRL_CHECK_CUDA(cudaGetLastError());
};

