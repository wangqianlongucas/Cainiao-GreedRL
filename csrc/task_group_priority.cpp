#include "task_group_priority.h"

void task_group_priority_cpu(
        int* group, int* priority, bool* value, bool* output,
        int batch_size, int task_num, int group_num)
{
    auto temp = torch::make_unique<int[]>(group_num);
    for(int b=0; b<batch_size; b++)
    {
        for(int i=0; i<group_num; i++){
            temp[i] = std::numeric_limits<int>::max();
        }
        
        for(int i=0; i<task_num; i++){
            if(value[i]){
                continue;
            }
            int g = group[i];
            int p = priority[i];
            if(p < temp[g]){
                temp[g] = p;
            }
        }
    
        for(int i=0; i<task_num; i++){
            int g = group[i];
            output[i] = priority[i]!=temp[g];
        }
        
        group += task_num;
        priority += task_num;
        value += task_num;
        output += task_num;
    }
};

auto task_group_priority(
        const torch::Tensor& group,
        const torch::Tensor& priority,
        const torch::Tensor& value) -> torch::Tensor 
{
    auto device = group.device();

    const int batch_size = group.size(0);
    const int task_num = group.size(1);
    const int group_num = group.max().item<int>() + 1;

    const int _group_num = group.min().item<int>();

    GRL_CHECK(group_num <= task_num && _group_num >= 0, "group value error");

    GRL_CHECK_TENSOR(group, device, false, false, batch_size, task_num);
    GRL_CHECK_TENSOR(priority, device, false, false, batch_size, task_num);
    GRL_CHECK_TENSOR(value, device, false, false, batch_size, task_num);

    auto output = torch::zeros({batch_size, task_num}, torch::dtype(torch::kBool).device(device));

    switch(device.type())
    {
        case torch::kCPU:
            task_group_priority_cpu(group.data_ptr<int>(), priority.data_ptr<int>(), value.data_ptr<bool>(),
                                    output.data_ptr<bool>(), batch_size, task_num, group_num); 
            break;
#ifdef CUDA_FOUND
        case torch::kCUDA:
            task_group_priority_cuda(group.data_ptr<int>(), priority.data_ptr<int>(), value.data_ptr<bool>(),
                                    output.data_ptr<bool>(), batch_size, task_num, group_num, device.index());
            break;
#endif
        default:
            GRL_ERROR("unsupported device: %s", device.str().c_str());
    }
    
    return output;
};
