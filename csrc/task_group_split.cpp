#include "task_group_split.h"

void task_group_split_cpu(
        int* group, bool* value, bool* output,
        const int batch_size, const int task_num, const int group_num)
{
    auto temp = torch::make_unique<bool[]>(group_num);
    for(int b=0; b<batch_size; b++)
    {
        for(int i=0; i<group_num; i++){
            temp[i] = false;
        }
        
        for(int i=0; i<task_num; i++){
            if(value[i]){
                int g = group[i];
                temp[g] = true;
            }
        }
        
        output[b] = false;
        for(int i=0; i<task_num; i++){
            int g = group[i];
            if(temp[g] && !value[i]){
                output[b] = true;
                break;
            }
        }
        
        group += task_num;
        value += task_num;
    }
};


auto task_group_split(
    const Tensor& group, const Tensor& value) -> Tensor
{
    auto device = group.device();
    const int batch_size = group.size(0);
    const int task_num = group.size(1);
    const int group_num = group.max().item<int>() + 1;
    const int _group_num = group.min().item<int>();

    GRL_CHECK(group_num <= task_num && _group_num >= 0, "group value error");

    GRL_CHECK_TENSOR(group, device, false, false, batch_size, task_num);
    GRL_CHECK_TENSOR(value, device, false, false, batch_size, task_num);

    auto output = torch::zeros({batch_size}, torch::dtype(torch::kBool).device(device));

    switch(device.type())
    {
        case torch::kCPU:
            task_group_split_cpu(group.data_ptr<int>(), value.data_ptr<bool>(),
                                 output.data_ptr<bool>(), batch_size, task_num, group_num); 
            break;
#ifdef CUDA_FOUND
        case torch::kCUDA:
            task_group_split_cuda(group.data_ptr<int>(), value.data_ptr<bool>(),
                                  output.data_ptr<bool>(), batch_size, task_num, group_num, device.index());
            break;
#endif
        default:
            GRL_ERROR("unsupported device: %s", device.str().c_str());
    }
    
    return output;
};
