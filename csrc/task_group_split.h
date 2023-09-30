#pragma once

#include "./common.h"

/**
 * tasks are divided into groups, 
 * if tasks in a group are all visited or all not visited,
 * output is is false, otherwise output is true
 *
 * group: task's group, shape is (batch_size, task_num)
 * value: task is visited or not, shape is (batch_size, task_num)
 *
 * output: the result, shape is (batch_size,)
 */
auto task_group_split(const Tensor& group, const Tensor& value) -> Tensor;

void task_group_split_cpu(
        int* group, bool* value, bool* output,
        const int batch_size, const int task_num, const int group_num);

void task_group_split_cuda(
        int* group, bool* value, bool* output,
        const int batch_size, const int task_num, const int group_num, const int device);

