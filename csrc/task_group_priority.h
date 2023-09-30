#pragma once

#include "./common.h"

/**
 * tasks are divided into groups,
 * tasks in a group are visited by it's priority.
 * the min priority value of unvisited tasks in a group is computed,
 * output is false, if the task's priority equal the computed min priority, otherwise output is true
 *
 * group: task's group, shape is (batch_size, task_num)
 * priority: task's priority, shape is (batch_size, task_num)
 * value: task is visited or not, shape is (batch_size, task_num)
 *
 * output: the result, shape is (batch_size, task_num)
 */
auto task_group_priority(
        const torch::Tensor& group,
        const torch::Tensor& priority,
        const torch::Tensor& value) -> torch::Tensor;

void task_group_priority_cpu(
        int* group, int* priority, bool* value, bool* ouput,
        int batch_size, int task_num, int group_num);

void task_group_priority_cuda(
        int* group, int* priority, bool* value, bool* ouput,
        int batch_size, int task_num, int group_num, int device);


