#include <pybind11/pybind11.h>
#include "task_group_split.h"
#include "task_group_priority.h"

namespace py = pybind11;

PYBIND11_MODULE(greedrl_c, m) {
    m.def("task_group_split", &task_group_split);
    m.def("task_group_priority", &task_group_priority);
}

