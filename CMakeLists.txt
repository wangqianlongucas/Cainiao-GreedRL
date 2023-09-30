cmake_minimum_required(VERSION 2.8.12)
project(greedrl_C_ LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)

find_package(PythonInterp REQUIRED)
execute_process(COMMAND "python" "-c" 
"
import os
import torch
from distutils import sysconfig as s
print(s.get_python_inc(plat_specific=True))
print(s.get_config_var('EXT_SUFFIX'))
print(os.path.dirname(torch.__file__))
"
RESULT_VARIABLE _PYTHON_SUCCESS
OUTPUT_VARIABLE _PYTHON_VALUES
ERROR_VARIABLE _PYTHON_ERROR_VALUE)

if(NOT _PYTHON_SUCCESS MATCHES 0)
    message("_PYTHON_SUCCESS: ${_PYTHON_SUCCESS}")
    message("_PYTHON_VALUES: ${_PYTHON_VALUES}")
    message("_PYTHON_ERROR_VALUE: ${_PYTHON_ERROR_VALUE}")
    message(FATAL_ERROR "get python config error!")
endif()

string(REGEX REPLACE "\n" ";" _PYTHON_VALUES ${_PYTHON_VALUES})
list(GET _PYTHON_VALUES 0 PYTHON_INCLUDE_DIR)
list(GET _PYTHON_VALUES 1 PYTHON_EXT_SUFFIX)
list(GET _PYTHON_VALUES 2 TORCH_HOME)

include_directories(
    ${PYTHON_INCLUDE_DIR}
    ${TORCH_HOME}/include
    ${TORCH_HOME}/include/TH
    ${TORCH_HOME}/include/THC
    ${TORCH_HOME}/include/torch/csrc/api/include
)

string(LENGTH "${CMAKE_SOURCE_DIR}/" SOURCE_PATH_LENGTH)
add_compile_options(-DSOURCE_PATH_LENGTH=${SOURCE_PATH_LENGTH})
add_compile_options(-D_GLIBCXX_USE_CXX11_ABI=0 -fvisibility=hidden -fopenmp)

if(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    add_link_options(-undefined dynamic_lookup)
endif()

file(GLOB_RECURSE CSRC_CPP csrc/*.cpp)

add_library(greedrl_c MODULE ${CSRC_CPP})
set_target_properties(greedrl_c PROPERTIES PREFIX "")
set_target_properties(greedrl_c PROPERTIES SUFFIX "${PYTHON_EXT_SUFFIX}")
target_compile_options(greedrl_c PRIVATE -Wno-sign-conversion -O3)
target_link_libraries(greedrl_c c10 torch torch_cpu torch_python)
target_link_directories(greedrl_c PRIVATE ${TORCH_HOME}/lib)

find_package(CUDA)
if(CUDA_FOUND)
    enable_language(CUDA)
    file(GLOB_RECURSE CSRC_CU csrc/*.cu)
    add_library(greedrl_cu OBJECT ${CSRC_CU})
    target_compile_options(greedrl_cu PRIVATE -keep -Xptxas -v --expt-relaxed-constexpr --expt-extended-lambda -O3)
    set_target_properties(greedrl_cu PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_ARCHITECTURES "70;75;80")
    add_compile_definitions(CUDA_FOUND)
    include_directories(${CUDA_INCLUDE_DIRS})
    target_link_libraries(greedrl_c torch_cuda greedrl_cu)
    target_link_directories(greedrl_c PRIVATE ${TORCH_HOME}/lib)
endif()
