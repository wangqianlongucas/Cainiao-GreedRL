#pragma once
#include <cfloat>
#include <climits>
#include <cstdint>
#include <limits>
#include <chrono>
#include <stdexcept>
#include <torch/extension.h>

#define ASSERT(c) assert(c)
#define ALIGN(v, n) ((v + n - 1) / n * n)
#define INF std::numeric_limits<float>::infinity()
#define __FILENAME__ (__FILE__+ SOURCE_PATH_LENGTH)

#define GRL_ERROR(format, args...)                                      \
    greedrl_error(__FILENAME__, __LINE__, format, ##args);              \


#define GRL_CHECK(flag, format, args...)                                \
    greedrl_check(__FILENAME__, __LINE__, flag, format, ##args);        \


#define MALLOC(ptr, T, size)                                            \
    ptr = (T*) malloc(sizeof(T) * (size));                              \
    GRL_CHECK(ptr != nullptr, "out of memory!");                        \


#define GALLOC(ptr, T, size)                                            \
    GRL_CHECK((size) > 0, "malloc 0 bytes");                            \
    T* const ptr = (T*) malloc(sizeof(T) * (size));                     \
    GRL_CHECK(ptr != nullptr, "out of memory!");                        \
    AllocGuard ptr##_##alloc##_##guard(ptr);                            \


#define REALLOC(ptr, T, size)                                           \
    GRL_CHECK((size) > 0, "malloc 0 bytes");                            \
    ptr = (T*) realloc(ptr, sizeof(T) * (size));                        \
    GRL_CHECK(ptr != nullptr, "out of memory!");                        \


#define GRL_CHECK_TENSOR(tensor, device, allow_sub_contiguous, allow_null, ...)     \
    greedrl_check_tensor(__FILENAME__, __LINE__, tensor, #tensor, device,           \
                         allow_sub_contiguous, allow_null, {__VA_ARGS__});          \


const int GRL_WORKER_START = 0;
const int GRL_WORKER_END = 1;
const int GRL_TASK = 2;
const int GRL_FINISH = 3;

const int MAX_BATCH_SIZE = 100000; 
const int MAX_TASK_COUNT = 5120;
const int MAX_SHARED_MEM = 48128;

using String = std::string;
using Device = torch::Device;
using Tensor = torch::Tensor;
using TensorMap = std::map<String, Tensor>;
using TensorList = std::vector<Tensor>;


inline void greedrl_error(const char* const file, const int64_t line,
                          const char* const format, ...)
{
    const int N = 2048;
    static char buf[N];

    va_list args;
    va_start(args, format);
    int n = vsnprintf(buf, N, format, args);
    va_end(args);

    if(n < N)
    {
        snprintf(buf+n, N-n, " at %s:%ld", file, line);
    }

    throw std::runtime_error(buf);
}

inline void greedrl_check(const char* const file, const int64_t line,
                          const bool flag, const char* const format, ...)
{
    if(flag)
    {
        return;
    }
    
    const int N = 2048;
    static char buf[N];

    va_list args;
    va_start(args, format);
    int n = vsnprintf(buf, N, format, args);
    va_end(args);

    if(n < N)
    {
        snprintf(buf+n, N-n, " at %s:%ld", file, line);
    }

    throw std::runtime_error(buf);
}

// contiguous except the 1st dimension
inline bool is_sub_contiguous(const Tensor& tensor)
{
    int dim = tensor.dim();
    if(dim==1) return true;

    auto sizes = tensor.sizes();
    auto strides = tensor.strides();  
    
    if(strides[dim-1] != 1) return false;
 
    int s = 1;
    for(int i=dim-2; i>0; i--)
    {
        s *= sizes[i+1];
        if(strides[i] != s) return false;
    }
    
    return true;

};

inline void greedrl_check_tensor(const char* const file, 
                                 const int line,
                                 const Tensor& tensor,
                                 const String& name, 
                                 const Device& device, 
                                 bool allow_sub_contiguous,
                                 bool allow_null,
                                 std::initializer_list<int> sizes)
{
    greedrl_check(file, line, tensor.numel() < 1000 * 1000 * 1000, "tensor size too large");

    auto device2 = tensor.device();
    greedrl_check(file, line, device2==device,
            "'%s' device is %s, but expect %s",
            name.c_str(), device2.str().c_str(), device.str().c_str());
    
    bool is_contiguous = allow_sub_contiguous ? is_sub_contiguous(tensor) : tensor.is_contiguous();
    greedrl_check(file, line, is_contiguous, "'%s' is not contiguous", name.c_str());
    
    if(allow_null && tensor.data_ptr() == nullptr) return;
    
    if(tensor.dim() != sizes.size())
    {
        greedrl_error(file, line, "'%s' dim is %d, but expect %d", name.c_str(), (int)tensor.dim(), (int)sizes.size());
    }
    int i=0;
    for(auto s:sizes)
    {
        greedrl_check(file, line, tensor.size(i)==s, "'%s' size(%d) is %d, but expect %d", name.c_str(), i, (int)tensor.size(i), s);
        i++;
    }
}


#ifdef CUDA_FOUND

#include <cuda_runtime_api.h>

#define GRL_CHECK_CUDA(error)\
    greedrl_check_cuda(error, __FILENAME__, __LINE__);                                                  

inline void greedrl_check_cuda(const cudaError_t& error,
                               const char* file, const int64_t line)
{
    if(error==cudaSuccess)
    {
        return;
    }
    
    const int N = 2048;
    static char buf[N];
    snprintf(buf, N, "%s, at %s:%ld", cudaGetErrorString(error), file, line);
    throw std::runtime_error(buf);
}

cudaDeviceProp& cuda_get_device_prop(int i);

#endif
