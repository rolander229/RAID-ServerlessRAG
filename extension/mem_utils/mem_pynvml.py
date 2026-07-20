import ctypes
import pynvml

from contextlib import contextmanager
import pynvml

@contextmanager
def _nvml():
    try:
        pynvml.nvmlInit()
        yield
    finally:
        pynvml.nvmlShutdown()

@_nvml()
def report_used_memory(index=0, msg=""):
    # Get the handle for the same device using NVML
    handle = pynvml.nvmlDeviceGetHandleByIndex(index)
    # Get memory information
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"{msg}: Used memory: {info.used / 1024 ** 2} MB")
    return info.used

@_nvml()
def report_used_mem(index=0):
    # Get the handle for the same device using NVML
    handle = pynvml.nvmlDeviceGetHandleByIndex(index)
    # Get memory information
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info.used

# def estimate_cuda_context_memory():
#     # Load the CUDA driver library
#     cuda = ctypes.CDLL('libcuda.so')

#     report_used_memory(msg="before cuinit")

#     # Initialize the CUDA Driver API
#     result = cuda.cuInit(0)
#     if result != 0:
#         raise Exception("Failed to initialize the CUDA driver API")

#     report_used_memory(msg="after cuinit")

#     device = 0
#     # Create contexts on device 0
#     contexts = []
#     mem_stats = []
#     for i in range(3):
#         context = ctypes.c_void_p()
#         result = cuda.cuCtxCreate(ctypes.byref(context), 0, device)
#         if result != 0:
#             raise Exception("Failed to create one or both CUDA contexts")
#         mem_used = report_used_memory(msg=f"after creating {i + 1} cuda context")
#         mem_stats.append(mem_used)
#         contexts.append(context)
#     first_context_mem_used = mem_stats[0]
#     added_context_mem_used = ((mem_stats[2] - mem_stats[1]) + (mem_stats[1] - mem_stats[0])) / 2
#     # to MB
#     print(f"Estimated CUDA context memory usage: {first_context_mem_used / 1024 ** 2:.2f} MB")
#     print(f"Estimated additional memory usage per context: {added_context_mem_used / 1024 ** 2:.2f} MB")
#     return first_context_mem_used

def estimate_cuda_context_memory():
    # Load the CUDA driver library
    cuda = ctypes.CDLL('libcuda.so')

    # report_used_memory(msg="before cuinit")

    # Initialize the CUDA Driver API
    result = cuda.cuInit(0)
    if result != 0:
        raise Exception("Failed to initialize the CUDA driver API")

    # report_used_memory(msg="after cuinit")

    device = 0
    # Create contexts on device 0

    context = ctypes.c_void_p()
    result = cuda.cuCtxCreate(ctypes.byref(context), 0, device)
    if result != 0:
        raise Exception("Failed to create one or both CUDA contexts")
    mem_used = report_used_mem()
    # to GB
    return mem_used / (1024 ** 3)

# cuda_default_context_memory = estimate_cuda_context_memory()
# print(cuda_default_context_memory)