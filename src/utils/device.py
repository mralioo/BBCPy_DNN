import os
import subprocess
import multiprocessing

import numpy as np
import psutil
import torch
import GPUtil


def get_nvidia_gpus():
    try:
        result = subprocess.check_output(["nvidia-smi", "--query-gpu=gpu_name,memory.total", "--format=csv,noheader,nounits"]).decode()
        return result.strip().split("\n")
    except:
        return []

def get_cpu_cores():
    return multiprocessing.cpu_count()

def get_system_memory():
    virtual_memory = psutil.virtual_memory()
    return virtual_memory.total / (1024 ** 3)  # Convert bytes to GB


def print_gpu_info():
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return

    get_nvidia_gpus()

    # Print PyTorch and CUDA version
    print("PyTorch Version:", torch.__version__)
    print("CUDA Version:", torch.version.cuda)

    # Print available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # Print CPU info
    print(f"Number of CPU cores: {get_cpu_cores()}")
    print(f"System memory: {get_system_memory():.2f} GB")


def print_memory_usage():
    # Virtual (total) memory
    total_memory = psutil.virtual_memory().total / (1024 ** 3)  # Convert to GB
    available_memory = psutil.virtual_memory().available / (1024 ** 3)  # Convert to GB
    used_memory = total_memory - available_memory

    print(f"Total Memory: {total_memory:.2f} GB")
    print(f"Available Memory: {available_memory:.2f} GB")
    print(f"Used Memory: {used_memory:.2f} GB")


def print_cpu_cores():
    total_cores = psutil.cpu_count(logical=True)  # This counts hyper-threaded cores as well
    physical_cores = psutil.cpu_count(logical=False)

    # Get the current process info
    process = psutil.Process(os.getpid())
    used_cores = len(process.cpu_affinity())

    print(f"Total CPU Cores (Logical/Hyper-threaded): {total_cores}")
    print(f"Total CPU Cores (Physical): {physical_cores}")
    print(f"Number of Cores Used by Current Process: {used_cores}")

def print_gpu_memory():
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        print(f"GPU {gpu.id} - {gpu.name}")
        print(f"  Total Memory: {gpu.memoryTotal} MB")
        print(f"  Free Memory: {gpu.memoryFree} MB")
        print(f"  Used Memory: {gpu.memoryUsed} MB")
        print("")


def print_data_info(srm_obj):

    print("data shape: ", srm_obj.shape)
    print("max timepoints: ", np.max(srm_obj))
    print("min timepoints: ", np.min(srm_obj))
    print("mean timepoints: ", np.mean(srm_obj))
    print("std timepoints: ", np.std(srm_obj))
