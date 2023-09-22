import os
import subprocess
import multiprocessing
import psutil
import torch


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
