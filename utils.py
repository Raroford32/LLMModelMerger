import torch
import os

def get_available_memory():
    """
    Get the available GPU memory if CUDA is available, otherwise return system memory.
    """
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
    else:
        import psutil
        return psutil.virtual_memory().available

def check_disk_space(path):
    """
    Check available disk space at the given path.
    """
    total, used, free = shutil.disk_usage(path)
    return free

def ensure_directory(path):
    """
    Ensure that a directory exists, creating it if necessary.
    """
    os.makedirs(path, exist_ok=True)

def log_memory_usage():
    """
    Log current memory usage.
    """
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    else:
        import psutil
        print(f"RAM usage: {psutil.virtual_memory().percent}%")

