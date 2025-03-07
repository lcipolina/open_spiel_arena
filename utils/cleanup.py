import atexit
import multiprocessing
import contextlib
import gc
import torch
import ray
from vllm.distributed import (
    destroy_model_parallel,
    destroy_distributed_environment
)

def full_cleanup():
    """Cleans up all resources: LLMs, GPUs, Ray, and multiprocessing."""
    print("Shutting down: Clearing all resources...")

    # Shut down Ray if it's running
    if ray.is_initialized():
        ray.shutdown()

    # Ensure vLLM's distributed model is fully shut down
    destroy_model_parallel()
    destroy_distributed_environment()

    # Ensure all multiprocessing child processes are terminated
    for child in multiprocessing.active_children():
        child.terminate()

    # Clean up PyTorch Distributed
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()

    # Run garbage collection to clear lingering references
    gc.collect()

    # Free unused GPU memory
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    print("Cleanup complete: All processes and memory released.")

# Register cleanup to run automatically at exit
atexit.register(full_cleanup)
