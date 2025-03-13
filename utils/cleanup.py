import atexit
import multiprocessing
import contextlib
import gc
import torch
import ray
import os
import glob
from torch.utils.tensorboard import SummaryWriter
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


# Tensorboard files cleanup
import shutil

# Cleanup function
def cleanup_tensorboard_logs(log_dir: str = "runs/kuhn_poker", keep_last: int = 5):
    """
    Deletes older TensorBoard logs to save disk space.

    Args:
        log_dir (str): The directory where TensorBoard logs are stored.
        keep_last (int): Number of most recent logs to keep.
    """
    if not os.path.exists(log_dir):
        return  # No logs to delete

    files = sorted(glob.glob(os.path.join(log_dir, "events.out.tfevents.*")), key=os.path.getmtime)

    # Delete all but the most recent 'keep_last' logs
    for file in files[:-keep_last]:
        try:
            os.remove(file)
            print(f"Deleted old TensorBoard log: {file}")
        except Exception as e:
            print(f"Failed to delete {file}: {e}")

# Call cleanup before initializing TensorBoard logging
#cleanup_tensorboard_logs(log_dir: str = "runs/kuhn_poker", keep_last: int = 5)

