"""
llm_registry.py - Central llm registry
Handles LLM loading, dynamic GPU allocation
"""

import os
import json
import torch
import gc
import litellm
from typing import Dict, Any
from vllm import LLM as vLLM
from llm_utils import cleanup_vllm


#TODO: this should go into the SLURM file

# Enable quantized weights (reduce model size)
os.environ["VLLM_OPENVINO_ENABLE_QUANTIZED_WEIGHTS"] = "ON"

# Set precision for KV cache (optional, can try "u8" or "fp16")
os.environ["VLLM_OPENVINO_CPU_KV_CACHE_PRECISION"] = "u8"


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# TODO: eventually delete this as it will be on SLURM
#import subprocess

# Load required modules
#subprocess.run("module load CUDA", shell=True, check=True)
#subprocess.run("module load Stages/2025 Python/3.12.3", shell=True, check=True)


# Debugging Paths - Set these via SLURM later  #TODO:later delete this!
MODELS_DIR="/p/data1/mmlaion/marianna/models"
MODEL_CONFIG_FILE="/p/project/ccstdl/cipolina-kun1/open_spiel_arena/configs/models.json"
LITELLM_CONFIG_FILE = "/p/project/ccstdl/cipolina-kun1/open_spiel_arena/configs/litellm.json"

# Retrieve paths from environment variables
#MODELS_DIR = os.getenv("MODELS_DIR")   #Commented for debugging! TODO:delete this!
#MODEL_CONFIG_FILE = os.getenv("MODEL_CONFIG_FILE")
if MODELS_DIR is None or MODEL_CONFIG_FILE is None:
    raise ValueError("Error: Environment variables MODELS_DIR or MODEL_CONFIG_FILE are not set.")


# Load LiteLLM Model List from JSON
def load_litellm_model_list() -> list:
    """Loads the list of models served by LiteLLM from a JSON file."""
    with open(LITELLM_CONFIG_FILE, "r") as f:
        data = json.load(f)
    return data["models"]

# Check if a Model is a LiteLLM API Model
LITELLM_MODELS = set(load_litellm_model_list())

def is_litellm_model(model_name: str) -> bool:
    """Checks if a model is served via LiteLLM API."""
    return model_name in LITELLM_MODELS


# Load Local Model List from JSON
def load_model_list() -> list:
    """
    Loads the list of vLLM models from a JSON file.

    Returns:
        List[str]: List of model names.
    """
    with open(MODEL_CONFIG_FILE, "r") as f: # type: ignore
        data = json.load(f)
    return data["models"]

# Master Function: Detects Model Type & Calls Correct Loader
CURRENT_LLM = None
def load_llm_model(model_name: str):
    """
    Detects whether the model should be loaded using vLLM (local) or LiteLLM (API).
    Cleans up old vLLM models only if needed.

    Args:
        model_name (str): Name of the model to load.

    Returns:
        vLLM instance (if local) or LiteLLM API response handler (if API-based).
    """

    # Skip unloading if the model is already loaded
    if CURRENT_LLM is not None and CURRENT_LLM.model == model_name:
        print(f"{model_name} is already loaded, skipping reload.")
        return CURRENT_LLM

    # Check if we have enough free GPU memory for another model
    free_memory = get_free_gpu_memory()
    required_memory = estimate_model_memory(model_name)

    if free_memory < required_memory:
        print(f"⚠️ Not enough memory! Unloading {CURRENT_LLM.model} to load {model_name}...")
        cleanup_vllm(CURRENT_LLM)
        CURRENT_LLM = None

    # Load model using the appropriate method
    if is_litellm_model(model_name):
        CURRENT_LLM = load_litellm_model(model_name)
    else:
        CURRENT_LLM = load_vllm_model(model_name)

    return CURRENT_LLM

# Loads API-Based Models via LiteLLM
def load_litellm_model(model_name: str):
    """
    Loads an API-based model (e.g., GPT-4, Claude, Mistral) using LiteLLM.

    Args:
        model_name (str): Name of the model to load.

    Returns:
        A LiteLLM API response handler.
    """
    print(f"Using LiteLLM for API-based model: {model_name}")

    return litellm.Completion(model=model_name)


# Detect Quantization Type for vLLM Models
def detect_quantization(model_path: str) -> str:
    """Detects if a model is quantized (4-bit, 8-bit) based on saved files."""
    if not os.path.exists(model_path):
        return "fp16"  # Default assumption if we can't check files

    files = os.listdir(model_path)
    if any("4bit" in f or "awq" in f or "ggml" in files):
        return "4bit"
    elif any("8bit" in f for f in files):
        return "8bit"
    else:
        return "fp16"

# Assigns GPU Based on Model Size & Quantization
def auto_assign_gpu(model_name: str, model_path: str) -> vLLM:
    """Dynamically assigns GPU & tensor parallelism based on model size, available memory, and quantization.

    Args:
        model_name (str): The name of the model to load.
        model_path (str): The file path where the model is stored.

    Returns:
        vLLM: A vLLM model instance configured with optimized GPU allocation.

    Notes:
        - Uses tensor parallelism for large models (`>13B` models use `tp=2`, `>70B` use `tp=4`).
        - Disables `custom_all_reduce` when `tensor_parallel_size > 1`.
        - Ensures GPU memory utilization does not exceed 90% (`gpu_memory_utilization=0.9`).
    """

    num_params = detect_model_size(model_name)  # Get the model size
    quantization = detect_quantization(model_path)

    # Adjust Memory Calculation Based on Quantization
    if quantization == "4bit":
        bytes_per_param = 0.5  # 4-bit quantization → 0.5 bytes per param
    elif quantization == "8bit":
        bytes_per_param = 1  # 8-bit quantization → 1 byte per param
    else:  # Default: FP16/BF16
        bytes_per_param = 2

    # Compute Required GPU Memory (GB)
    required_memory = (num_params * bytes_per_param)

    # Get Available GPUs
    num_gpus = torch.cuda.device_count()
    free_gpus = []

    # Check Available Memory Per GPU
    for gpu_id in range(num_gpus):
        free_mem = torch.cuda.mem_get_info(gpu_id)[0] / 1e9  # Convert to GB
        if free_mem > required_memory * 1.2:  # 20% safety margin
            free_gpus.append(gpu_id)

    # Assign Tensor Parallelism Based on Model Size
    if num_params > 70:
        tensor_parallel_size = min(4, num_gpus)  # Use up to 4 GPUs
    elif num_params > 13:
        tensor_parallel_size = min(2, num_gpus)  # Use up to 2 GPUs
    else:
        tensor_parallel_size = 1  # Default to 1 GPU

    # Assign GPUs
    assigned_gpus = free_gpus[:tensor_parallel_size]
    assigned_gpus_str = ", ".join([f"cuda:{gpu}" for gpu in assigned_gpus])

    print(f"🖥️ Loading {model_name} on GPUs [{assigned_gpus_str}] with TP={tensor_parallel_size}")

    return vLLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        device=f"cuda:{assigned_gpus[0]}",  # Gets the first GPU from the assigned list as the primary GPU
        gpu_memory_utilization=0.9,
        disable_custom_all_reduce=(tensor_parallel_size > 1)
    )

    """Dynamically assigns GPU & tensor parallelism based on model size & quantization."""

    # Detect Model Size
    if "7b" in model_name.lower():
        num_params = 7
    elif "13b" in model_name.lower():
        num_params = 13
    elif "33b" in model_name.lower():
        num_params = 33
    elif "70b" in model_name.lower() or "72b" in model_name.lower():
        num_params = 70
    else:
        num_params = 7  # Default assumption

    # Detect Quantization Type
    quantization = detect_quantization(model_path)

    # Adjust Memory Calculation Based on Quantization
    if quantization == "4bit":
        bytes_per_param = 0.5  # 4-bit quantization → 0.5 bytes per param
    elif quantization == "8bit":
        bytes_per_param = 1  # 8-bit quantization → 1 byte per param
    else:  # Default: FP16/BF16
        bytes_per_param = 2

    # Compute Required GPU Memory (GB)
    required_memory = (num_params * bytes_per_param)

    # Get Available GPUs
    num_gpus = torch.cuda.device_count()
    free_gpus = []

    # Check Available Memory Per GPU
    for gpu_id in range(num_gpus):
        free_mem = torch.cuda.mem_get_info(gpu_id)[0] / 1e9  # Convert to GB
        if free_mem > required_memory * 1.2:  # 20% safety margin
            free_gpus.append(gpu_id)

    # Assign GPU and Tensor Parallelism
    if free_gpus:
        assigned_gpu = free_gpus[0]
        tensor_parallel_size = 1
    else:
        assigned_gpu = 0  # Default to first GPU
        tensor_parallel_size = 2 if num_params > 10 else 1  # Use TP if large model

    print(f"🖥️ Loading {model_name} on GPU {assigned_gpu} with TP={tensor_parallel_size} (Quantization: {quantization})")

    # Disable `custom_all_reduce` when `tp > 1`
    return vLLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        device=f"cuda:{assigned_gpu}",
        disable_custom_all_reduce=(tensor_parallel_size > 1)  # Disable custom reduce when TP > 1
    )

# Loads vLLM Models with Dynamic GPU Allocation
def load_vllm_model(model_name: str) -> vLLM:
    """
    Loads a model using vLLM from the pre-downloaded model directory
    and assigns it dynamically to a GPU.

    Args:
        model_name (str): The name of the model to load.

    Returns:
        vLLM: An instance of the vLLM model.
    """
    model_path = f"{MODELS_DIR}/{model_name}"
    return auto_assign_gpu(model_name, model_path)

def detect_model_size(model_name: str) -> int:
    """Detects the approximate model size in billions of parameters from the name."""

    model_name = model_name.lower()

    if "7b" in model_name:
        return 7
    elif "13b" in model_name:
        return 13
    elif "33b" in model_name:
        return 33
    elif "70b" in model_name or "72b" in model_name:
        return 70
    else:
        print(f"⚠️ Warning: Could not detect size for {model_name}. Assuming 7B.")
        return 7  # Default to 7B if unknown

import torch

def get_free_gpu_memory() -> float:
    """Returns the total free GPU memory across all available GPUs (in GB)."""

    total_free_memory = 0.0
    num_gpus = torch.cuda.device_count()

    for gpu_id in range(num_gpus):
        free_mem = torch.cuda.mem_get_info(gpu_id)[0] / 1e9  # Convert to GB
        total_free_memory += free_mem

    print(f"Available GPU memory: {total_free_memory:.2f} GB")
    return total_free_memory

def estimate_model_memory(model_name: str) -> float:
    """Estimates the GPU memory required for a model (in GB)."""

    num_params = detect_model_size(model_name)  # Get the model size
    quantization = detect_quantization(f"{MODELS_DIR}/{model_name}")

    if quantization == "4bit":
        bytes_per_param = 0.5  # 4-bit quantization → 0.5 bytes per param
    elif quantization == "8bit":
        bytes_per_param = 1  # 8-bit quantization → 1 byte per param
    else:  # Default: FP16/BF16
        bytes_per_param = 2

    estimated_memory = num_params * bytes_per_param
    print(f"Estimated GPU memory for {model_name}: {estimated_memory:.2f} GB (Quantization: {quantization})")

    return estimated_memory





# Dynamically Register All Models
LLM_REGISTRY: Dict[str, Dict[str, Any]] = {}
MODEL_LIST = load_model_list() + load_litellm_model_list()  # Combine Local & LiteLLM Models

for model in MODEL_LIST:
    LLM_REGISTRY[model] = {
        "display_name": model,
        "description": f"LLM model {model} (vLLM or LiteLLM detected automatically).",
        "model_loader": lambda model_name=model: load_llm_model(model_name),
    }
