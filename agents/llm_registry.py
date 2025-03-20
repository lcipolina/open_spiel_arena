"""
llm_registry.py - Central llm registry
Handles LLM loading, dynamic GPU allocation
"""

import os
import json
import torch
import litellm
from typing import Dict, Any
from vllm import LLM as vLLM


#TODO: this should go into the SLURM file

# Ensure all GPUs are visible
num_gpus = torch.cuda.device_count()
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(num_gpus))
print(f"Using {num_gpus} GPUs: {os.environ['CUDA_VISIBLE_DEVICES']}")

# Set NCCL variables to optimize multi-GPU and multi-node communication
os.environ["NCCL_DEBUG"] = "WARN"  # Reduce verbosity, change to "INFO" for debugging
''''
os.environ["NCCL_P2P_DISABLE"] = "0"  # Enable direct GPU-to-GPU communication
os.environ["NCCL_IB_DISABLE"] = "0"  # Enable Infiniband for multi-node
os.environ["NCCL_SHM_DISABLE"] = "0"  # Enable shared memory communication
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"  # Avoid deadlocks on NCCL failures
'''

# Enable better CUDA memory management (prevents OOM errors)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# Debugging Paths - Set these via SLURM later  #TODO:later delete this!
MODELS_DIR="/p/data1/mmlaion/marianna/models"
MODEL_CONFIG_FILE="/p/project/ccstdl/cipolina-kun1/open_spiel_arena/configs/models.json"
LITELLM_CONFIG_FILE = "/p/project/ccstdl/cipolina-kun1/open_spiel_arena/configs/litellm.json"

# Retrieve paths from environment variables
#MODELS_DIR = os.getenv("MODELS_DIR")   #Commented for debugging! TODO:delete this!
#MODEL_CONFIG_FILE = os.getenv("MODEL_CONFIG_FILE")
if MODELS_DIR is None or MODEL_CONFIG_FILE is None:
    raise ValueError("Error: Environment variables MODELS_DIR or MODEL_CONFIG_FILE are not set.")


LLM_REGISTRY: Dict[str, Dict[str, Any]] = {}

def initialize_llm_registry():
    """Initializes the LLM registry dynamically."""
    global LLM_REGISTRY
    MODEL_LIST = load_model_list() + load_litellm_model_list()
    for model in MODEL_LIST:
        LLM_REGISTRY[model] = {
            "display_name": model,
            "description": f"LLM model {model}.",
            "model_loader": lambda model_name=model: load_llm_model(model_name),
        }


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

#######################################################################
# Master Function: Detects Model Type & Calls Correct Loader
#######################################################################

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

    global CURRENT_LLM  # Explicitly refer to the global variable

    # Skip unloading if the model is already loaded
    if CURRENT_LLM is not None: # and CURRENT_LLM.model == model_name: #TODO: see if we need to load the model again!
        print(f"{model_name} is already loaded, skipping reload.")
        return CURRENT_LLM

    # Check if we have enough free GPU memory for another model
    free_memory = get_free_gpu_memory()
    required_memory = estimate_model_memory(model_name)

    if free_memory < required_memory:
        print(f"Not enough memory! Unloading {CURRENT_LLM.model} to load {model_name}...")
        # cleanup_vllm(CURRENT_LLM) # TODO
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
    """Detects if a model is quantized (4-bit, 8-bit) based on saved files.

    Args:
        model_path (str): The directory containing model files.

    Returns:
        str: The detected quantization type (`"4bit"`, `"8bit"`, or `"fp16"`).
    """
    if not os.path.exists(model_path):
        return "fp16"  # Default assumption if we can't check files

    files = os.listdir(model_path)

    if any("4bit" in f or "awq" in f or "ggml" in f for f in files):
        return "4bit"
    elif any("8bit" in f for f in files):
        return "8bit"
    else:
        return "fp16"

# Loads vLLM Models with Dynamic GPU Allocation
def load_vllm_model(model_name: str) -> vLLM:
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

    model_path = f"{MODELS_DIR}/{model_name}"

    model_name = 'gemma-2-27b-it' #'Mistral-7B-Instruct-v0.1' #TODO: later delete this!

    num_params = detect_model_size(model_name)  # Get the model size
    quantization = detect_quantization(model_path)
    total_free_memory = get_free_gpu_memory() # Get total free GPU memory

    # Adjust Memory Calculation Based on Quantization
    if quantization == "4bit":
        bytes_per_param = 0.5  # 4-bit quantization → 0.5 bytes per param
    elif quantization == "8bit":
        bytes_per_param = 1  # 8-bit quantization → 1 byte per param
    else:  # Default: FP16/BF16
        bytes_per_param = 2

    # Compute Required GPU Memory (GB)
    required_memory = (num_params * bytes_per_param)

    # Assign Tensor Parallelism Based on Model Size & Memory
    if required_memory > (total_free_memory * 0.9):  # Not enough memory even with multiple GPUs → OOM Risk
        raise MemoryError(f" Not enough GPU memory for {model_name}! Required: {required_memory:.2f} GB, Available: {total_free_memory:.2f} GB.")

    if required_memory > (torch.cuda.mem_get_info(0)[0] / 1e9 * 0.9):  # Model won't fit in a single GPU → Use TP
        tensor_parallel_size = min(torch.cuda.device_count(), max(2, required_memory // (torch.cuda.mem_get_info(0)[0] / 1e9 * 0.9)))
    else:
        tensor_parallel_size = 1  # Model fits in a single GPU

    # Assign GPUs
    assigned_gpus = list(range(tensor_parallel_size))
    assigned_gpus_str = ", ".join([f"cuda:{gpu}" for gpu in assigned_gpus])

    print(f" Loading {model_name} on GPUs [{assigned_gpus_str}] with TP={tensor_parallel_size}")

    #model_path = "/p/data1/mmlaion/marianna/models/Mistral-7B-Instruct-v0.1"
    model_path = "/p/data1/mmlaion/marianna/models/google/codegemma-7b-it"
    tensor_parallel_size = 2 # TODO: needs to be an even number!
    return vLLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        #device=f"cuda:{assigned_gpus[0]}",  # Gets the first GPU from the assigned list as the primary GPU
        gpu_memory_utilization=0.7,
        disable_custom_all_reduce=(tensor_parallel_size > 1),
        trust_remote_code=True,
        dtype="half",  # Force float16 on V100, bfloat16 on A100+
        max_num_batched_tokens=1096,  # 512 - Increase to match reduced max_model_len
        max_model_len=1096  # 512 -  Reduce max sequence length
    )

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



def get_free_gpu_memory() -> float:
    """Returns the total free GPU memory across all available GPUs (in GB)."""

    total_free_memory = 0.0
    num_gpus = torch.cuda.device_count()

    for gpu_id in range(num_gpus):
        free_mem = torch.cuda.mem_get_info(gpu_id)[0] / 1e9  # Convert to GB
        total_free_memory += free_mem

    print(f"Available GPU memory: {total_free_memory:.2f} GB") if total_free_memory == 0 else None
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
