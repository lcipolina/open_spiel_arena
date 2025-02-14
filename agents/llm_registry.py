"""
llm_registry.py - Central llm registry
Registry of available LLMs
"""

import os
import json
from typing import Dict, Any

# Set CUDA paths to ensure GPU availability #TODO: eventually delete this. It's for JFZ
os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ["PATH"]
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")

# Force vLLM to recognize the correct execution platform
os.environ["VLLM_PLATFORM"] = "cuda"  # Use "cpu" if debugging

# Ensure PyTorch detects GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # Adjust to match your GPU setup

from vllm import LLM

# For Debugging:  #TODO:delete this!
MODELS_DIR="/p/data1/mmlaion/marianna/models"
MODEL_CONFIG_FILE="/p/project/ccstdl/cipolina-kun1/open_spiel_arena/configs/models.json"

# Retrieve paths from environment variables
#MODELS_DIR = os.getenv("MODELS_DIR")   #Commented for debugging! TODO:delete this!
#MODEL_CONFIG_FILE = os.getenv("MODEL_CONFIG_FILE")
if MODELS_DIR is None or MODEL_CONFIG_FILE is None:
    raise ValueError("Error: Environment variables MODELS_DIR or MODEL_CONFIG_FILE are not set.")

def load_model_list() -> list:
    """
    Loads the list of LLM models from a JSON file.

    Returns:
        List[str]: List of model names.
    """
    with open(MODEL_CONFIG_FILE, "r") as f: # type: ignore
        data = json.load(f)
    return data["models"]


def load_vllm_model(model_name: str) -> LLM:
    """
    Loads a model using vLLM from the pre-downloaded model directory.

    Args:
        model_name (str): The name of the model to load.

    Returns:
        LLM: An instance of the vLLM model.
    """
    model_path = f"{MODELS_DIR}/{model_name}"
    model = LLM(model="/p/data1/mmlaion/marianna/models/deepseek-coder-33b-instruct", tensor_parallel_size=1, device="cuda")
    return model


# Dynamically register all models from JSON
LLM_REGISTRY: Dict[str, Dict[str, Any]] = {}
MODEL_LIST = load_model_list()

for model in MODEL_LIST:
    LLM_REGISTRY[model] = {
        "display_name": model,
        "description": f"LLM model {model} using vLLM.",
        "model_loader": lambda model_name=model: load_vllm_model(model_name),
    }