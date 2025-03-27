# utils/llm_utils.py
"""Utility functions for Large Language Model (LLM) integration.

This module provides helper functions to generate prompts and interact with LLMs
for decision-making in game simulations.
"""

import os
import json
from typing import List, Optional
from transformers import AutoTokenizer, pipeline

from agents.llm_registry import initialize_llm_registry

initialize_llm_registry() #TODO: I don't kike this design!
from agents.llm_registry import LLM_REGISTRY

#TODO: all this goes into the SLURM later! delete!
# Set environment variable to allow PyTorch to dynamically allocate more memory on GPUs
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

#TODO: add model_name on the arguments so we can try different models
def format_prompt(input_text:str,request_explanation=True)->str:
    """Formats the input prompt for reasoning-first output, using Hugging Face's
    chat template function.

    Args:
        input_text (str): The game prompt.
        request_explanation (bool): Whether to request the model's reasoning.

    Returns:
        str: The correctly formatted prompt for the model.
    """

    # TODO: construir esto con el model name nomas y el model path del OS
    model_path = "/p/data1/mmlaion/marianna/models/google/codegemma-7b-it"

    print("'llm_utils.py': Using hardcoded codegemma-7b-it, for the prompt tokenizer ! fix me!!")

    print('format_prompt(prompt_string):', input_text) #TODO: delete this! just for debugging

    # If explanation is requested, add a message.
    if request_explanation:
       input_text += (
            "\n\nFirst, think through the game strategy and explain your reasoning."
            "\nOnly after that, decide on the best action to take."
        )

    # Modify prompt to request structured JSON output
    json_instruction = (
        "\n\nReply in the following JSON format:\n"
        "{\n  'reasoning': <str>,\n  'action': <int>\n}"
    )

    input_text += json_instruction  # Append JSON request
    messages = [{"role": "user", "content": input_text}]

    # Format using apply_chat_template
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return formatted_prompt  # This is passed to vLLM
