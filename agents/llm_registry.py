"""
llm_registry.py - Central llm registry 
"""

from transformers import pipeline

# Registry of available LLMs
# These are just examples of models that run fast enough to work in this proof of concept.
# But they don't give us good results.
# Ideally we would use 'intruct-finetuned' models.
# I have tried some models and these actually worked well (i.e. followed the prompt):
#       microsoft/Phi-3-mini-4k-instruct, Qwen/Qwen2.5-Coder-32B-Instruct,
#       Qwen/Qwen2.5-72B-Instruct, mistralai/Mistral-7B-Instruct-v0.3,dolly-v2-3b, dolly-v2-12b

# Note. Need to read the models documentation on how to prompt them. See example for Microsoft's Phi.


LLM_REGISTRY = {
    "gpt2": {
        "display_name": "GPT-2",
        "description": "A medium-sized transformer-based language model by OpenAI.",
        "model_loader": lambda: pipeline("text-generation", model="gpt2"),
    },
    "flan_t5_small": {
        "display_name": "FLAN-T5 Small",
        "description": "A fine-tuned T5 model optimized for instruction-following tasks.",
        "model_loader": lambda: pipeline("text-generation", model="google/flan-t5-small"),
    },
    "distilgpt2": {
        "display_name": "DistilGPT-2",
        "description": "A smaller and faster version of GPT-2.",
        "model_loader": lambda: pipeline("text-generation", model="distilgpt2"),
    },
}
