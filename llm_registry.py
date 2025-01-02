from transformers import pipeline

# Registry of available LLMs
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
