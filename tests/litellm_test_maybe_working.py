import litellm

def generate_text(prompt: str, model: str = "gpt2") -> str:
    """Generate text using LiteLLM in CPU mode.

    Args:
        prompt: The input text to generate a response for.
        model: The Hugging Face model name.

    Returns:
        str: The generated text.
    """
    response = litellm.completion(
        model=model,
        model_provider="huggingface",  # Fix: Explicitly specify Hugging Face
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100
    )
    return response["choices"][0]["message"]["content"]

if __name__ == "__main__":
    user_prompt = "What is the capital of France?"
    response = generate_text(user_prompt)
    print("Response:", response)
