import litellm
import os

# Set API keys via environment variables (for OpenAI, Anthropic, etc.)
os.environ["OPENAI_API_KEY"] = "your_openai_api_key"
os.environ["HUGGINGFACE_API_KEY"] = "your_huggingface_api_key"

def generate_text(prompt: str, model: str = "gpt-3.5-turbo") -> str:
    """Generate text using LiteLLM.

    Args:
        prompt: The input text to generate a response for.
        model: The model name (e.g., OpenAI or Hugging Face model).

    Returns:
        str: The generated response.
    """
    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100
    )
    return response["choices"][0]["message"]["content"]

if __name__ == "__main__":
    user_prompt = "What is the capital of France?"
    response = generate_text(user_prompt)
    print("Response:", response)
