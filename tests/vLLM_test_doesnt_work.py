from vllm import LLM, SamplingParams
import vllm

'''DOESNT WORK ON LOGIN NODES - need to try on SLURM'''

def generate_text(model_name: str, prompt: str, max_tokens: int = 100) -> str:
    """Generate text using vLLM.

    Args:
        model_name: The name or path of the LLM model.
        prompt: The input text to generate a response for.
        max_tokens: The maximum number of tokens to generate.

    Returns:
        str: The generated text.
    """

    # export CUDA_VISIBLE_DEVICES=0
    '''
    llm = LLM(
        model=model_name,
        device="cuda",
        enforce_eager=True,       # Forces eager execution (disables async)
        disable_log_stats=True,   # Disables logging that might trigger async processes
        gpu_memory_utilization=0.8  # Helps vLLM optimize memory use
    )
    '''

    llm = LLM(
            model=model_name,
            device="cpu",                # Force CPU mode
            enforce_eager=True,           # Disable async processing
            disable_log_stats=True        # Prevents potential async logging issues
        )


    sampling_params = SamplingParams(max_tokens=max_tokens)

    outputs = llm.generate(prompt, sampling_params)
    return outputs[0].text if outputs else ""

if __name__ == "__main__":
    #model_name = "mistralai/Mistral-7B-Instruct-v0.2" #doesn't work needs authentication
    #model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    #model_name = "tiiuae/falcon-7b-instruct"
    model_name = "gpt2" # it works, it downloads something


    user_prompt = "What is the capital of France?"
    response = generate_text(model_name, user_prompt)

    print("Response:", response)
