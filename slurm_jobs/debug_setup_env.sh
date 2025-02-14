
export DEBUG=1  # Enable debugging mode

# New Keys, JSC
export ANTHROPIC_API_KEY=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
export OPENAI_API_KEY=sk-OXhU0WIUgABJu_J4NNSImwBhk0efVg-iKhjwmVQw6tT3BlbkFJZJgkYlWs0CQTfVqYV9kyh2t9fxCyu3aeNaTIePVvwA
export GEMINI_API_KEY=AIzaSyDb3Txuq7RLAyz4nLBd61_hN8u4duzjFyU
export MISTRAL_API_KEY=jh7Cu3lUNxsALVVuhQxFIjoNfjeX0QPK

export PYTHONPATH="/p/project/ccstdl/cipolina-kun1/open_spiel_arena:$PYTHONPATH"

# Set game names as an environment variable
export GAME_NAMES="kuhn_poker,matrix_rps,tic_tac_toe,connect_four"
# Relevant paths
export PYTHONPATH="/p/project/ccstdl/cipolina-kun1/open_spiel_arena:$PYTHONPATH"
export MODELS_DIR="/p/data1/mmlaion/marianna/models"
export MODEL_CONFIG_FILE="/p/project/ccstdl/cipolina-kun1/open_spiel_arena/configs/models.json"


export VLLM_ATTENTION_BACKEND="FLASHINFER"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib"
export GLOO_SOCKET_IFNAME=ib0
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
export NCCL_NET_GDR_LEVEL=0

export OPENBLAS_NUM_THREADS=1  # en caso de que haya muchos threads open
export OMP_NUM_THREADS=1

export VLLM_PLATFORM="cpu"  # Importante para VLLM!

module load CUDA
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# For longer evaluations
export PYDEVD_WARN_EVALUATION_TIMEOUT=10


#Para que vaya mas rapido
ray stop  # Stop any previous Ray instances
ray start --head --num-cpus=8 --num-gpus=1 # Start a fresh local Ray cluster

# Without this it gives weird no file errors
chmod u+r /p/project/ccstdl/cipolina-kun1/open_spiel_arena/scripts/simulate.py

python -m debugpy --listen 0.0.0.0:5678 --wait-for-client /p/project/ccstdl/cipolina-kun1/open_spiel_arena/scripts/simulate.py
