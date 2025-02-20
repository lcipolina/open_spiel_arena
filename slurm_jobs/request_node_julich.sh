#!/bin/bash
# SCRIPT USED FOR INTERACTIVE DEBUGGING ON JUWELS AND HOME VS CODE SSH CONNECTION
# Requests a compute node, captures & modifies the hostname for jump forwarding,
# and automatically starts the interactive debugger.
#
# Run as follows:
#   chmod +x slurm_jobs/request_node_julich.sh
#   ./slurm_jobs/request_node_julich.sh

echo "Checking available partitions with idle nodes..."

# Function to check if a partition has idle nodes
check_partition_idle() {
    local partition="$1"
    sinfo -h -p "$partition" -o "%t" | grep -qw "idle"
}

# Define partitions in order of preference
partitions=("develgpus" "gpus" "develbooster")
available_partition=""

# Iterate over partitions until one with idle nodes is found
for partition in "${partitions[@]}"; do
    if check_partition_idle "$partition"; then
        available_partition="$partition"
        break
    fi
done

if [[ -z "$available_partition" ]]; then
    echo "No available partitions with idle nodes found."
    exit 1
fi

echo "Selected partition: $available_partition"

# Set account and time limit based on the selected partition
case "$available_partition" in
    "develgpus")
        selected_account="cstdl"
        time_limit="2:00:00"
        ;;
    "gpus")
        selected_account="cstdl"
        time_limit="6:00:00"
        ;;
    "develbooster")
        selected_account="transfernetx"
        time_limit="6:00:00"
        ;;
    *)
        echo "Error: Unknown partition selected!"
        exit 1
        ;;
esac

echo "Using account: $selected_account with time limit: $time_limit"
echo "Requesting an interactive compute node..."

# Launch an interactive session via srun.
# Once connected, the compute node will:
#   - Print and modify its hostname.
#   - Save the modified hostname for jump forwarding.
#   - Activate your conda environment and start the interactive debugger.
srun --gres=gpu:1 \
     --partition="$available_partition" \
     --account="$selected_account" \
     --time="$time_limit" \
     --pty bash -c '
         HOST=$(hostname)
         MODIFIED_HOST="${HOST/.juwels/i.juwels}"
         echo "Original hostname: $HOST"
         echo "Modified hostname for home SSH: $MODIFIED_HOST"
         echo "$MODIFIED_HOST" > ~/current_compute_node.txt
         echo "Activating environment and launching interactive debugger..."
         source /p/scratch/laionize/cache-kun1/miniconda3/bin/activate /p/scratch/laionize/cache-kun1/llm && \
         python -m debugpy --listen 0.0.0.0:9000 --wait-for-client /p/project/ccstdl/cipolina-kun1/open_spiel_arena/scripts/simulate.py
     '
