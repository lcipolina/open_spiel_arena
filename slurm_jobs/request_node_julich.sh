#!/bin/bash

# SCRIPT USED FOR INTERACTIVE DEBUGGING ON JUWELS AND HOME VS CODE SSH CONNECTION

# Requests a compute node and prints the modified name for me to do jump forwarding from home.
# Extracts the host name and modifies it to connect from home
# Saves the modified hostname to ~/current_compute_node.txt (accessible from your home PC via SSH).


# Run like this: ./slurm_jobs/request_node_julich.sh
#chmod +x slurm_jobs/request_node_julich.sh


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

# Start an interactive session with srun.
# The command first captures the compute node's hostname,
# replaces '.juwels' with 'i.juwels', writes the modified hostname to a file,
# and then gives you an interactive bash shell.
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
         exec bash
     '
