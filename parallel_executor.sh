#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 <command_file> [max_parallel]"
    echo "  <command_file>: File containing commands to execute"
    echo "  [max_parallel]: Maximum number of parallel processes (default: 3)"
    exit 1
}

# Check if at least a file name is provided
if [ $# -eq 0 ]; then
    usage
fi

command_file="$1"
max_parallel=${2:-3}  # Use the second argument if provided, otherwise default to 3

# Validate max_parallel is a positive integer
if ! [[ "$max_parallel" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: max_parallel must be a positive integer"
    usage
fi

# Function to run a command
run_command() {
    local cmd="$1"
    echo "Starting: $cmd"
    eval "$cmd"
    echo "Finished: $cmd"
}

# Read commands from file and store in an array, ignoring blank lines and lines with only dashes
mapfile -t commands < <(sed -e '/^[[:space:]]*$/d' -e '/^-\+$/d' "$command_file")

# Initialize an array to store background PIDs
pids=()

# Process commands
for cmd in "${commands[@]}"; do
    # Wait if we're already running max_parallel commands
    while [ ${#pids[@]} -ge $max_parallel ]; do
        for i in "${!pids[@]}"; do
            if ! kill -0 ${pids[$i]} 2>/dev/null; then
                unset 'pids[$i]'
            fi
        done
        pids=("${pids[@]}")  # Re-index array
        [ ${#pids[@]} -ge $max_parallel ] && sleep 1
    done

    # Run the command in background and store its PID
    run_command "$cmd" &
    pids+=($!)
done

# Wait for all remaining background processes to finish
for pid in "${pids[@]}"; do
    wait $pid
done

echo "All commands completed."