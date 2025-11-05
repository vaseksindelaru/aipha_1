# run_local_monitor.sh - Script to run local code monitoring for Shadow
#!/bin/bash

# Configuration
LOCAL_CODE_PATH="/home/vaclav/Aipha_0.0.1"
SHADOW_MEMORY_PATH="./aipha_memory_storage/action_history"

# Create directories if they don't exist
mkdir -p "$SHADOW_MEMORY_PATH"

# Set Python path to include shadow directory
export PYTHONPATH="${PYTHONPATH}:$(pwd)/shadow"

echo "=== Local Code Monitor for Shadow Real-time Understanding ==="
echo "Local path: $LOCAL_CODE_PATH"
echo "Memory path: $SHADOW_MEMORY_PATH"
echo ""

# Function to run monitoring check
run_monitor() {
    echo "=== $(date) - Running local code monitor ==="
    python3 shadow/local_code_monitor.py \
        --local-path "$LOCAL_CODE_PATH" \
        --memory-path "$SHADOW_MEMORY_PATH"
    echo ""
}

# Initial check and codebase analysis
echo "Performing initial local codebase analysis..."
run_monitor

# Continuous monitoring loop
echo "Starting continuous local code monitoring (Ctrl+C to stop)..."
echo "Shadow will now understand code changes in real-time..."
echo ""

python3 shadow/local_code_monitor.py \
    --local-path "$LOCAL_CODE_PATH" \
    --memory-path "$SHADOW_MEMORY_PATH" \
    --continuous