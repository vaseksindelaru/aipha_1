# run_github_monitor.sh - Script to run GitHub repository monitoring for Shadow
#!/bin/bash

# Configuration
REPO_URL="https://github.com/vaseksindelaru/aipha_0.0.1.git"
LOCAL_REPO_PATH="./monitored_repos/aipha_0.0.1"
SHADOW_MEMORY_PATH="./aipha_memory_storage/action_history"
CHECK_INTERVAL=300  # 5 minutes

# Create directories if they don't exist
mkdir -p "$LOCAL_REPO_PATH"
mkdir -p "$SHADOW_MEMORY_PATH"

# Set Python path to include shadow directory
export PYTHONPATH="${PYTHONPATH}:$(pwd)/shadow"

echo "=== GitHub Repository Monitor for Shadow ==="
echo "Repository: $REPO_URL"
echo "Local path: $LOCAL_REPO_PATH"
echo "Memory path: $SHADOW_MEMORY_PATH"
echo "Check interval: ${CHECK_INTERVAL} seconds"
echo ""

# Function to run monitoring check
run_monitor() {
    echo "=== $(date) - Running GitHub monitor check ==="
    python3 shadow/github_monitor.py \
        --repo-url "$REPO_URL" \
        --local-path "$LOCAL_REPO_PATH" \
        --memory-path "$SHADOW_MEMORY_PATH"
    echo ""
}

# Initial check
run_monitor

# Continuous monitoring loop
echo "Starting continuous monitoring (Ctrl+C to stop)..."
while true; do
    echo "Waiting ${CHECK_INTERVAL} seconds until next check..."
    sleep $CHECK_INTERVAL
    run_monitor
done