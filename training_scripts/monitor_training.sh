#!/bin/bash
# Monitor training progress with filtered output

LOG_FILE="${1:-logs/training_rotaug_*.log}"

echo "Monitoring training progress..."
echo "Press Ctrl+C to stop monitoring"
echo ""

# Show only important messages (epochs, losses, errors)
tail -f $LOG_FILE 2>/dev/null | grep --line-buffered -E "Epoch|Train|Val|loss|ERROR|Error|Exception|Traceback|Finished|Starting|Resuming|checkpoint|WARNING.*optimizer|WARNING.*scaler|WARNING.*loss" || {
    echo "No log file found matching: $LOG_FILE"
    echo "Available log files:"
    ls -lt logs/training_rotaug_*.log 2>/dev/null | head -5
}
