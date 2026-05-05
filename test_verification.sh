#!/bin/bash
set -e

echo "=== 1. Testing Rollback ==="
echo "def add(a, b): return a + b" > test_rollback.py
sumo run "Add type hints to all functions in test_rollback.py" --no-committee
RUN_ID=$(sumo snapshots list | awk 'NR==4 {print $1}' | grep -v 'Run')
if [ -z "$RUN_ID" ]; then
    echo "No run ID found"
    exit 1
fi
echo "Rolling back run: $RUN_ID"
sumo rollback "$RUN_ID" -y
cat test_rollback.py
rm test_rollback.py
