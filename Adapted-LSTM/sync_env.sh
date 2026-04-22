#!/usr/bin/env bash
set -euo pipefail

# Export the *current active* Python environment into a lock-style file.
#
# Usage (on the server):
#   cd /workspace/CITS5206-Capstone-Group-17/Adapted-LSTM
#   source /venv/main/bin/activate   # or your project venv
#   ./sync_env_from_ssm.sh
#
# Output:
#   requirements.lock.txt (pip freeze from the active env)

cd "$(dirname "$0")"

python -m pip --version >/dev/null
python -m pip freeze > requirements.lock.txt

echo "Wrote: $(pwd)/requirements.lock.txt"

