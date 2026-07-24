#!/usr/bin/env bash
# Source this file from the repository root:
#   source scripts/activate_kds.sh
if ! declare -F conda >/dev/null 2>&1; then
  # Conda is installed here on the target server. Loading conda.sh is required
  # for `conda activate` in non-interactive shells.
  source /home/conda/etc/profile.d/conda.sh
fi
export CONDARC="/home/wcho/Occlusion_renderer/.condarc"
conda activate kds-occlusion
