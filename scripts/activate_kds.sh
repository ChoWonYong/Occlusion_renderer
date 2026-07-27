#!/usr/bin/env bash
# Activate the kds-occlusion environment. Source it from any working directory:
#   source /path/to/Occlusion_renderer/scripts/activate_kds.sh
#
# The envs live in a repo-local envs_dir declared only in the repo's .condarc,
# and conda does not read that file on its own, so CONDARC has to be exported
# before `conda activate` can resolve the env by name.

# Repo root taken from this script's own location, so no path is hardcoded.
__kds_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

if ! declare -F conda >/dev/null 2>&1; then
  # `conda activate` needs conda's shell function, which non-interactive shells
  # do not load. Locate conda.sh via CONDA_EXE, then via conda on PATH.
  __kds_base=""
  if [ -n "${CONDA_EXE:-}" ]; then
    __kds_base="$(dirname -- "$(dirname -- "${CONDA_EXE}")")"
  elif command -v conda >/dev/null 2>&1; then
    __kds_base="$(conda info --base 2>/dev/null)"
  fi
  if [ -r "${__kds_base}/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    . "${__kds_base}/etc/profile.d/conda.sh"
  else
    echo "activate_kds.sh: conda.sh not found; set CONDA_EXE or put conda on PATH." >&2
    unset __kds_repo __kds_base
    return 1 2>/dev/null || exit 1
  fi
  unset __kds_base
fi

export CONDARC="${__kds_repo}/.condarc"
unset __kds_repo
conda activate kds-occlusion
