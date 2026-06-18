#!/bin/bash
# Universal Slurm submission script for the A8 image analysis coursework.
#
# One-time setup on a login node:
#   export A8_SCRATCH_BASE="/rds/user/${USER}/hpc-work/a8"
#   mkdir -p "${A8_SCRATCH_BASE}"/{pip-cache,tmp,cache,matplotlib,torch}
#   python -m venv "${A8_SCRATCH_BASE}/venv"
#   ln -sfn "${A8_SCRATCH_BASE}/venv" .venv
#   source "${A8_SCRATCH_BASE}/venv/bin/activate"
#   TMPDIR="${A8_SCRATCH_BASE}/tmp" PIP_CACHE_DIR="${A8_SCRATCH_BASE}/pip-cache" python -m pip install --upgrade pip
#   TMPDIR="${A8_SCRATCH_BASE}/tmp" PIP_CACHE_DIR="${A8_SCRATCH_BASE}/pip-cache" python -m pip install -r requirements.txt
#
# Usage:
#   sbatch scripts/sbatch_run.sh python scripts/run_b1.py --device cuda --epochs 50
#
# By default, the wrapper uses ${A8_SCRATCH_BASE}/venv. To use a different venv path:
#   sbatch --export=ALL,A8_VENV=/path/to/venv scripts/sbatch_run.sh python scripts/run_b1.py --device cuda

#SBATCH -p ampere
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --job-name=a8_image_analysis
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "${REPO_ROOT}"

module purge || true
module load rhel8/default-amp || true

set -euo pipefail

export A8_SCRATCH_BASE="${A8_SCRATCH_BASE:-/rds/user/${USER}/hpc-work/a8}"
export A8_PIP_CACHE_DIR="${A8_PIP_CACHE_DIR:-${A8_SCRATCH_BASE}/pip-cache}"
export A8_TMP_DIR="${A8_TMP_DIR:-${A8_SCRATCH_BASE}/tmp}"
export A8_CACHE_DIR="${A8_CACHE_DIR:-${A8_SCRATCH_BASE}/cache}"
mkdir -p "${A8_PIP_CACHE_DIR}" "${A8_TMP_DIR}" "${A8_CACHE_DIR}" \
  "${A8_SCRATCH_BASE}/matplotlib" "${A8_SCRATCH_BASE}/torch"

export TMPDIR="${A8_TMP_DIR}"
export TEMP="${A8_TMP_DIR}"
export TMP="${A8_TMP_DIR}"
export PIP_CACHE_DIR="${A8_PIP_CACHE_DIR}"
export XDG_CACHE_HOME="${A8_CACHE_DIR}"
export MPLCONFIGDIR="${A8_SCRATCH_BASE}/matplotlib"
export TORCH_HOME="${A8_SCRATCH_BASE}/torch"

VENV_PATH="${A8_VENV:-${A8_SCRATCH_BASE}/venv}"
if [ ! -f "${VENV_PATH}/bin/activate" ]; then
  echo "Python virtual environment not found at '${VENV_PATH}'." >&2
  echo "Create it first with:" >&2
  echo "  export A8_SCRATCH_BASE=\"/rds/user/\${USER}/hpc-work/a8\"" >&2
  echo "  mkdir -p \"\${A8_SCRATCH_BASE}\"/{pip-cache,tmp,cache,matplotlib,torch}" >&2
  echo "  python -m venv \"\${A8_SCRATCH_BASE}/venv\"" >&2
  echo "  ln -sfn \"\${A8_SCRATCH_BASE}/venv\" .venv" >&2
  echo "  source \"\${A8_SCRATCH_BASE}/venv/bin/activate\"" >&2
  echo "  TMPDIR=\"\${A8_SCRATCH_BASE}/tmp\" PIP_CACHE_DIR=\"\${A8_SCRATCH_BASE}/pip-cache\" python -m pip install --upgrade pip" >&2
  echo "  TMPDIR=\"\${A8_SCRATCH_BASE}/tmp\" PIP_CACHE_DIR=\"\${A8_SCRATCH_BASE}/pip-cache\" python -m pip install -r requirements.txt" >&2
  exit 2
fi

set +u
source "${VENV_PATH}/bin/activate"
set -u

export PYTHONPATH="${PYTHONPATH:-}:${REPO_ROOT}/src:${REPO_ROOT}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${VENV_PATH}/lib"
export OMP_NUM_THREADS=16
export PYTHONUNBUFFERED=1

mkdir -p logs
logfile="logs/${SLURM_JOB_NAME:-job}_${SLURM_JOB_ID:-manual}_$(date +%Y-%m-%d_%H-%M-%S).log"

{
  echo "=========================================="
  echo "Job:       ${SLURM_JOB_ID:-manual}"
  echo "Node:      $(hostname)"
  echo "GPUs:      ${CUDA_VISIBLE_DEVICES:-unset}"
  echo "Repo root: ${REPO_ROOT}"
  echo "Venv:      ${VENV_PATH}"
  echo "Scratch:   ${A8_SCRATCH_BASE}"
  echo "TMPDIR:    ${TMPDIR}"
  echo "Pip cache: ${PIP_CACHE_DIR}"
  echo "Python:    $(command -v python)"
  echo "Command:   $@"
  echo "Logging to: ${logfile}"
  echo "Start Time: $(date)"
  echo "=========================================="
  python - <<'PY_CHECK'
import torch
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda_device={torch.cuda.get_device_name(0)}")
PY_CHECK
  "$@"
  echo "------------------------------------------"
  echo "End Time:   $(date)"
} 2>&1 | tee "$logfile"
