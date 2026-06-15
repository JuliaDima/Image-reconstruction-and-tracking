# Image Analysis Coursework

This repository contains the coursework implementation for the MPhil in Data
Intensive Science image analysis assignment.

The current implementation covers Part I.A.1, Part I.A.2, and Part I.B(i): simulating the
forward image acquisition process, then applying classical total variation (TV)
denoising and a Gaussian-filter baseline.

## Setup

Use a project-local Python virtual environment for this coursework repository.
Large pip caches and temporary build files should go on RDS rather than the small
local filesystem. Create the environment once from the repository root with:

```bash
export A8_SCRATCH_BASE="/rds/user/${USER}/hpc-work/a8"
mkdir -p "${A8_SCRATCH_BASE}"/{pip-cache,tmp,cache,matplotlib,torch}
python -m venv .venv
source .venv/bin/activate
TMPDIR="${A8_SCRATCH_BASE}/tmp" PIP_CACHE_DIR="${A8_SCRATCH_BASE}/pip-cache" python -m pip install --upgrade pip
TMPDIR="${A8_SCRATCH_BASE}/tmp" PIP_CACHE_DIR="${A8_SCRATCH_BASE}/pip-cache" python -m pip install -r requirements.txt
```

If a previous install failed with `No space left on device`, remove the partial
venv and recreate it with the commands above:

```bash
rm -rf .venv
```

For later local runs, reactivate it with:

```bash
source .venv/bin/activate
```

## GPU Jobs With Slurm

The Slurm wrapper `scripts/sbatch_run.sh` activates `.venv` by default, sets the
project `PYTHONPATH`, routes caches/temp files to
`/rds/user/${USER}/hpc-work/a8`, logs to `logs/`, and prints PyTorch/CUDA
availability at job start. Override the scratch base with `A8_SCRATCH_BASE` if
needed.

Submit the Part I.B(i) GPU training job from the repository root with:

```bash
sbatch scripts/sbatch_run.sh python scripts/run_b1.py --device cuda --epochs 50
```

To use a different virtual environment, pass `A8_VENV`:

```bash
sbatch --export=ALL,A8_VENV=/path/to/venv,A8_SCRATCH_BASE=/rds/user/${USER}/hpc-work/a8 scripts/sbatch_run.sh python scripts/run_b1.py --device cuda --epochs 50
```

## Part I.A.1

Run the forward acquisition simulation:

```bash
python scripts/run_a1.py
```

By default this downloads the PhC-C2DH-U373 dataset into `data/`, selects the
first TIFF frame from sequence `01`, and writes figures plus metadata to
`outputs/part_i/a1/`.

Useful options:

```bash
python scripts/run_a1.py --help
python scripts/run_a1.py --image-path path/to/image.tif
python scripts/run_a1.py --sigma-blur 2.0 --noise-std 0.001 --seed 42
```

## Part I.A.2

Run the classical restoration experiment:

```bash
python scripts/run_a2.py
```

This reuses the A.1 acquisition settings, evaluates TV denoising for three
regularisation strengths, compares against a Gaussian filter, and writes figures
plus metrics to `outputs/part_i/a2/`.

Useful options:

```bash
python scripts/run_a2.py --help
python scripts/run_a2.py --lambdas 0.001,0.01,0.08 --iterations 250
python scripts/run_a2.py --image-path path/to/image.tif
```

## Part I.B(i)

Train and evaluate the unrolled PGD model locally for a short smoke run:

```bash
python scripts/run_b1.py --epochs 1 --max-train-images 2 --max-test-images 2 --device cpu
```

Submit the full training job to the GPU queue:

```bash
sbatch scripts/sbatch_run.sh python scripts/run_b1.py --device cuda --epochs 50
# or with another venv:
sbatch --export=ALL,A8_VENV=/path/to/venv,A8_SCRATCH_BASE=/rds/user/${USER}/hpc-work/a8 scripts/sbatch_run.sh python scripts/run_b1.py --device cuda --epochs 50
```

The implementation uses a 6-step unrolled PGD network, iteration-specific CNN
proximal maps, learnable nonnegative step sizes, L1 training loss, blur kernel
`11x11` with `sigma=2.0`, and noise variance `0.001`. Outputs are written to
`outputs/part_i/b1/`.

## Report

The LaTeX report draft is in `report/main.tex`. It covers Part I.A.1, Part I.A.2, and Part
I.B(i), and uses `\IfFileExists` so it compiles before and after generating the
figures.

## Tests

Run the tests with:

```bash
pytest
```
