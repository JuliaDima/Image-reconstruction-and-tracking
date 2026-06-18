# Image Analysis Coursework

This repository contains the coursework implementation for the MPhil in Data
Intensive Science image analysis assignment.

The implementation covers the following structure: Part I.A classical restoration, Part I.B unrolled reconstruction experiments, Part II.A Horn-Schunck optical flow, Part II.B YOLO segmentation utilities, and Part II.C cell tracking utilities.

## Setup

Use a project-local Python virtual environment for this coursework repository.
Large pip caches and temporary build files should go on RDS rather than the small
local filesystem. Create the environment once from the repository root with:

```bash
export A8_SCRATCH_BASE="/rds/user/${USER}/hpc-work/a8"
mkdir -p "${A8_SCRATCH_BASE}"/{pip-cache,tmp,cache,matplotlib,torch}
python -m venv "${A8_SCRATCH_BASE}/venv"
source "${A8_SCRATCH_BASE}/venv/bin/activate"
TMPDIR="${A8_SCRATCH_BASE}/tmp" PIP_CACHE_DIR="${A8_SCRATCH_BASE}/pip-cache" python -m pip install --upgrade pip
TMPDIR="${A8_SCRATCH_BASE}/tmp" PIP_CACHE_DIR="${A8_SCRATCH_BASE}/pip-cache" python -m pip install -r requirements.txt
# For full YOLO/laptrack runs, also install:
TMPDIR="${A8_SCRATCH_BASE}/tmp" PIP_CACHE_DIR="${A8_SCRATCH_BASE}/pip-cache" python -m pip install -r requirements-ml.txt
```

If a previous install failed with `No space left on device`, remove the partial
venv and recreate it with the commands above:

```bash
rm -rf "${A8_SCRATCH_BASE}/venv"
```

For later local runs, reactivate it with:

```bash
source "/rds/user/${USER}/hpc-work/a8/venv/bin/activate"
```

## GPU Jobs With Slurm

The Slurm wrapper `scripts/sbatch_run.sh` activates an RDS-backed venv by default (`/rds/user/${USER}/hpc-work/a8/venv`), sets the project `PYTHONPATH`, routes caches/temp files to `/rds/user/${USER}/hpc-work/a8`, logs to `logs/`, and prints PyTorch/CUDA availability at job start. Override the scratch base with `A8_SCRATCH_BASE` or the environment with `A8_VENV` if needed.

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


## Part I.B(ii)

Train and evaluate the out-of-distribution unrolled PGD experiment:

```bash
python scripts/run_b2.py --epochs 1 --max-train-images 2 --max-test-images 2 --device cpu
sbatch scripts/sbatch_run.sh python scripts/run_b2.py --device cuda --epochs 50
```

This trains on `21x21`, `sigma=4.0` blur and evaluates on the `11x11`, `sigma=2.0` test degradation. Outputs are written to `outputs/part_i/b2/`.

## Part I.B(iii)

Train the shared-parameter unrolled PGD model and evaluate longer iteration counts:

```bash
python scripts/run_b3.py --epochs 1 --max-train-images 2 --max-test-images 1 --device cpu
sbatch scripts/sbatch_run.sh python scripts/run_b3.py --device cuda --epochs 50
```

The convergence table and plot for `T=K,4K,8K,16K` are written to `outputs/part_i/b3/`.

## Part II.A

Run Horn-Schunck optical flow on a consecutive pair from sequence `01`:

```bash
python scripts/run_motion.py --index 0 --alpha 1.0 --iterations 200
```

Outputs include the frame pair, absolute-difference image, energy trace, quiver plot, HSV flow map, and metadata under `outputs/part_ii/a/`.

## Part II.B

Export segmentation labels from `01_GT/SEG` and `02_GT/SEG` to YOLO segmentation format:

```bash
python scripts/export_yolo.py
```

Train YOLO segmentation with hardware acceleration when available:

```bash
sbatch scripts/sbatch_run.sh python scripts/train_yolo.py --yolo-dataset-dir outputs/part_ii/b/yolo_dataset --epochs 100 --device 0
```

Overlay predictions from a trained model:

```bash
python scripts/run_yolo_predict.py --model-path path/to/best.pt --image-path data/PhC-C2DH-U373/01/t000.tif
```

The conversion and rasterisation utilities are in `image_analysis_coursework.yolo_segmentation`. Full YOLO training requires the optional `ultralytics` dependency.

## Part II.C

After segmenting all sequence `01` frames, track the cell centroids:

```bash
python scripts/run_tracking.py --labels-dir outputs/part_ii/c/labels --first-frame-path data/PhC-C2DH-U373/01/t000.tif
```

`laptrack.LapTrack` is used when installed; otherwise the code falls back to a deterministic nearest-neighbour tracker for testing and smoke runs. Outputs are written to `outputs/part_ii/c/`.


## Full Pipeline Submission

After the RDS venv and optional ML dependencies are installed, submit the full final-run pipeline with:

```bash
bash scripts/submit_full_pipeline.sh
```

This submits Slurm jobs for YOLO export, full B(i), B(ii), B(iii) GPU training, YOLO segmentation training, sequence-01 YOLO segmentation, and tracking from YOLO-derived masks. Useful overrides:

```bash
B_EPOCHS=75 YOLO_EPOCHS=150 bash scripts/submit_full_pipeline.sh
YOLO_MODEL=yolo11n-seg.pt YOLO_DEVICE=0 bash scripts/submit_full_pipeline.sh
```

Full-run outputs are written to `outputs/part_i/b1_full/`, `outputs/part_i/b2_full/`, `outputs/part_i/b3_full/`, `outputs/part_ii/b/yolo_runs/`, and `outputs/part_ii/c_yolo/`. Refresh the report metrics after these jobs finish.

## Report

The LaTeX report draft is in `report/main.tex`. It covers all coursework sections, and uses `\IfFileExists` so it compiles before and after generating the
figures.

## Tests

Run the tests with:

```bash
pytest
# or explicitly with the A8 venv
/rds/user/${USER}/hpc-work/a8/venv/bin/python -m pytest
```
