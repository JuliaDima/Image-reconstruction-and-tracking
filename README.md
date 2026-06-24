# Image Analysis Coursework

[![pipeline status](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/a8_coursework/eid23/badges/main/pipeline.svg)](https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/a8_coursework/eid23/-/commits/main)

This repository contains the coursework implementation for the MPhil in Data
Intensive Science image analysis assignment.

The implementation covers the following structure: Part I.A classical restoration, Part I.B unrolled reconstruction experiments, Part II.A Horn-Schunck optical flow, Part II.B YOLO segmentation utilities, and Part II.C cell tracking utilities.

## Repository layout

```
src/image_analysis_coursework/   installable package, one module per task
  a1.py a2.py unrolling.py        Part I forward model, TV restoration, unrolled PGD
  motion.py yolo_segmentation.py tracking.py   Part II flow, segmentation, tracking
  cli_*.py                        thin command-line entry points
scripts/                         runnable wrappers + Slurm submission helpers
tests/                           pytest suite (run in GitLab CI)
report/                          LaTeX source, self-contained figures/, and the PDF
Dockerfile, .pre-commit-config.yaml, pyproject.toml, LICENSE
```

Every experiment seeds NumPy, PyTorch, and Python and writes its figures and a
JSON metadata/metrics file, so the numbers quoted in the report can be traced
back to `outputs/`.

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
python scripts/run_motion.py --index 0 --alpha 1.0 --iterations 100 \
  --sweep-alphas 0.3,1,3 --sweep-iterations 10,50,100,200
```

Outputs include the frame pair, absolute-difference image, energy trace, quiver
plot, HSV flow map, parameter sweep, report-ready summary, and metadata under
`outputs/part_ii/a/`.

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
python scripts/run_yolo_predict.py \
  --model-path outputs/part_ii/b/yolo_runs/cells/weights/best.pt \
  --image-path data/PhC-C2DH-U373/01/t000.tif \
               data/PhC-C2DH-U373/01/t057.tif \
               data/PhC-C2DH-U373/01/t114.tif \
  --output-path outputs/part_ii/b/prediction_montage.png \
  --confidence 0.25
```

The conversion and rasterisation utilities are in
`image_analysis_coursework.yolo_segmentation`. Export writes binary round-trip
IoU statistics, while training writes a compact JSON summary containing scalar
mask metrics and the canonical checkpoint path. Full YOLO training requires the
optional `ultralytics` dependency.

## Part II.C

Segment every sequence `01` frame with one loaded model:

```bash
python scripts/run_yolo_sequence.py \
  --model-path outputs/part_ii/b/yolo_runs/cells/weights/best.pt \
  --data-dir data --sequence 01 \
  --output-dir outputs/part_ii/c/labels_yolo \
  --confidence 0.25
```

Then track the detected centroids:

```bash
python scripts/run_tracking.py \
  --labels-dir outputs/part_ii/c/labels_yolo \
  --first-frame-path data/PhC-C2DH-U373/01/t000.tif \
  --output-dir outputs/part_ii/c_yolo \
  --max-distance 35 --max-gap 1
```

`laptrack.LapTrack` is used when installed; otherwise the code falls back only
when the dependency is absent. Runtime tracker errors are surfaced. Metadata
records the actual backend, thresholds, physical distance conversion, and track
fragmentation statistics. Outputs are written to `outputs/part_ii/c_yolo/`.


## Full Pipeline Submission

After the RDS venv and optional ML dependencies are installed, submit the full final-run pipeline with:

```bash
bash scripts/submit_full_pipeline.sh
```

This submits Slurm jobs for YOLO export, full B(i), B(ii), B(iii) GPU training, YOLO segmentation training, sequence-01 YOLO segmentation, and tracking from YOLO-derived masks. Useful overrides:

```bash
B_EPOCHS=75 YOLO_EPOCHS=150 bash scripts/submit_full_pipeline.sh
YOLO_MODEL=yolo11n-seg.pt YOLO_DEVICE=0 bash scripts/submit_full_pipeline.sh
YOLO_CONFIDENCE=0.25 bash scripts/submit_full_pipeline.sh
```

Full-run outputs are written to `outputs/part_i/b1_full/`,
`outputs/part_i/b2_full/`, `outputs/part_i/b3_full/`,
`outputs/part_ii/b/yolo_runs/`, and `outputs/part_ii/c_yolo/`.

## Supplementary analysis figures

Two CPU-only figures used in the report (the PSF and its modulation transfer
function, and the Part I.A.2 reconstruction-error maps) are produced with:

```bash
python scripts/run_extra_figures.py
```

## Report

The LaTeX report source is in `report/main.tex`; the submission PDF is
`report/main.pdf`. All figures it uses are committed under `report/figures/`, so
the report rebuilds from a clean clone with `pdflatex main.tex` (run twice to
resolve references).

## Docker

A CPU image reproduces the environment for the classical and motion experiments:

```bash
docker build -t image-analysis .
docker run --rm image-analysis                       # runs the test suite
docker run --rm -v "$PWD/outputs:/app/outputs" image-analysis \
    python scripts/run_a2.py                         # runs an experiment
```

GPU training (Part I.B, YOLO) is run on the cluster through `scripts/sbatch_run.sh`.

## Development and code quality

Style is checked with `ruff` (configured in `pyproject.toml`) and enforced
through `pre-commit`. The `no-commit-to-branch` hook protects `main`, so
development happens on feature branches merged through merge requests:

```bash
pip install pre-commit && pre-commit install
pre-commit run --all-files
```

## Tests

Run the tests with:

```bash
pytest
# or explicitly with the A8 venv
/rds/user/${USER}/hpc-work/a8/venv/bin/python -m pytest
```

## Licence

Released under the MIT licence; see [LICENSE](LICENSE).

## AI Assistance

Claude (Anthropic) was used for the following tasks in this project:

- **Figure generation** — iterating on the Python plotting scripts to produce ...
- **LaTeX compilation** — debugging LaTeX/BibTeX errors, resolving package conflicts, and ensuring the report compiled cleanly to a single coherent PDF.
- **Report appearance** — refining layout, caption sizing, figure placement, and diagrams.

All experimental design, training runs, mathematical derivations, and written analysis are the author's own work.