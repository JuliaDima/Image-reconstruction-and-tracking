# Image Analysis Coursework

This repository contains the coursework implementation for the MPhil in Data
Intensive Science image analysis assignment.

The current implementation covers Part I.A.1 and Part I.A.2: simulating the
forward image acquisition process, then applying classical total variation (TV)
denoising and a Gaussian-filter baseline.

## Setup

Create and activate a virtual environment, then install the project:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
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

## Report

The LaTeX report draft is in `report/main.tex`. It covers Part I.A.1 and Part
I.A.2, and uses `\IfFileExists` so it compiles before and after generating the
figures.

## Tests

Run the tests with:

```bash
pytest
```
