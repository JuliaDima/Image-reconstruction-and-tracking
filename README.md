# Image Analysis Coursework

This repository contains the coursework implementation for the MPhil in Data
Intensive Science image analysis assignment.

The current implementation covers Part I.A.1: simulating the forward image
acquisition process by blurring a microscopy image with a Gaussian point-spread
function and adding Gaussian noise.

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

## Report

The LaTeX report draft is in `report/main.tex`. It is focused on Part I.A.1 and
uses `\IfFileExists` so it compiles before and after generating the figure.

## Tests

Run the tests with:

```bash
pytest
```
