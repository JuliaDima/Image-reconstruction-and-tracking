# Reproducible CPU environment for the image-analysis coursework.
# Build:  docker build -t image-analysis .
# Test:   docker run --rm image-analysis pytest -q
# Run:    docker run --rm -v "$PWD/outputs:/app/outputs" image-analysis \
#             python scripts/run_a2.py
FROM python:3.11-slim

# System libraries required by scikit-image / imagecodecs / matplotlib I/O.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY requirements.txt requirements-ml.txt pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --upgrade pip && python -m pip install -r requirements.txt

COPY . .

# Default command runs the unit test suite; override on the command line to run
# any of the scripts in scripts/.
CMD ["pytest", "-q"]
