#!/bin/bash
# Submit the full A8 coursework pipeline to Slurm.
# Run from the repository root:
#   bash scripts/submit_full_pipeline.sh

set -euo pipefail

A8_SCRATCH_BASE="${A8_SCRATCH_BASE:-/rds/user/${USER}/hpc-work/a8}"
A8_DATA_DIR="${A8_DATA_DIR:-${A8_SCRATCH_BASE}/data}"
A8_VENV="${A8_VENV:-${A8_SCRATCH_BASE}/venv}"
PYTHON_CMD="${PYTHON_CMD:-python}"
SBATCH_WRAPPER="${SBATCH_WRAPPER:-scripts/sbatch_run.sh}"

B_EPOCHS="${B_EPOCHS:-50}"
B_BATCH_SIZE="${B_BATCH_SIZE:-4}"
B_PATCH_SIZE="${B_PATCH_SIZE:-128}"
B_N_ITER="${B_N_ITER:-6}"
B_FEATURES="${B_FEATURES:-32}"
B_LR="${B_LR:-1e-4}"

YOLO_MODEL="${YOLO_MODEL:-yolo11n-seg.pt}"
YOLO_EPOCHS="${YOLO_EPOCHS:-100}"
YOLO_IMGSZ="${YOLO_IMGSZ:-640}"
YOLO_DEVICE="${YOLO_DEVICE:-0}"
YOLO_CONFIDENCE="${YOLO_CONFIDENCE:-0.25}"

B_TIME="${B_TIME:-02:00:00}"
YOLO_TIME="${YOLO_TIME:-02:00:00}"
SEGMENT_TIME="${SEGMENT_TIME:-01:00:00}"
TRACK_TIME="${TRACK_TIME:-00:30:00}"

submit_gpu() {
  local job_name="$1"
  local time_limit="$2"
  shift 2
  sbatch --parsable     --job-name="${job_name}"     --time="${time_limit}"     --export=ALL,A8_SCRATCH_BASE="${A8_SCRATCH_BASE}",A8_VENV="${A8_VENV}"     "${SBATCH_WRAPPER}" "$@"
}

submit_gpu_dep() {
  local job_name="$1"
  local time_limit="$2"
  local dependency="$3"
  shift 3
  sbatch --parsable     --job-name="${job_name}"     --time="${time_limit}"     --dependency="afterok:${dependency}"     --export=ALL,A8_SCRATCH_BASE="${A8_SCRATCH_BASE}",A8_VENV="${A8_VENV}"     "${SBATCH_WRAPPER}" "$@"
}

echo "Submitting A8 full pipeline"
echo "Scratch: ${A8_SCRATCH_BASE}"
echo "Data:    ${A8_DATA_DIR}"
echo "Venv:    ${A8_VENV}"

export_job=$(submit_gpu a8_export_yolo 00:45:00   "${PYTHON_CMD}" scripts/export_yolo.py   --data-dir "${A8_DATA_DIR}"   --output-dir outputs/part_ii/b/yolo_dataset)
echo "YOLO export job: ${export_job}"

b1_job=$(submit_gpu a8_b1_full "${B_TIME}"   "${PYTHON_CMD}" scripts/run_b1.py   --data-dir "${A8_DATA_DIR}"   --output-dir outputs/part_i/b1_full   --epochs "${B_EPOCHS}"   --batch-size "${B_BATCH_SIZE}"   --patch-size "${B_PATCH_SIZE}"   --n-iter "${B_N_ITER}"   --features "${B_FEATURES}"   --learning-rate "${B_LR}"   --device cuda)
echo "B(i) full job: ${b1_job}"

b2_job=$(submit_gpu_dep a8_b2_full "${B_TIME}" "${b1_job}"   "${PYTHON_CMD}" scripts/run_b2.py   --data-dir "${A8_DATA_DIR}"   --output-dir outputs/part_i/b2_full   --epochs "${B_EPOCHS}"   --batch-size "${B_BATCH_SIZE}"   --patch-size "${B_PATCH_SIZE}"   --n-iter "${B_N_ITER}"   --features "${B_FEATURES}"   --learning-rate "${B_LR}"   --device cuda)
echo "B(ii) full job: ${b2_job}"

b3_job=$(submit_gpu_dep a8_b3_full "${B_TIME}" "${b2_job}"   "${PYTHON_CMD}" scripts/run_b3.py   --data-dir "${A8_DATA_DIR}"   --output-dir outputs/part_i/b3_full   --epochs "${B_EPOCHS}"   --batch-size "${B_BATCH_SIZE}"   --patch-size "${B_PATCH_SIZE}"   --n-iter "${B_N_ITER}"   --features "${B_FEATURES}"   --learning-rate "${B_LR}"   --device cuda)
echo "B(iii) full job: ${b3_job}"

yolo_job=$(submit_gpu_dep a8_yolo_train "${YOLO_TIME}" "${b3_job}"   "${PYTHON_CMD}" scripts/train_yolo.py   --yolo-dataset-dir outputs/part_ii/b/yolo_dataset   --output-dir outputs/part_ii/b/yolo_runs   --model "${YOLO_MODEL}"   --epochs "${YOLO_EPOCHS}"   --imgsz "${YOLO_IMGSZ}"   --device "${YOLO_DEVICE}")
echo "YOLO train job: ${yolo_job}"

segment_job=$(submit_gpu_dep a8_yolo_segment "${SEGMENT_TIME}" "${yolo_job}"   "${PYTHON_CMD}" scripts/run_yolo_sequence.py   --model-path "$(pwd)/outputs/part_ii/b/yolo_runs/cells/weights/best.pt"   --data-dir "${A8_DATA_DIR}"   --sequence 01   --output-dir outputs/part_ii/c/labels_yolo   --confidence "${YOLO_CONFIDENCE}")
echo "YOLO sequence segmentation job: ${segment_job}"

track_job=$(submit_gpu_dep a8_track "${TRACK_TIME}" "${segment_job}"   "${PYTHON_CMD}" scripts/run_tracking.py   --labels-dir outputs/part_ii/c/labels_yolo   --first-frame-path "${A8_DATA_DIR}/PhC-C2DH-U373/01/t000.tif"   --output-dir outputs/part_ii/c_yolo)
echo "Tracking job: ${track_job}"

echo "Submitted jobs:"
echo "  export=${export_job}"
echo "  b1=${b1_job}"
echo "  b2=${b2_job}"
echo "  b3=${b3_job}"
echo "  yolo=${yolo_job}"
echo "  segment=${segment_job}"
echo "  track=${track_job}"
echo "Use: squeue -u ${USER}"
