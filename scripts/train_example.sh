#!/usr/bin/env bash
# Train StyleFlow with the paper's published recipe on a single GPU.
#
# Set DATASETS_ROOT to the parent directory of FashionVC/, ExpReduced/,
# and FashionTaobao-TB/ (see README for the expected layout).

set -euo pipefail

: "${DATASETS_ROOT:?Set DATASETS_ROOT to the path of the three CIG datasets}"
: "${MODEL_ID:=black-forest-labs/FLUX.1-dev}"
: "${OUTPUT_DIR:=runs/styleflow}"

accelerate launch \
  --num_processes 1 \
  --mixed_precision bf16 \
  train.py \
    --pretrained_model_name_or_path "${MODEL_ID}" \
    --datasets_root "${DATASETS_ROOT}" \
    --output_dir "${OUTPUT_DIR}" \
    --datasets fashionvc,expreduced,fashiontaobaotb \
    --train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_train_steps 93483 \
    --learning_rate 2e-4 \
    --rank 16 \
    --mixed_precision bf16 \
    --allow_tf32 \
    --gradient_checkpointing \
    --lr_scheduler cosine --lr_warmup_steps 100 \
    --guidance_scale 3.5 \
    --validation_steps 1000 --checkpointing_steps 1000 \
    --seed 0 \
    --report_to tensorboard
