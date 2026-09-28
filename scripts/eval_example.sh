#!/usr/bin/env bash
# Generate StyleFlow outputs on all three datasets for the four
# prompt-specificity levels, then compute FID/KID, paired LPIPS,
# CLIP-Score, and SigLIP catalog-alignment metrics.

set -euo pipefail

: "${DATASETS_ROOT:=./data}"
: "${MODEL_ID:=black-forest-labs/FLUX.1-dev}"
: "${CKPT_DIR:=runs/styleflow}"
: "${OUTPUT_ROOT:=results/styleflow}"
: "${METRICS_ROOT:=results/styleflow_metrics}"
: "${PROMPT_KEYS:=detailed,medium,low,dif}"

mkdir -p "${OUTPUT_ROOT}" "${METRICS_ROOT}"

for entry in fashionvc:8:FashionVC expreduced:4:ExpReduced fashiontaobaoTB:2:FashionTaobao-TB; do
  IFS=':' read -r DSET BS LABEL <<< "$entry"
  echo "=== Generating ${DSET} (bs=${BS}) ==="
  python generate.py \
    --pretrained_model_name_or_path "${MODEL_ID}" \
    --checkpoint_dir "${CKPT_DIR}" \
    --dataset "${DSET}" \
    --datasets_root "${DATASETS_ROOT}" \
    --output_root "${OUTPUT_ROOT}" \
    --prompt_keys "${PROMPT_KEYS}" \
    --batch_size "${BS}" \
    --steps 20 --guidance_scale 3.5 --seed 0

  for KEY in $(echo "${PROMPT_KEYS}" | tr ',' ' '); do
    GEN_DIR="${OUTPUT_ROOT}/${LABEL}/test/prompt_${KEY}"
    OUT_DIR="${METRICS_ROOT}/${DSET}/${KEY}"
    mkdir -p "${OUT_DIR}"
    echo "  -> ${DSET}/prompt_${KEY}"

    python -m eval.fid_kid \
      --reference_dir "${DATASETS_ROOT}/${LABEL}/img" \
      --generated_dir "${GEN_DIR}" \
      --output_json "${OUT_DIR}/fid_kid.json"

    python -m eval.lpips_paired \
      --reference_dir "${DATASETS_ROOT}/${LABEL}/img" \
      --generated_dir "${GEN_DIR}" \
      --output_json "${OUT_DIR}/lpips.json"

    python -m eval.clip_score \
      --datasets_root "${DATASETS_ROOT}" \
      --dataset "${DSET}" \
      --generated_dir "${GEN_DIR}" \
      --prompt_column "${KEY}" \
      --output_json "${OUT_DIR}/clip_score.json"

    python -m eval.siglip_catalog_alignment \
      --datasets_root "${DATASETS_ROOT}" \
      --dataset "${DSET}" \
      --generated_dir "${GEN_DIR}" \
      --output_json "${OUT_DIR}/siglip_alignment.json"
  done
done

echo "Done. Metrics under ${METRICS_ROOT}/"
