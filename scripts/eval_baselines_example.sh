#!/usr/bin/env bash
# Run the same four-metric sweep used by StyleFlow on a baseline's
# already-generated images. Assumes the baseline outputs are at
#
#   baselines/baseline_outputs/<BASELINE>/<DatasetLabel>/test/prompt_<key>/
#       <top_id>_<bottom_id>_template.jpg
#
# (see baselines/README.md for the conventions used when generating
# each baseline).

set -euo pipefail

: "${DATASETS_ROOT:=./data}"
: "${BASELINE_OUTPUTS_ROOT:=baselines/baseline_outputs}"
: "${METRICS_ROOT:=results/baselines_metrics}"
: "${BASELINES:=DiFashion GeCo MGCM_text Pix2PixCM custom_gan_text}"
: "${PROMPT_KEYS:=detailed medium low dif}"

mkdir -p "${METRICS_ROOT}"

for BASELINE in ${BASELINES}; do
  for entry in fashionvc:FashionVC expreduced:ExpReduced fashiontaobaoTB:FashionTaobao-TB; do
    IFS=':' read -r DSET LABEL <<< "$entry"
    for KEY in ${PROMPT_KEYS}; do
      GEN_DIR="${BASELINE_OUTPUTS_ROOT}/${BASELINE}/${LABEL}/test/prompt_${KEY}"
      if [ ! -d "${GEN_DIR}" ]; then
        echo "[skip] ${GEN_DIR} not found (baseline ${BASELINE} not run for this cell)"
        continue
      fi
      OUT_DIR="${METRICS_ROOT}/${BASELINE}/${DSET}/${KEY}"
      mkdir -p "${OUT_DIR}"
      echo "=== ${BASELINE} / ${DSET} / prompt_${KEY} ==="

      python -m eval.fid_kid \
        --reference_dir "${DATASETS_ROOT}/${LABEL}/img" \
        --generated_dir "${GEN_DIR}" \
        --output_json "${OUT_DIR}/fid_kid.json"

      python -m eval.lpips_paired \
        --reference_dir "${DATASETS_ROOT}/${LABEL}/img" \
        --generated_dir "${GEN_DIR}" \
        --output_json "${OUT_DIR}/lpips.json"

      # Pix2PixCM does not consume text; skip CLIP-Score for it.
      if [ "${BASELINE}" != "Pix2PixCM" ]; then
        python -m eval.clip_score \
          --datasets_root "${DATASETS_ROOT}" \
          --dataset "${DSET}" \
          --generated_dir "${GEN_DIR}" \
          --prompt_column "${KEY}" \
          --output_json "${OUT_DIR}/clip_score.json"
      fi

      python -m eval.siglip_catalog_alignment \
        --datasets_root "${DATASETS_ROOT}" \
        --dataset "${DSET}" \
        --generated_dir "${GEN_DIR}" \
        --output_json "${OUT_DIR}/siglip_alignment.json"
    done
  done
done

echo "Done. Baseline metrics under ${METRICS_ROOT}/"
