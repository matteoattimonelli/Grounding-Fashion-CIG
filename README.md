# Grounding Free-Form Instructions for Fashion Complementary Image Generation

Anonymous code repository accompanying the EMNLP 2026 submission
*"Grounding Free-Form Instructions for Fashion Complementary Image Generation"*

This repository contains the three enriched CIG benchmarks
(FashionVC, ExpReduced, FashionTaobao-TB) along with the code for training and evaluating StyleFlow.

## Layout

```
.
├── styleflow/
│   ├── pipeline.py              # StyleFlowPipeline (standalone Diffusers pipeline)
│   ├── data.py                  # FashionDataset (training) and FashionPromptDataset (eval)
│   └── __init__.py
├── train.py                     # LoRA training entry point
├── generate.py                  # Inference / generation entry point
├── eval/
│   ├── fid_kid.py               # FID and KID via torch-fidelity
│   ├── lpips_paired.py          # LPIPS paired against the ground-truth bottom
│   ├── clip_score.py            # CLIP-Score via open_clip ViT-B/32
│   └── siglip_catalog_alignment.py  # MRR / Recall / nDCG @10 and @50 via SigLIP
├── baselines/
│   └── README.md                # how to obtain, adapt, run, and evaluate the four baselines
├── scripts/
│   ├── train_example.sh
│   ├── eval_example.sh
│   └── eval_baselines_example.sh
├── requirements.txt
└── LICENSE
```

## Quick start

### Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Data layout

The three datasets are expected under `${DATASETS_ROOT}` with the
following structure, matching the public releases:

```
${DATASETS_ROOT}/
├── FashionVC/
│   ├── img/                                # *.jpg files
│   ├── files/train_full_columns_dif_G.csv  # train pairs + multi-granularity instructions
│   └── files/test_full_disj.csv            # held-out pairs
├── ExpReduced/
│   └── ...same structure...
└── FashionTaobao-TB/
    └── ...same structure...
```

Each CSV contains the columns `tshirt` (seed item id), `positive_pant`
(target item id), and `bottom_description / detailed / medium / low`
(instructions at four levels of specificity).

### Train

```bash
accelerate launch --num_processes 1 --mixed_precision bf16 train.py \
  --pretrained_model_name_or_path black-forest-labs/FLUX.1-dev \
  --datasets_root "${DATASETS_ROOT}" \
  --output_dir runs/styleflow \
  --train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_train_steps 100000 \
  --learning_rate 2e-4 \
  --rank 16 \
  --mixed_precision bf16 \
  --allow_tf32 \
  --gradient_checkpointing \
  --lr_scheduler cosine --lr_warmup_steps 100 \
  --guidance_scale 3.5
```

`scripts/train_example.sh` packages the same command. The recipe
matches the paper: batch size 1, gradient-accumulation 8, 93,483
optimization steps over the joint dataset, rank-16 LoRA on the MM-DiT
attention and FFN blocks, cosine LR with 100 warm-up steps. The trainer
saves a periodic checkpoint every 1,000 steps and a final
`pytorch_lora_weights.safetensors` to the run root on clean
completion.

### Generate

```bash
# From a directory produced by train.py (looks up pytorch_lora_weights.safetensors)
python generate.py \
  --pretrained_model_name_or_path black-forest-labs/FLUX.1-dev \
  --checkpoint runs/styleflow \
  --dataset fashionvc \
  --datasets_root "${DATASETS_ROOT}" \
  --output_root results/styleflow \
  --prompt_keys detailed,medium,low,dif \
  --batch_size 8 \
  --steps 20 --guidance_scale 3.5

# Or directly from a single .safetensors LoRA file (e.g., the released checkpoint)
python generate.py \
  --pretrained_model_name_or_path black-forest-labs/FLUX.1-dev \
  --checkpoint path/to/styleflow_lora_weights.safetensors \
  --dataset fashionvc \
  --datasets_root "${DATASETS_ROOT}" \
  --output_root results/styleflow \
  --prompt_keys detailed,medium,low,dif \
  --batch_size 8
```

This produces `results/styleflow/FashionVC/test/prompt_<level>/<top_id>_<bottom_id>_template.jpg`
for each `(top, bottom)` test pair.

### Evaluate

```bash
# FID + KID against the bottom catalog
python -m eval.fid_kid --reference_dir "${DATASETS_ROOT}/FashionVC/img" \
                       --generated_dir results/styleflow/FashionVC/test/prompt_detailed

# Paired LPIPS against the ground-truth bottom
python -m eval.lpips_paired --reference_dir "${DATASETS_ROOT}/FashionVC/img" \
                            --generated_dir results/styleflow/FashionVC/test/prompt_detailed

# CLIP-Score
python -m eval.clip_score --datasets_root "${DATASETS_ROOT}" \
                          --dataset fashionvc \
                          --generated_dir results/styleflow/FashionVC/test/prompt_detailed \
                          --prompt_column detailed

# SigLIP catalog alignment (MRR / Recall / nDCG @10, @50)
python -m eval.siglip_catalog_alignment --datasets_root "${DATASETS_ROOT}" \
                                        --dataset fashionvc \
                                        --generated_dir results/styleflow/FashionVC/test/prompt_detailed
```

`scripts/eval_example.sh` runs the full sweep across the three datasets
and four prompt levels.

### Baselines

`baselines/README.md` documents how to obtain, adapt, train, and
evaluate the four baselines reported in the paper (DiFashion, GeCo,
MGCM, Pix2PixCM). Their source code is not vendored here — each
upstream project has its own license — but the README gives the exact
adaptation we applied (free-form instruction wiring) and the recipe
used. `scripts/eval_baselines_example.sh` runs the same four-metric
sweep on baseline outputs once they are placed under
`baselines/baseline_outputs/<BASELINE>/...`.
