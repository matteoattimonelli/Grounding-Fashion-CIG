# Grounding Free-Form Instructions for Fashion Complementary Image Generation

Anonymous code repository accompanying the paper
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
├── data/
│   ├── FashionVC/files/{train.csv, test.csv}
│   ├── ExpReduced/files/{train.csv, test.csv}
│   └── FashionTaobao-TB/files/{train.csv, test.csv}
└── requirements.txt
```

## Quick start

### Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Data layout

The three enriched CIG benchmarks (FashionVC, ExpReduced,
FashionTaobao-TB) are shipped with this repository under `data/`. The
CSV files (multi-granularity free-form instructions) are included
directly; the corresponding image folders are obtained from the
original dataset releases as described below (Datasets Download).

```
${DATASETS_ROOT}/                          # set DATASETS_ROOT=./data after running the image downloads
├── FashionVC/
│   ├── img/                                # *.jpg files (downloaded from upstream — see below)
│   └── files/
│       ├── train.csv                       # train pairs + 5 instruction columns (detailed/medium/low/empty/dif)
│       └── test.csv                        # held-out pairs + 4 instruction columns + Type_New for the template
├── ExpReduced/
│   └── ...same structure...
└── FashionTaobao-TB/
    └── ...same structure...
```

`train.csv` schema: `tshirt, positive_pant, detailed, medium, low,
empty, dif`. One row per (top, bottom) pair, with one column per
instruction-specificity level.

`test.csv` schema: `tshirt, positive_pant, detailed, medium, low,
empty, Type_New`. The DiFashion-style template prompt (the `dif`
level) is rendered on the fly from `Type_New` at inference and
evaluation time, so the file does not need to store it explicitly.

**Training-time CSV explosion.** At training time, `FashionDataset`
*explodes* `train.csv` into a long table with one row per
(top, bottom, prompt-level) triple, so a single epoch traverses every
pair under every requested instruction level exactly once (5x by
default). `__getitem__` then reads the per-row prompt directly — no
stochastic in-dataloader prompt sampling. Pass
`--prompt_levels detailed,medium,low,empty,dif` (default) to control
which subset of levels is exposed.

By default all paths point at `./data` (i.e., the bundled folder);
override with `--datasets_root ${DATASETS_ROOT}` if you keep the
datasets elsewhere.

# Datasets Download

Due to a size limit, only the train and test splits of the dataset have been uploaded at this time.
The validation splits will be shared once possible.


The dataset files (in CSV format) are provided in the ```datasets``` directory. The structure of the folder is as follows:
<!-- ```
├── datasets/
│   ├── FashionVC/
│   │   ├── files/      
│   │   ├── img/        
│   ├── ExpReduced/
│   │   ├── files/      
│   │   ├── img/        
│   ├── FashionTaobaoTB/
│   │   ├── files/      
│   │   ├── img/ 
``` -->
```
datasets/
├── FashionVC/
│   ├── files/
│   ├── img/
├── ExpReduced/
│   ├── files/
│   ├── img/
├── FashionTaobaoTB/
│   ├── files/
│   ├── img/

```

## FashionVC and ExpReduced

To download the images for the `FashionVC` and `ExpReduced` datasets, follow these steps:

1. Clone the dataset repository:
```sh
git clone https://bitbucket.org/Jay_Ren/fashion_recommendation_tkde2018_code_dataset.git
```
2. Extract the image files:
```sh
unzip ./fashion_recommendation_tkde2018_code_dataset/img.zip -d ./datasets/ExpReduced
unzip ./fashion_recommendation_tkde2018_code_dataset/FashionVC/img.zip -d ./datasets/FashionVC
```

## FashionTaobaoTB

To download the images for FashionTaobaoTB, you need to have `Node.js` installed. Please follow these steps:

1. Navigate to the `datasets/FashionTaobaoTB` folder.
2. Initialize a new Node.js project and install required dependencies:
```sh
npm init -y
npm install superagent csv-parser cli-progress
npm install -g typescript
npm install -g tsc
```
3. Compile and run the TypeScript script to download the images:
```sh
tsc index.ts
node index.js
```

If you encounter any errors related to missing types, run the following command to install the necessary Node.js types:
```sh
npm i --save-dev @types/node
```

After the image download finishes, you need to preprocess the images. Please run the following script:
```sh
python3 preprocess_imgs.py
```

**Note**: Due to restrictions, we are unable to share the images directly. If any link is no longer valid, please remove the corresponding entry for the missing image from the CSV files.

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

`baselines/README.md` documents how to train and evaluate every baseline
shipped in this repository (DiFashion, GeCo, MGCM_text, Pix2PixCM,
custom_gan_text), with per-baseline training commands and the exact
adaptation used to consume our free-form instructions.
`scripts/eval_baselines_example.sh` runs the same four-metric sweep on
baseline outputs once they are placed under
`baselines/baseline_outputs/<BASELINE>/...`.
