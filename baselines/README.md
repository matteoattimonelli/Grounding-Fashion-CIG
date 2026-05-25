# Baselines

This folder documents how the four baselines reported in the main paper
were trained, run, and evaluated. We do **not** vendor the baselines'
source code here — each upstream project has its own license. Instead,
this README gives, for every baseline, (i) the upstream source, (ii) the
adaptation we applied to consume our free-form instructions, (iii) the
recipe used for our experiments, and (iv) how to feed the baseline
outputs into the evaluation scripts under `../eval/`.

After running any baseline, place its generated images under

```
baseline_outputs/<BaselineName>/<DatasetLabel>/test/prompt_<level>/<top_id>_<bottom_id>_template.jpg
```

so that the same `python -m eval.fid_kid`, `eval.lpips_paired`,
`eval.clip_score`, and `eval.siglip_catalog_alignment` commands documented
in the top-level README evaluate them with the same protocol used for
StyleFlow.

---

## DiFashion (LDM-based, top + template-text)

* Upstream: original DiFashion repository (see paper bibliography for
  citation; we do not reproduce the URL here because the venue's
  anonymity rules disallow links that may identify reviewers' searches).
* License: research-only (consult upstream).

**Adaptation.** DiFashion was originally trained with category-template
prompts (`"a photo of a <type>"`). We feed it our free-form
instructions by replacing the prompt fed to its text encoder with the
appropriate column (`detailed / medium / low` from
`test_full_disj.csv`, or the template prompt for the `dif` column),
keeping all other model hyperparameters at their upstream defaults.

**Recipe.** Adapter LoRA training on the same joint dataset for the
same number of optimisation steps as StyleFlow (93{,}483), bs=1,
gradient-accumulation 8, AdamW lr 2e-4 cosine, 100 warmup. Inference
uses DiFashion's default 50-step schedule (we report wall-clock for
reference in the appendix but do not normalise step counts across
baselines).

**Outputs.** Following its original I/O convention, save generated
images as `<top_id>_<bottom_id>_template.jpg` under
`baseline_outputs/DiFashion/<DatasetLabel>/test/prompt_<level>/`.

---

## GeCo (GAN-based, top → bottom)

* Upstream: original GeCo repository.
* License: research-only.

**Adaptation.** GeCo conditions only on the seed top by default. To
expose the text channel for a fair comparison with our setting, we
extend its generator with a frozen pretrained CLIP text encoder and
inject the pooled text embedding into the latent space via a small
linear projection trained jointly with the rest of the model. All
other architectural choices stay at the upstream defaults.

**Recipe.** Default GeCo training hyperparameters at the
upstream-supported resolutions (128x128) for the same number of epochs
as in the original paper; only the text-injection module is initialised
from scratch.

**Outputs.** Save under
`baseline_outputs/GeCo/<DatasetLabel>/test/prompt_<level>/<top_id>_<bottom_id>_template.jpg`.

---

## MGCM (GAN-based, multi-garment compatibility)

* Upstream: original MGCM repository.
* License: research-only.

**Adaptation.** Same CLIP-text-injection trick as GeCo: we wire a
frozen pretrained CLIP text encoder into the generator. Compatibility
discriminator and StyleGAN-style prior are kept at upstream defaults.
Outputs are saved at the upstream-supported 64x64 resolution and then
upsampled at evaluation time only.

**Recipe.** Upstream defaults; we report the published numbers in the
ablation tables for cross-checking.

**Outputs.** Save under
`baseline_outputs/MGCM/<DatasetLabel>/test/prompt_<level>/<top_id>_<bottom_id>_template.jpg`.

---

## Pix2PixCM (GAN-based, no text)

* Upstream: image-to-image translation variant used in MGCM's paper
  (Pix2Pix with a compatibility head).
* License: research-only.

**Adaptation.** Pix2PixCM is a pure image-to-image baseline and does
not consume text by design. We include it as a text-free reference. It
is therefore evaluated only on FID/KID/LPIPS; CLIP-Score is reported as
`--` in the tables.

**Recipe.** Upstream defaults at the supported 64x64 resolution.

**Outputs.** Save under
`baseline_outputs/Pix2PixCM/<DatasetLabel>/test/prompt_<level>/<top_id>_<bottom_id>_template.jpg`.

---

## Evaluating any baseline with our scripts

Once a baseline's outputs follow the layout above, run the same four
commands as for StyleFlow, just pointing at the new directory:

```bash
# Example for DiFashion / FashionVC / detailed
GEN_DIR=baselines/baseline_outputs/DiFashion/FashionVC/test/prompt_detailed

python -m eval.fid_kid     --reference_dir ${DATASETS_ROOT}/FashionVC/img --generated_dir ${GEN_DIR}
python -m eval.lpips_paired --reference_dir ${DATASETS_ROOT}/FashionVC/img --generated_dir ${GEN_DIR}
python -m eval.clip_score  --datasets_root ${DATASETS_ROOT} --dataset fashionvc \
                            --generated_dir ${GEN_DIR} --prompt_column detailed
python -m eval.siglip_catalog_alignment --datasets_root ${DATASETS_ROOT} --dataset fashionvc \
                                        --generated_dir ${GEN_DIR}
```

`../scripts/eval_baselines_example.sh` runs the full sweep across the
four baselines, three datasets, and four prompt levels, mirroring the
StyleFlow sweep in `../scripts/eval_example.sh`.
