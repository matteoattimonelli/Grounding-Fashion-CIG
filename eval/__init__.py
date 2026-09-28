"""Evaluation modules for StyleFlow generations.

Each module is runnable as ``python -m eval.<module>`` and takes a
generated-image directory plus the dataset path. Modules:

* ``fid_kid``                 — FID and KID via torch-fidelity
* ``lpips_paired``            — LPIPS paired against the ground-truth bottom
* ``clip_score``              — open_clip ViT-B/32 cosine similarity
* ``siglip_catalog_alignment`` — MRR / Recall / nDCG @10 and @50 via SigLIP
"""
