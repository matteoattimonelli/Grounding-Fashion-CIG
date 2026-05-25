"""StyleFlow: instruction-driven complementary garment generation on FLUX.

This package exposes a single self-contained inference pipeline,
``StyleFlowPipeline``, and dataset utilities. ``StyleFlowPipeline``
intentionally does NOT inherit from ``FluxControlPipeline`` or any
other FLUX pipeline class; it is built directly on top of
``diffusers.DiffusionPipeline`` and bundles all helpers it needs.
"""

from .pipeline import StyleFlowPipeline, expand_x_embedder_for_seed_concat

__all__ = ["StyleFlowPipeline", "expand_x_embedder_for_seed_concat"]
