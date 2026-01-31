"""Evaluation utilities for SSL attention analysis.

This module provides tools for validating SSL model features before
attention analysis:

- Linear Probe: Validates that frozen features are discriminative for
  architectural style classification

Example:
    >>> from ssl_attention.evaluation import train_linear_probe_sklearn
    >>> from ssl_attention.config import STYLE_NAMES
    >>>
    >>> # features: (N, D) tensor, labels: (N,) tensor
    >>> result = train_linear_probe_sklearn(features, labels, class_names=STYLE_NAMES)
    >>> print(f"CV Accuracy: {result.cv_mean:.1%} ± {result.cv_std:.1%}")
    >>>
    >>> # Expected: >50% accuracy (random = 25%)
    >>> # If accuracy is near random, attention analysis may not be meaningful
"""

from ssl_attention.evaluation.linear_probe import (
    ProbeResult,
    compare_model_features,
    extract_features_for_probe,
    print_probe_summary,
    train_linear_probe_sklearn,
)

__all__ = [
    "ProbeResult",
    "train_linear_probe_sklearn",
    "extract_features_for_probe",
    "compare_model_features",
    "print_probe_summary",
]
