"""
Feature dictionary and extraction scaffolding for Phase 4 state representation.

Public API
----------
FeatureSpec
    Typed descriptor for a single state feature (source, units, aggregation, ranges).
AggregationRule
    Enum of supported within-window aggregation strategies.
MissingStrategy
    Enum of imputation fallback strategies.
FEATURE_REGISTRY
    Ordered dict of all v1 feature specs keyed by feature identifier.
load_feature_registry
    Load and optionally filter the registry from a config mapping.
"""

from mimic_sepsis_rl.mdp.features.dictionary import (
    FEATURE_REGISTRY,
    FEATURE_SPEC_VERSION,
    AggregationRule,
    FeatureFamily,
    FeatureSpec,
    MissingStrategy,
    load_feature_registry,
)

__all__ = [
    "AggregationRule",
    "FeatureFamily",
    "FeatureSpec",
    "MissingStrategy",
    "FEATURE_REGISTRY",
    "FEATURE_SPEC_VERSION",
    "load_feature_registry",
]
