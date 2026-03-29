"""
Baseline package for the MIMIC Sepsis Offline RL pipeline.

Provides comparison baselines that all RL algorithms must beat or
contextualize:

- **clinician**: Replays the clinician's observed actions.
- **no_treatment**: Static policy that always selects action 0 (no treatment).
- **behavior_cloning**: Supervised learning of the clinician's policy.
"""
