"""
MDP construction layer for the MIMIC Sepsis Offline RL pipeline.

Submodules
----------
features
    Feature dictionary, extraction scaffolding, and preprocessing contracts.
actions
    Treatment encoding: vasopressor standardisation, IV fluid aggregation,
    and the 5×5 discrete action map.
rewards
    Terminal 90-day mortality rewards and configurable intermediate shaping.
reward_models
    Typed data classes for the reward contract (RewardConfig, StepReward).
preprocessing
    Leakage-safe normalisation and clipping artifacts.
"""
