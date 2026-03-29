"""
Dataset layer for the MIMIC Sepsis Offline RL pipeline.

Submodules
----------
transitions
    Build replay-ready (s_t, a_t, r_t, s_{t+1}, done) tuples from the
    frozen state, action, and reward surfaces.
replay_buffer
    Episode-aware serialisation of transition datasets for offline RL
    trainers.
"""
