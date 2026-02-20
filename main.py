"""
Deep Reinforcement Learning with JAX — DQN on CartPole-v1
=========================================================

Bu örnek şunları içeriyor:
  • Flax (linen) ile Q-Network tanımlama
  • Optax ile adam optimizer
  • Experience Replay Buffer
  • Epsilon-greedy exploration
  • Target network (soft update)
  • JAX jit ile hızlandırılmış train step

Çalıştırmak için:
    uv run python main.py
"""

from __future__ import annotations

import os
import random
from collections import deque
from typing import NamedTuple

# Apple Silicon Metal GPU — jax-metal 0.1.1 + jaxlib 0.4.38

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn


# ─────────────────────────────────────────────
# 1. Hyperparameters
# ─────────────────────────────────────────────
class Config(NamedTuple):
    env_name: str = "CartPole-v1"
    num_episodes: int = 500
    max_steps: int = 500
    gamma: float = 0.99          # discount factor
    lr: float = 1e-3             # learning rate
    batch_size: int = 64
    buffer_size: int = 10_000
    eps_start: float = 1.0       # exploration başlangıç
    eps_end: float = 0.01        # exploration bitiş
    eps_decay: int = 500         # kaç episode'da decay
    target_update_freq: int = 10 # her N episode'da target net güncelle
    hidden_dim: int = 128
    seed: int = 42


# ─────────────────────────────────────────────
# 2. Q-Network (Flax)
# ─────────────────────────────────────────────
class QNetwork(nn.Module):
    """Basit 2 katmanlı MLP — Q(s, a) değerlerini tahmin eder."""

    hidden_dim: int
    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x  # shape: (batch, action_dim)


# ─────────────────────────────────────────────
# 3. Experience Replay Buffer
# ─────────────────────────────────────────────
class Transition(NamedTuple):
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Basit FIFO replay buffer."""

    def __init__(self, capacity: int) -> None:
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> tuple[jnp.ndarray, ...]:
        batch = random.sample(self.buffer, batch_size)
        states = jnp.array(np.array([t.state for t in batch]), dtype=jnp.float32)
        actions = jnp.array([t.action for t in batch], dtype=jnp.int32)  # Metal: int32!
        rewards = jnp.array([t.reward for t in batch], dtype=jnp.float32)
        next_states = jnp.array(np.array([t.next_state for t in batch]), dtype=jnp.float32)
        dones = jnp.array([t.done for t in batch], dtype=jnp.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)


# ─────────────────────────────────────────────
# 4. Training Step (JIT compiled)
# ─────────────────────────────────────────────
def make_train_step(apply_fn, tx, gamma):
    """apply_fn ve tx'i closure ile yakala → JIT sorunsuz çalışır."""

    @jax.jit
    def train_step(params, target_params, opt_state,
                   states, actions, rewards, next_states, dones):
        def loss_fn(params):
            # Mevcut Q değerleri
            q_values = apply_fn(params, states)                         # (B, A)
            q_selected = q_values[jnp.arange(actions.shape[0]), actions]  # (B,)

            # Target Q değerleri (target network ile)
            next_q = apply_fn(target_params, next_states)               # (B, A)
            next_q_max = jnp.max(next_q, axis=-1)                      # (B,)
            target = rewards + gamma * next_q_max * (1.0 - dones)       # (B,)

            # MSE loss
            loss = jnp.mean((q_selected - jax.lax.stop_gradient(target)) ** 2)
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    return train_step


# ─────────────────────────────────────────────
# 5. Epsilon-greedy Action Selection
# ─────────────────────────────────────────────
def select_action(
    params, state: np.ndarray, epsilon: float, apply_fn, num_actions: int
) -> int:
    """Epsilon-greedy aksiyon seçimi."""
    if random.random() < epsilon:
        return random.randint(0, num_actions - 1)
    state_jnp = jnp.array(state[np.newaxis, :], dtype=jnp.float32)  # (1, obs_dim)
    q_values = apply_fn(params, state_jnp)        # (1, A)
    action = int(jnp.argmax(q_values[0]).item())
    return max(0, min(action, num_actions - 1))  # Metal güvenlik kontrolü


# ─────────────────────────────────────────────
# 6. Ana Eğitim Döngüsü
# ─────────────────────────────────────────────
def train() -> None:
    cfg = Config()
    print("=" * 60)
    print(f"  DQN — {cfg.env_name}")
    print(f"  JAX backend : {jax.default_backend()}")
    print(f"  Devices      : {jax.devices()}")
    print("=" * 60)

    # Ortam
    env = gym.make(cfg.env_name)
    obs_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n

    # Model
    q_net = QNetwork(hidden_dim=cfg.hidden_dim, action_dim=num_actions)
    rng = jax.random.key(cfg.seed)
    dummy_obs = jnp.ones((1, obs_dim))
    params = q_net.init(rng, dummy_obs)
    target_params = jax.tree.map(lambda x: x.copy(), params)

    # Optimizer
    tx = optax.adam(cfg.lr)
    opt_state = tx.init(params)

    # JIT-compiled train step (closure ile)
    train_step = make_train_step(q_net.apply, tx, cfg.gamma)

    # Replay buffer
    buffer = ReplayBuffer(cfg.buffer_size)
    random.seed(cfg.seed)

    # Metrikler
    episode_rewards: list[float] = []
    best_avg = -float("inf")

    for episode in range(1, cfg.num_episodes + 1):
        state, _ = env.reset(seed=cfg.seed + episode)
        total_reward = 0.0

        for _ in range(cfg.max_steps):
            # Epsilon decay
            epsilon = cfg.eps_end + (cfg.eps_start - cfg.eps_end) * max(
                0.0, 1.0 - episode / cfg.eps_decay
            )

            action = select_action(params, state, epsilon, q_net.apply, num_actions)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.push(Transition(state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            # Eğitim
            if len(buffer) >= cfg.batch_size:
                s, a, r, ns, d = buffer.sample(cfg.batch_size)
                params, opt_state, loss = train_step(
                    params, target_params, opt_state, s, a, r, ns, d,
                )

            if done:
                break

        episode_rewards.append(total_reward)

        # Target network güncelleme
        if episode % cfg.target_update_freq == 0:
            target_params = jax.tree.map(lambda x: x.copy(), params)

        # Loglama
        if episode % 10 == 0:
            avg_10 = np.mean(episode_rewards[-10:])
            avg_100 = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else avg_10
            if avg_100 > best_avg:
                best_avg = avg_100
            print(
                f"Episode {episode:4d} | "
                f"Reward {total_reward:6.1f} | "
                f"Avg10 {avg_10:6.1f} | "
                f"Avg100 {avg_100:6.1f} | "
                f"Best {best_avg:6.1f} | "
                f"ε {epsilon:.3f}"
            )

        # Çözüldü mü?
        if len(episode_rewards) >= 100 and np.mean(episode_rewards[-100:]) >= 475.0:
            print(f"\n🎉 Ortam çözüldü! Episode {episode} — Avg100 = {np.mean(episode_rewards[-100:]):.1f}")
            break

    env.close()
    print("\nEğitim tamamlandı!")
    print(f"  Son 100 episode ortalaması: {np.mean(episode_rewards[-100:]):.1f}")
    print(f"  En iyi 100-episode ortalaması: {best_avg:.1f}")


if __name__ == "__main__":
    train()
