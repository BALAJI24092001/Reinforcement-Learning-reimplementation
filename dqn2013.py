"""Breakout components following Mnih et al., arXiv:1312.5602v1 (2013).

See README.md for paper references and choices not specified in the paper.
"""
from collections import deque

import ale_py
import cv2
import gymnasium as gym
import numpy as np
import torch
from torch import nn


def preprocess_frame(frame, crop_top=18):
    """RGB -> grayscale -> 110 high x 84 wide -> 84 x 84 playing area."""
    if not 0 <= crop_top <= 26:
        raise ValueError("crop_top must lie between 0 and 26")
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 110), interpolation=cv2.INTER_AREA)
    return resized[crop_top:crop_top + 84, :].copy()


class PaperObservation(gym.Wrapper):
    """Repeat actions four raw frames; stack four sampled, preprocessed frames.

    No max pooling, random reset no-ops, automatic FIRE, or life-loss terminals.
    The policy must choose FIRE to serve, just as it chooses paddle movements.
    """
    def __init__(self, env, frame_skip=4, crop_top=18):
        super().__init__(env)
        if frame_skip < 1:
            raise ValueError("frame_skip must be positive")
        self.frame_skip = frame_skip
        self.crop_top = crop_top
        self.frames = deque(maxlen=4)
        self.observation_space = gym.spaces.Box(0, 255, (4, 84, 84), np.uint8)

    def reset(self, **kwargs):
        frame, info = self.env.reset(**kwargs)
        processed = preprocess_frame(frame, self.crop_top)
        self.frames.clear()
        self.frames.extend([processed] * 4)
        return np.stack(self.frames), info

    def step(self, action):
        reward_sum = 0.0
        for count in range(1, self.frame_skip + 1):
            frame, reward, terminated, truncated, info = self.env.step(action)
            reward_sum += float(reward)
            if terminated or truncated:
                break
        self.frames.append(preprocess_frame(frame, self.crop_top))
        info = dict(info, raw_frames=count)
        return np.stack(self.frames), reward_sum, terminated, truncated, info


def make_env(render_mode=None, video_folder=None, seed=None):
    gym.register_envs(ale_py)
    base = gym.make(
        "ALE/Breakout-v5", frameskip=1, repeat_action_probability=0.0,
        full_action_space=False, render_mode=render_mode,
    )
    if video_folder is not None:
        # Record inside action repeat so the video preserves the emulator frame rate.
        base = gym.wrappers.RecordVideo(
            base, video_folder=str(video_folder),
            episode_trigger=lambda episode_id: True, name_prefix="breakout-2013",
            fps=60,  # Raw Atari frames; ALE's render metadata can default to 30.
        )
    env = PaperObservation(base)
    env.action_space.seed(seed)
    return env


class DQN(nn.Module):
    """Section 4.1: 16/32 convolution filters, 256 dense units, linear Q outputs."""
    def __init__(self, n_actions):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2), nn.ReLU(),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(32 * 9 * 9, 256), nn.ReLU(), nn.Linear(256, n_actions),
        )

    def forward(self, x):
        return self.head(self.features(x.float() / 255.0))


class ReplayBuffer:
    """Uniform circular replay with shared uint8 frames, allocated as collected.

    Call add sequentially for a single environment. Mark *every* reset boundary
    with episode_end, including truncations. Adjacent stacks share frame arrays;
    eviction never invalidates frames still referenced by another transition.
    """
    def __init__(self, capacity, obs_shape=(4, 84, 84), device="cpu"):
        if capacity < 1 or tuple(obs_shape) != (4, 84, 84):
            raise ValueError("positive capacity and (4, 84, 84) observations required")
        self.capacity = capacity
        self.device = device
        self.data = [None] * capacity
        self.position = self.size = 0
        self._current_frames = None

    def add(self, state, action, reward, next_state, terminated, episode_end=None):
        if episode_end is None:
            episode_end = terminated
        if self._current_frames is None:
            self._current_frames = tuple(np.array(f, dtype=np.uint8, copy=True) for f in state)
        next_frames = self._current_frames[1:] + (np.array(next_state[-1], dtype=np.uint8, copy=True),)
        self.data[self.position] = (
            self._current_frames, int(action), float(reward), next_frames, bool(terminated),
        )
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self._current_frames = None if episode_end else next_frames

    def sample(self, batch_size):
        if self.size == 0:
            raise ValueError("cannot sample empty replay")
        indices = np.random.randint(self.size, size=batch_size)
        states, actions, rewards, next_states, terminals = zip(*(self.data[i] for i in indices))
        return (
            torch.as_tensor(np.array(states), device=self.device),
            torch.as_tensor(actions, dtype=torch.long, device=self.device),
            torch.as_tensor(rewards, dtype=torch.float32, device=self.device),
            torch.as_tensor(np.array(next_states), device=self.device),
            torch.as_tensor(terminals, dtype=torch.float32, device=self.device),
        )


def epsilon_by_step(step, decay_steps=1_000_000, start=1.0, end=0.1):
    fraction = min(max(step / decay_steps, 0.0), 1.0)
    return start + fraction * (end - start)


def optimize_model(network, optimizer, replay, batch_size=32, gamma=0.99):
    if replay.size < batch_size:
        return None
    states, actions, rewards, next_states, terminals = replay.sample(batch_size)
    predicted = network(states).gather(1, actions[:, None]).squeeze(1)
    # Algorithm 1 uses the current Q parameters, held fixed for this update.
    with torch.no_grad():
        targets = rewards + gamma * (1.0 - terminals) * network(next_states).max(1).values
    loss = nn.functional.mse_loss(predicted, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    # Measure, but do not clip: neither Huber loss nor gradient clipping is specified.
    norm = torch.stack([p.grad.detach().square().sum() for p in network.parameters()
                        if p.grad is not None]).sum().sqrt()
    optimizer.step()
    return {
        "loss": loss.item(), "q_mean": predicted.detach().mean().item(),
        "target_q_mean": targets.mean().item(),
        "td_error": (targets - predicted.detach()).abs().mean().item(),
        "gradient_norm": norm.item(),
        "reward_rate": 100 * (rewards != 0).float().mean().item(),
    }
