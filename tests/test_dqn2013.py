import unittest
from unittest.mock import patch

import cv2
import gymnasium as gym
import numpy as np
import torch
from torch import nn

from dqn2013 import DQN, PaperObservation, ReplayBuffer, epsilon_by_step
from dqn2013 import make_env, optimize_model, preprocess_frame


class FakeAtari(gym.Env):
    observation_space = gym.spaces.Box(0, 255, (210, 160, 3), np.uint8)
    action_space = gym.spaces.Discrete(4)

    def __init__(self, end_after=100, truncate=False):
        self.end_after, self.truncate = end_after, truncate
        self.calls = 0

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self.calls = 0
        return np.zeros((210, 160, 3), np.uint8), {}

    def step(self, action):
        self.calls += 1
        end = self.calls >= self.end_after
        return (np.full((210, 160, 3), self.calls, np.uint8), 2.0,
                end and not self.truncate, end and self.truncate, {})


class PaperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_exact_network(self):
        net = DQN(4)
        x = torch.zeros(2, 4, 84, 84, dtype=torch.uint8)
        self.assertEqual(net.features[:2](x.float()).shape, (2, 16, 20, 20))
        self.assertEqual(net.features[:4](x.float()).shape, (2, 32, 9, 9))
        self.assertEqual(net(x).shape, (2, 4))
        self.assertEqual(sum(p.numel() for p in net.parameters()), 677172)
        self.assertEqual(sum(isinstance(m, nn.Conv2d) for m in net.modules()), 2)
        self.assertEqual(net.head[0].out_features, 256)
        self.assertIsInstance(net.head[-1], nn.Linear)

    def test_preprocessing_resizes_then_crops(self):
        frame = np.random.default_rng(4).integers(0, 256, (210, 160, 3), dtype=np.uint8)
        result = preprocess_frame(frame)
        expected = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY),
                              (84, 110), interpolation=cv2.INTER_AREA)[18:102]
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(result.dtype, np.uint8)
        direct_resize = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), (84, 84))
        self.assertFalse(np.array_equal(result, direct_resize))

    def test_repeat_stack_and_early_episode_end(self):
        env = PaperObservation(FakeAtari(end_after=6))
        state, _ = env.reset()
        np.testing.assert_array_equal(state, 0)
        state, reward, terminal, truncated, info = env.step(1)
        self.assertEqual(reward, 8)
        self.assertFalse(terminal or truncated)
        self.assertEqual(info['raw_frames'], 4)
        np.testing.assert_array_equal(state[:, 0, 0], [0, 0, 0, 4])
        state, reward, terminal, truncated, info = env.step(2)
        self.assertEqual(reward, 4)
        self.assertTrue(terminal)
        self.assertEqual(info['raw_frames'], 2)
        np.testing.assert_array_equal(state[:, 0, 0], [0, 0, 4, 6])
        state, _ = env.reset()
        np.testing.assert_array_equal(state, 0)

    def test_wrapper_truncation(self):
        env = PaperObservation(FakeAtari(end_after=2, truncate=True))
        env.reset()
        _, reward, terminal, truncated, info = env.step(0)
        self.assertEqual((reward, terminal, truncated, info['raw_frames']), (4, False, True, 2))

    def test_replay_wraparound_sharing_and_reset_boundaries(self):
        replay = ReplayBuffer(3)
        expected = {}
        for episode in range(3):
            state = np.full((4, 84, 84), episode * 50, np.uint8)
            for step in range(3):
                next_state = np.concatenate([state[1:], np.full((1, 84, 84), episode*50+step+1, np.uint8)])
                tag = episode*10+step
                slot = replay.position
                # Second episode ends in truncation, not a true terminal.
                terminal = step == 2 and episode != 1
                replay.add(state, tag, tag, next_state, terminal, episode_end=step == 2)
                expected[slot] = (state.copy(), next_state.copy(), tag, terminal)
                if step == 1:
                    prev = replay.data[(slot-1) % replay.capacity]
                    self.assertIs(prev[3][0], replay.data[slot][0][0])
                state = next_state
            self.assertIsNone(replay._current_frames)
            for slot, (s, ns, tag, terminal) in expected.items():
                with patch('numpy.random.randint', return_value=np.array([slot])):
                    batch = replay.sample(1)
                np.testing.assert_array_equal(batch[0][0].numpy(), s)
                np.testing.assert_array_equal(batch[3][0].numpy(), ns)
                self.assertEqual(batch[1].item(), tag)
                self.assertEqual(batch[4].item(), terminal)
        self.assertEqual(replay.size, 3)

    def test_td_target_terminal_mask_and_detachment(self):
        class ConstantQ(nn.Module):
            def __init__(self):
                super().__init__()
                self.q = nn.Parameter(torch.tensor([2.0, 4.0]))

            def forward(self, x):
                return self.q.expand(len(x), -1)

        class FixedReplay:
            size = 2

            def sample(self, batch_size):
                # First game-over: target 1. Second nonterminal/truncated: 1+.5*4=3.
                return (torch.zeros(2, 1), torch.tensor([0, 0]), torch.ones(2),
                        torch.zeros(2, 1), torch.tensor([1.0, 0.0]))

        net = ConstantQ()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
        metrics = optimize_model(net, optimizer, FixedReplay(), 2, gamma=0.5)
        self.assertAlmostEqual(metrics['loss'], 1.0)
        self.assertAlmostEqual(metrics['target_q_mean'], 2.0)
        # Opposite errors cancel for action 0. Target-only action 1 receives no gradient.
        torch.testing.assert_close(net.q, torch.tensor([2.0, 4.0]))

    def test_epsilon_schedule(self):
        self.assertEqual(epsilon_by_step(0), 1.0)
        self.assertAlmostEqual(epsilon_by_step(500_000), 0.55)
        self.assertAlmostEqual(epsilon_by_step(1_000_000), 0.1)
        self.assertAlmostEqual(epsilon_by_step(10_000_000), 0.1)

    def test_real_breakout_and_rmsprop_update(self):
        env = make_env(seed=7)
        try:
            state, _ = env.reset(seed=7)
            self.assertEqual(env.unwrapped.get_action_meanings(), ['NOOP', 'FIRE', 'RIGHT', 'LEFT'])
            replay = ReplayBuffer(40)
            for step in range(40):
                action = 1 if step == 0 else env.action_space.sample()
                next_state, reward, terminated, truncated, _ = env.step(action)
                replay.add(state, action, np.sign(reward), next_state, terminated, terminated or truncated)
                state = next_state
                if terminated or truncated:
                    state, _ = env.reset()
            net = DQN(4)
            before = [p.detach().clone() for p in net.parameters()]
            optimizer = torch.optim.RMSprop(net.parameters(), lr=2.5e-4, alpha=0.95, eps=0.01)
            metrics = optimize_model(net, optimizer, replay)
            self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, net.parameters())))
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
