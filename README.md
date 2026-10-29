# Atari Breakout with the 2013 Deep Q-Network

Implementation of Mnih et al., **Playing Atari with Deep Reinforcement Learning**,
arXiv:1312.5602v1 (19 December 2013), using the supplied nine-page PDF. The notebook
runs real Atari Breakout in the Arcade Learning Environment (ALE).

## Run

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Open `atari_breakout_dqn.ipynb` in VS Code, select this environment as its kernel,
and run cells in order. Keep the notebook working directory at the repository root.
The notebook imports the testable components in `dqn2013.py`.

- Use `RUN_PROFILE = "smoke"` first: 256 decisions, 256 replay slots, the same network
  and learning rule. This checks execution; it does not train a competent player.
- The default `"paper"` profile configures 10 million decisions and 1 million replay
  entries. Full replay needs roughly 7+ GiB, plus model, optimizer and batch memory.
  Training is a substantial compute job; CUDA is selected when available.
- `SHOW_LIVE_WINDOWS = True` is the default. Training opens **three** resizable
  desktop windows: gameplay, episode performance (score/rolling mean and episode
  length), and learning progress (squared TD loss and mean predicted Q).
  Counters appear in separate headers above content, never over the game or plots.
  Game refresh is capped at 15 FPS, plots refresh once per second, and window events
  are processed every decision. Q/Esc or closing any monitor window stops and saves.
  Kernel interruption also saves and closes windows. Use `False` for headless runs.
  This setting is defined at the top of the **Train and monitor** cell itself, so
  rerunning it cannot inherit a stale value from an earlier configuration cell.
  The cell reloads the monitor module, prints the active Python and module paths,
  and waits for all three windows to report visibility before training starts.
  Startup failures and GUI errors are reported explicitly instead of silently
  closing the windows as if the user had requested a stop.
- Training writes `outputs/breakout_dqn2013_final.pt` and periodic weight snapshots
  to `outputs/breakout_dqn2013_checkpoint.pt`. These are weights, not resumable
  training snapshots (optimizer, replay and random-generator state are not saved).
- Evaluation and video cells use epsilon 0.05. Recordings go into new directories
  under `outputs/breakout_video_2013/`.

Old `breakout_dqn_*` weights have a different architecture and cannot be loaded into
the new network. Previous output files are preserved; stale notebook cell outputs
have been cleared. Train the new model before judging its performance.

## Paper-to-code mapping

| Paper location | Specification | Implementation |
| --- | --- | --- |
| Section 4.1, p. 5 | Grayscale; resize to 110 x 84; crop to 84 x 84; stack 4 frames | `preprocess_frame`, `PaperObservation`; tensor order `(B, 4, 84, 84)` |
| Section 4.1, p. 6 | 16 filters of 8 x 8, stride 4, ReLU | `DQN.features[0:2]`, output `16 x 20 x 20` |
| Section 4.1, p. 6 | 32 filters of 4 x 4, stride 2, ReLU | `DQN.features[2:4]`, output `32 x 9 x 9` |
| Section 4.1, p. 6 | 256 fully connected ReLU units | Flatten 2592 features, `DQN.head[0:2]` |
| Section 4.1, p. 6 | Linear output for every valid action | Four outputs: NOOP, FIRE, RIGHT, LEFT; no softmax |
| Algorithm 1, p. 5; Equation 2, p. 3 | Uniform replay, squared TD error, one Q-network | `ReplayBuffer`, `optimize_model`; detach next-state targets before updating the same network |
| Section 5, p. 6 | RMSProp, minibatch 32 | `torch.optim.RMSprop`, `BATCH_SIZE = 32` |
| Section 5, p. 6 | Positive/negative training rewards become +1/-1 | `np.sign(reward)` when adding to replay; scores keep raw rewards |
| Section 5, p. 6 | Epsilon 1 to 0.1 over first million frames | Linear schedule, then fixed at 0.1 |
| Section 5, p. 6 | 10 million training frames, million-frame replay | 10 million decision budget, million transition slots; convention explained below |
| Section 5, p. 6 | Repeat action for k=4 in Breakout | Base ALE frameskip 1, wrapper performs four steps, stops early on episode end |
| Figure 2 and Section 5.3, p. 7 | Evaluation epsilon 0.05, fixed step budget | 10,000 decisions; original rewards; incomplete game reported separately |

The Breakout network contains **677,172 parameters**. The old notebook used a
three-convolution, 512-unit network, Adam, Huber loss, gradient clipping, and a target
network copied every 5,000 decisions. Those are not the architecture and Algorithm 1
in this supplied 2013 paper, so they have been replaced. There is a detached TD
*target value*, but no separate target *network*.

## Explicit implementation choices and reproduction limits

The PDF does not specify enough details to reproduce the historical experiment
bit for bit. The network layout matches it; the following choices remain explicit:

- **Preprocessing:** OpenCV RGB-to-grayscale and area interpolation; crop resized
  rows `[18:102]`, retaining the Breakout ball, bricks and paddle. No two-frame max
  pooling. Initial stacks repeat the reset image. Input intensities are divided by
  255. The paper does not specify crop coordinates, interpolation, stack padding or
  intensity scaling. Padding in convolutions is zero pixels (valid convolutions).
- **Optimizer:** learning rate `0.00025`, RMSProp alpha `0.95`, epsilon `0.01`, zero
  momentum, uncentered statistics; PyTorch default parameter initialization. These
  numerical choices and the discount `0.99` are not stated in the supplied PDF.
  No Huber loss or gradient clipping is added.
- **Timing:** one update per agent decision, beginning once 32 transitions exist.
  This short batch-fill delay is an implementation choice. The PDF's use of
  "frames" does not explicitly disambiguate raw emulator frames from observed
  frames after action repeat. Here training/epsilon budgets count decisions, giving
  approximately 40 million raw frames in the full run. For the literal raw-frame
  interpretation, divide those two budgets by four (2.5 million / 250,000 decisions).
  Replay capacity counts transitions, sharing overlapping frames; reset stacks add
  a small amount of extra image storage.
- **Environment:** modern ALE/Gymnasium, minimal action set, sticky actions disabled,
  no random reset no-ops, no forced FIRE, and no artificial life-loss terminals or
  penalties. The agent chooses FIRE to start and re-serve. A time-limit truncation
  resets the stack but does not suppress the Bellman bootstrap; game-over does.
  The installed ALE version and its built-in time limit differ from historical ALE.
- **Evaluation:** standalone evaluation rather than automated periodic evaluation.
  Partial or time-limited games are reported separately from completed-game means.
  The diagnostic Q curves use training minibatches, not the fixed held-out-state
  maximum-Q curves described in Section 5.1. No paper-level score is claimed.

The paper's contributions are learning action values directly from pixels and using
uniform experience replay to reduce correlations and reuse experience. Its results
span seven games with shared settings (Space Invaders alone uses k=3). Table 1
reports a Breakout mean of 168 and best episode of 225; reproducing those scores
requires an actual training/evaluation experiment, not just matching layer sizes.

Modern emulator behavior and recording APIs were checked against the
[ALE environment documentation](https://ale.farama.org/environments/).

## Validation

`tests/test_dqn2013.py` checks exact layer geometry and parameter count, preprocessing
order, frame repeat and early termination, replay wraparound and reset boundaries,
terminal/truncation target semantics, optimizer updates, and a real ALE interaction.
`tests/test_training_monitor.py` checks separate status headers, chart data and
rolling means, empty chart states, and window stop/cleanup behavior.
`scripts/smoke_notebook.py` executes every notebook code cell with the smoke profile
using a noninteractive plotting backend, including training, final charts, evaluation
and video. By default it disables desktop windows for automated execution. To test
the native popups during real training, add `--live --training-only --steps 3000`.

```powershell
.\.venv\Scripts\python.exe scripts\smoke_notebook.py
```

The smoke runner writes into a fresh temporary directory under `tmp/` so existing
training artifacts are not overwritten. It validates execution, not convergence.
