"""Execute notebook code, including real ALE/video, with small isolated outputs."""
import ast
import argparse
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--live', action='store_true', help='Open the three desktop monitor windows.')
parser.add_argument('--steps', type=int, default=256)
parser.add_argument('--training-only', action='store_true', help='Stop after training and final plots.')
args = parser.parse_args()

import matplotlib
matplotlib.use('Agg')
import cv2
import torch
torch.set_num_threads(2)

nb = json.loads((ROOT / 'atari_breakout_dqn.ipynb').read_text(encoding='utf-8'))
temp_root = ROOT / 'tmp'
temp_root.mkdir(exist_ok=True)
output = Path(tempfile.mkdtemp(prefix='notebook-smoke-', dir=temp_root))
scope = {'__name__': '__main__'}
for index, cell in enumerate(nb['cells']):
    if args.training_only and index > 20:
        break
    if cell['cell_type'] != 'code':
        continue
    source = ''.join(cell['source'])
    ast.parse(source)  # Validate original syntax before smoke-only substitutions.
    source = source.replace('RUN_PROFILE = "paper"', 'RUN_PROFILE = "smoke"')
    source = source.replace('SHOW_LIVE_WINDOWS = True', f'SHOW_LIVE_WINDOWS = {args.live}')
    source = source.replace('TOTAL_STEPS = 10_000_000 if RUN_PROFILE == "paper" else 256',
                            f'TOTAL_STEPS = {args.steps}')
    source = source.replace('OUTPUT_DIR = Path.cwd() / "outputs"', f'OUTPUT_DIR = Path({str(output)!r})')
    print(f'Executing cell {index}', flush=True)
    exec(compile(source, f'notebook-cell-{index}', 'exec'), scope)

assert scope['optimizer_updates'] == max(0, scope['step'] - 31)
assert scope['replay'].size == min(scope['step'], 256)
if scope['step'] > 0:
    assert scope['MODEL_PATH'].is_file()
if scope['optimizer_updates'] > 0:
    assert scope['optimization_steps'][-1] == scope['optimizer_updates']
if not args.training_only:
    assert scope['mp4_files'], 'video was not written'
for video in scope.get('mp4_files', []):
    capture = cv2.VideoCapture(str(video))
    try:
        readable, frame = capture.read()
        assert readable and frame.shape == (210, 160, 3), video
        assert abs(capture.get(cv2.CAP_PROP_FPS) - 60) < 0.01, video
    finally:
        capture.release()
print(f"PASS: {scope['optimizer_updates']} optimizer updates, weights and charts. Outputs: {output}")
