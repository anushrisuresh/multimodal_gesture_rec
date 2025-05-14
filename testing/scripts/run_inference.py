import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import argparse
import time
import numpy as np

from src.utils import load_config, LABELS
from src.audio import get_block
from src.preprocessing import block_to_mel
from src.inference import InferenceEngine

def main():
    parser = argparse.ArgumentParser(description="Run real-time gesture inference")
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config.yaml')
    args = parser.parse_args()

    cfg = load_config(args.config)
    engine = InferenceEngine(cfg['model_path'])

    sr   = cfg['sample_rate']
    dur  = cfg['block_duration']
    fft  = cfg['n_fft']
    hop  = cfg['hop_length']
    n_m  = cfg['n_mels']
    fmax = cfg['fmax']
    thr  = cfg.get('threshold', 0.5)

    print("Starting inference loop…")
    last_probs = None
    while True:
        block = get_block(sr, dur)
        mel   = block_to_mel(block, sr, fft, hop, n_m, fmax)
        probs = engine.classify(mel)
        print(", ".join(f"{lab}:{p:.3f}" for lab, p in zip(LABELS, probs)))
        if last_probs is not None and np.allclose(probs, last_probs):
            print("Warning: probabilities unchanged between iterations. Check ultrasound input.")
        last_probs = probs.copy()
        idx   = int(probs.argmax())
        conf  = float(probs[idx])
        if conf >= thr:
            print(f"{time.time():.2f}: {LABELS[idx]} ({conf:.2%})")
        else:
            print("No class gets above the threshold")

if __name__ == "__main__":
    main()
