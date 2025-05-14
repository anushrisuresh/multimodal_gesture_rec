import argparse
import time

from src.utils      import load_config, LABELS
from src.audio      import get_block
from src.preprocess import block_to_mel
from src.inference  import InferenceEngine

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
    while True:
        block = get_block(sr, dur)
        mel   = block_to_mel(block, sr, fft, hop, n_m, fmax)
        probs = engine.classify(mel)
        idx   = int(probs.argmax())
        conf  = float(probs[idx])
        if conf >= thr:
            print(f"{time.time():.2f}: {LABELS[idx]} ({conf:.2%})")

if __name__ == "__main__":
    main()
