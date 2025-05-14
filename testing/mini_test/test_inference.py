import numpy as np
import pytest

from src.preprocessing import block_to_mel
from src.utils import load_config, LABELS
from src.inference import InferenceEngine

@pytest.fixture
def dummy_audio():
    # 1.92s of silence at 48kHz
    return np.zeros(int(48000 * 1.92), dtype=np.float32)

def test_block_to_mel_shape(dummy_audio):
    mel = block_to_mel(dummy_audio, 48000, 1200, 480, 64, 24000)
    assert mel.shape == (1, 192, 64)

def test_config_load(tmp_path):
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text("foo: bar\n")
    cfg = load_config(str(cfg_file))
    assert cfg['foo'] == 'bar'

def test_inference_interface(tmp_path):
    # Create a tiny dummy TFLite model or skip if not available
    # Here, just instantiate engine and verify methods exist
    engine = InferenceEngine("models/fusion_model.tflite")
    dummy_mel = np.zeros((1,192,64), dtype=np.float32)
    probs = engine.classify(dummy_mel)
    assert isinstance(probs, np.ndarray)
    assert probs.shape[0] == len(LABELS)
