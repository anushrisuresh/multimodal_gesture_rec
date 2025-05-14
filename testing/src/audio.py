import sounddevice as sd
import numpy as np
_last_input_device = None

def get_block(sample_rate: int, block_duration: float) -> np.ndarray:
    """
    Record `block_duration` seconds from the default mic, normalize to [-1,1].
    Returns:
        block: float32 array of shape (samples,)
    """
    global _last_input_device
    dev = sd.default.device[0]
    if dev != _last_input_device:
        info = sd.query_devices(dev)
        print(f"Using input device: {info['name']} (index {dev})")
        _last_input_device = dev
    try:
        sd.check_input_settings(device=sd.default.device[0], channels=1, samplerate=sample_rate)
        num_samples = int(sample_rate * block_duration)
        recording = sd.rec(num_samples, samplerate=sample_rate, channels=1, dtype='int16')
        sd.wait()
        return recording.flatten().astype(np.float32) / 32768.0
    except Exception:
        return np.zeros(int(sample_rate * block_duration), dtype=np.float32)
