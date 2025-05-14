import sounddevice as sd
import numpy as np

def get_block(sample_rate: int, block_duration: float) -> np.ndarray:
    """
    Record `block_duration` seconds from the default mic, normalize to [-1,1].
    Returns:
        block: float32 array of shape (samples,)
    """
    num_samples = int(sample_rate * block_duration)
    recording = sd.rec(num_samples,
                       samplerate=sample_rate,
                       channels=1,
                       dtype='int16')
    sd.wait()
    # Normalize to [-1, 1]
    return recording.flatten().astype(np.float32) / 32768.0
