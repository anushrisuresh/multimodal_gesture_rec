import sounddevice as sd
import numpy as np

SR = 48000
BLOCK_DUR = 1.92
BLOCK_SAMPLES = int(SR * BLOCK_DUR)

def get_block():
        audio = sd.rec(BLOCK_SAMPLES, samplerate=SR, channels=1, dtype='int16')
        sd.wait()
        return audio.flatten().astype(np.float32) / 32768.0  #normalize

