import numpy as np
import librosa

def block_to_mel(block: np.ndarray,
                 sample_rate: int,
                 n_fft: int,
                 hop_length: int,
                 n_mels: int,
                 fmax: int) -> np.ndarray:

    S = librosa.feature.melspectrogram(
        y=block,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmax=fmax,
        power=2.0,
        center=True
    )
    S_db = librosa.power_to_db(S, ref=np.max)

    mel = S_db.T[np.newaxis, ...]

    n_frames = block.shape[0] // hop_length
    mel = mel[:, :n_frames, :]
    print(mel.shape)
    return mel
