import numpy as np
import tflite_runtime.interpreter as tflite

class InferenceEngine:
    def __init__(self, model_path: str):
        import os
        import glob
        if not os.path.isfile(model_path):
            dirpath = os.path.dirname(model_path) or '.'
            alt = os.path.join(dirpath, 'quantized_model_int8.tflite')
            if os.path.isfile(alt):
                model_path = alt
            else:
                candidates = glob.glob(os.path.join(dirpath, '*.tflite'))
                if candidates:
                    model_path = candidates[0]
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        shapes = [d['shape'] for d in self.input_details]
        self.mic_idx = next(i for i,s in enumerate(shapes) if s[1] != 391)
        self.us_idx = 1 - self.mic_idx

        # Prepare dummy ultrasound
        us_shape = self.input_details[self.us_idx]['shape']
        self.dummy_us = np.zeros(us_shape,
                                 dtype=self.input_details[self.us_idx]['dtype'])

    def classify(self, mel_block: np.ndarray) -> np.ndarray:

        # set mic input
        self.interpreter.set_tensor(
            self.input_details[self.mic_idx]['index'],
            mel_block.astype(self.input_details[self.mic_idx]['dtype'])
        )
        # set randomized ultrasound input for testing
        us_input = np.ones_like(self.dummy_us)
        # us_input = np.random.randn(*self.dummy_us.shape).astype(self.dummy_us.dtype)
        self.interpreter.set_tensor(
            self.input_details[self.us_idx]['index'],
            us_input
        )
        self.interpreter.invoke()
        return self.interpreter.get_tensor(
            self.output_details[0]['index']
        ).flatten()

_engine = None
def classify(mel_block, model_path=None):
    global _engine
    if _engine is None:
        _engine = InferenceEngine(model_path)
    return _engine.classify(mel_block)
