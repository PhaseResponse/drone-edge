import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import realtime_pipeline as pipeline


class SpectrogramContractTests(unittest.TestCase):
    def test_model_input_shape_and_dtype(self):
        audio = np.zeros(pipeline.N_SEGMENT, dtype=np.float32)

        spectrogram = pipeline.samples_to_spectrogram(audio)

        self.assertEqual(spectrogram.shape, (193, 39))
        self.assertEqual(spectrogram.dtype, np.float32)

    def test_output_is_finite_and_normalized(self):
        rng = np.random.default_rng(0)
        audio = rng.normal(0.0, 0.01, pipeline.N_SEGMENT).astype(np.float32)

        spectrogram = pipeline.samples_to_spectrogram(audio)

        self.assertTrue(np.isfinite(spectrogram).all())
        self.assertGreaterEqual(float(spectrogram.min()), 0.0)
        self.assertLessEqual(float(spectrogram.max()), 1.0)


    def test_tone_peak_is_at_expected_frequency(self):
        frequency = 1000.0
        samples = np.arange(pipeline.N_SEGMENT)
        audio = (
            1e-4
            * np.sin(2 * np.pi * frequency * samples / pipeline.FS)
        ).astype(np.float32)

        spectrogram = pipeline.samples_to_spectrogram(audio)

        dominant_bin = int(np.argmax(spectrogram.mean(axis=1)))
        expected_bin = round(frequency / (pipeline.FS / pipeline.N_FFT))
        self.assertEqual(dominant_bin, expected_bin)


if __name__ == "__main__":
    unittest.main()
