import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import batear_detection as batear


class ProcessFileContractTests(unittest.TestCase):
    def test_per_frame_outputs_stay_aligned_for_silence(self):
        audio = np.zeros(
            batear.FFT_SIZE * 4,
            dtype=np.float32,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "silence.wav"
            wavfile.write(
                wav_path,
                batear.SAMPLE_RATE,
                audio,
            )

            result = batear.process_file(wav_path)

        expected_frames = len(result["times"])
        self.assertGreater(expected_frames, 0)

        for key in (
            "confidences",
            "detections",
            "fundamentals",
            "h2_ratios",
            "h3_ratios",
            "alarms",
        ):
            with self.subTest(key=key):
                self.assertEqual(
                    len(result[key]),
                    expected_frames,
                )

        np.testing.assert_array_equal(
            result["h2_ratios"],
            np.zeros(expected_frames),
        )
        np.testing.assert_array_equal(
            result["h3_ratios"],
            np.zeros(expected_frames),
        )


if __name__ == "__main__":
    unittest.main()
