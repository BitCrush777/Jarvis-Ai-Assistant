import unittest
import numpy as np
from core.audio_capture import (
    enumerate_microphones,
    find_best_microphone,
    resample_audio,
    AdaptiveVAD,
)


class TestAudioCapture(unittest.TestCase):

    def test_enumeration(self):
        devs = enumerate_microphones(verbose=False)
        self.assertIsInstance(devs, list)
        self.assertGreater(len(devs), 0)
        first = devs[0]
        self.assertIn("index", first)
        self.assertIn("name", first)
        self.assertIn("hostapi", first)
        self.assertIn("max_input_channels", first)
        self.assertIn("default_samplerate", first)

    def test_find_best_microphone(self):
        best = find_best_microphone()
        self.assertIsInstance(best, dict)
        self.assertIn("device_index", best)
        self.assertIn("samplerate", best)
        self.assertIn("channels", best)
        self.assertGreater(best["samplerate"], 0)
        self.assertGreater(best["channels"], 0)

    def test_resample_audio(self):
        orig_sr = 48000
        target_sr = 16000
        duration = 1.0  # 1 second
        # Generate 1 second of 440 Hz sine wave
        t = np.linspace(0, duration, int(orig_sr * duration), endpoint=False)
        orig_samples = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        resampled = resample_audio(orig_samples, orig_sr, target_sr)
        self.assertEqual(len(resampled), 16000)
        self.assertEqual(resampled.dtype, np.float32)
        # Check signal energy is preserved
        orig_rms = float(np.sqrt(np.mean(orig_samples ** 2)))
        resampled_rms = float(np.sqrt(np.mean(resampled ** 2)))
        self.assertAlmostEqual(orig_rms, resampled_rms, places=2)

    def test_adaptive_vad_silence_vs_speech(self):
        vad = AdaptiveVAD(
            sensitivity_multiplier=2.5,
            min_threshold=0.005,
            min_speech_duration=0.1,
            silence_duration=0.2,
            sample_rate=16000,
        )

        # 1. Feed low-amplitude noise (silence)
        noise = np.random.uniform(-0.002, 0.002, 1024).astype(np.float32)
        for _ in range(5):
            res = vad.process_chunk(noise)
            self.assertIsNone(res)
            self.assertFalse(vad.in_speech)

        # 2. Feed high-amplitude speech signal
        speech_chunk = np.sin(np.linspace(0, 50, 1024)).astype(np.float32) * 0.2
        res = vad.process_chunk(speech_chunk)
        self.assertTrue(vad.in_speech)

        # 3. Continue speech for a few chunks
        for _ in range(3):
            vad.process_chunk(speech_chunk)

        # 4. Feed silence chunks until silence limit triggers utterance completion
        utterance = None
        for _ in range(6):
            r = vad.process_chunk(noise)
            if r is not None:
                utterance = r
                break

        self.assertIsNotNone(utterance)
        self.assertFalse(vad.in_speech)
        self.assertGreater(len(utterance), 1024)

    def test_vad_reset(self):
        vad = AdaptiveVAD(sample_rate=16000)
        # Put in speech
        speech_chunk = np.ones(1024, dtype=np.float32) * 0.1
        vad.process_chunk(speech_chunk)
        self.assertTrue(vad.in_speech)
        self.assertGreater(len(vad._speech_buffer), 0)

        # Call reset
        vad.reset()
        self.assertFalse(vad.in_speech)
        self.assertEqual(len(vad._speech_buffer), 0)
        self.assertEqual(vad._silence_samples, 0)
        self.assertEqual(vad._speech_samples, 0)

    def test_pipeline_pause_resume(self):
        from core.audio_capture import AudioInputPipeline
        pipe = AudioInputPipeline()
        self.assertFalse(pipe._is_paused)
        pipe.pause()
        self.assertTrue(pipe._is_paused)
        pipe.resume()
        self.assertFalse(pipe._is_paused)


if __name__ == "__main__":
    unittest.main()
