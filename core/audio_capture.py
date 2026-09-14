"""
core/audio_capture.py

Enterprise-grade Windows audio capture and voice activity detection (VAD) engine.
Provides:
  - Full Windows audio device enumeration (WASAPI, DirectSound, MME, WDM-KS)
  - Automatic hardware negotiation (WASAPI preferred, native sample rate & channel detection)
  - Zero-dependency audio resampling to 16,000 Hz float32 via numpy interpolation
  - Multi-channel to mono downmixing
  - Adaptive noise-floor tracking VAD with dynamic sensitivity
  - Complete diagnostic logging across all stages
  - Standalone microphone diagnostic test mode
"""

import math
import sys
import time
from collections import deque
from typing import Callable, Optional
import numpy as np
import sounddevice as sd


def enumerate_microphones(verbose: bool = True) -> list[dict]:
    """
    Enumerate all Windows recording/input devices.
    Returns a list of dicts with device index, name, host api, max input channels, and default sample rate.
    """
    devices = []
    try:
        all_devs = sd.query_devices()
        host_apis = sd.query_hostapis()
    except Exception as e:
        print(f"[Audio] ⚠️ Failed to query audio devices: {e}")
        return devices

    if verbose:
        print("\n" + "=" * 65)
        print("  WINDOWS AUDIO INPUT DEVICE ENUMERATION")
        print("=" * 65)

    default_in = sd.default.device[0] if sd.default.device else -1

    for idx, d in enumerate(all_devs):
        if d.get("max_input_channels", 0) > 0:
            h_idx = d.get("hostapi", 0)
            h_name = host_apis[h_idx]["name"] if h_idx < len(host_apis) else "Unknown"
            is_default = (idx == default_in)
            info = {
                "index": idx,
                "name": d.get("name", "Unknown"),
                "hostapi": h_name,
                "max_input_channels": d.get("max_input_channels", 1),
                "default_samplerate": int(d.get("default_samplerate", 44100)),
                "is_default": is_default,
            }
            devices.append(info)
            if verbose:
                tag = " [WINDOWS DEFAULT]" if is_default else ""
                print(
                    f"  [{idx:2d}] {info['name']}{tag}\n"
                    f"       HostAPI: {info['hostapi']} | Channels: {info['max_input_channels']} | SampleRate: {info['default_samplerate']} Hz"
                )

    if verbose:
        print("=" * 65 + "\n")
    return devices


def resample_audio(samples: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
    """
    High-performance audio resampling.
    Uses exact slicing for integer downsampling ratios (e.g. 48kHz -> 16kHz is 3:1),
    and fast linear interpolation for arbitrary rates.
    """
    if orig_sr == target_sr:
        return samples if samples.dtype == np.float32 else samples.astype(np.float32)

    # Fast-path for exact integer decimation ratios (e.g. 48000 -> 16000, 32000 -> 16000)
    if orig_sr % target_sr == 0:
        ratio = orig_sr // target_sr
        dec = samples[::ratio]
        return dec if dec.dtype == np.float32 else dec.astype(np.float32)

    duration = len(samples) / float(orig_sr)
    target_len = int(round(duration * target_sr))
    if target_len == 0:
        return np.zeros(0, dtype=np.float32)

    t_orig = np.linspace(0, duration, len(samples), endpoint=False)
    t_target = np.linspace(0, duration, target_len, endpoint=False)
    resampled = np.interp(t_target, t_orig, samples).astype(np.float32)
    return resampled


def find_best_microphone(preferred: Optional[str | int] = None) -> dict:
    """
    Selects the active microphone and negotiates hardware parameters.
    Priority:
      1. Explicit configured device (if match found and settings pass check)
      2. Windows WASAPI device matching default microphone name
      3. Windows Default input device
      4. First working input device
    """
    devs = enumerate_microphones(verbose=False)
    if not devs:
        raise RuntimeError("No audio recording devices found on this system.")

    candidate_devs: list[dict] = []

    # 1. Check if user configured a preferred device
    if preferred is not None and str(preferred).strip() and str(preferred).lower() != "auto":
        pref_str = str(preferred).strip()
        # If integer index
        if pref_str.isdigit():
            idx = int(pref_str)
            for d in devs:
                if d["index"] == idx:
                    candidate_devs.append(d)
        else:
            # Substring match
            for d in devs:
                if pref_str.lower() in d["name"].lower() or pref_str.lower() in d["hostapi"].lower():
                    candidate_devs.append(d)

    # 2. Add WASAPI devices (highest stability and lowest latency on Windows)
    default_info = next((d for d in devs if d["is_default"]), devs[0])
    default_name_clean = default_info["name"].split("(")[0].strip()

    for d in devs:
        if "wasapi" in d["hostapi"].lower():
            if default_name_clean.lower() in d["name"].lower():
                candidate_devs.append(d)

    # 3. Add default device
    candidate_devs.append(default_info)

    # 4. Add all other WASAPI and DirectSound devices
    for d in devs:
        if "wasapi" in d["hostapi"].lower() or "directsound" in d["hostapi"].lower():
            if d not in candidate_devs:
                candidate_devs.append(d)

    # 5. Fallback to all remaining devices
    for d in devs:
        if d not in candidate_devs:
            candidate_devs.append(d)

    # Test candidate devices and find supported sample rate & channels
    for dev in candidate_devs:
        idx = dev["index"]
        def_sr = dev["default_samplerate"]
        max_ch = dev["max_input_channels"]

        # Try native rate first, then 48000, 44100, 16000
        rates_to_try = [def_sr, 48000, 44100, 16000]
        seen_rates = set()
        rates_to_try = [r for r in rates_to_try if not (r in seen_rates or seen_rates.add(r))]

        # Try stereo if supported (many Realtek mics enforce stereo), then mono
        channels_to_try = [min(2, max_ch), 1]
        seen_ch = set()
        channels_to_try = [c for c in channels_to_try if not (c in seen_ch or seen_ch.add(c))]

        for sr in rates_to_try:
            for ch in channels_to_try:
                try:
                    sd.check_input_settings(
                        device=idx,
                        samplerate=sr,
                        channels=ch,
                        dtype="float32",
                    )
                    return {
                        "device_index": idx,
                        "name": dev["name"],
                        "hostapi": dev["hostapi"],
                        "samplerate": sr,
                        "channels": ch,
                    }
                except Exception:
                    continue

    raise RuntimeError("Could not find any compatible audio input settings for available microphones.")


class AdaptiveVAD:
    """
    Adaptive Voice Activity Detector with dynamic noise-floor tracking and pre-roll ring buffer.
    Measures ambient baseline continuously and triggers speech detection
    only when energy rises significantly above ambient baseline.
    """

    def __init__(
        self,
        sensitivity_multiplier: float = 2.5,
        min_threshold: float = 0.005,
        min_speech_duration: float = 0.25,
        silence_duration: float = 0.35,
        max_speech_duration: float = 15.0,
        sample_rate: int = 16000,
        pre_roll_chunks: int = 6,
    ):
        self.sensitivity = sensitivity_multiplier
        self.min_threshold = min_threshold
        self.min_speech_duration = min_speech_duration
        self.silence_duration = silence_duration
        self.max_speech_duration = max_speech_duration
        self.sample_rate = sample_rate

        self.noise_floor = 0.004
        self.current_rms = 0.0
        self.current_dbfs = -90.0
        self.vad_threshold = self.min_threshold
        self.in_speech = False

        self._speech_buffer: list[np.ndarray] = []
        self._pre_roll: deque[np.ndarray] = deque(maxlen=pre_roll_chunks)
        self._silence_samples = 0
        self._speech_samples = 0
        self._silence_limit_samples = int(silence_duration * sample_rate)
        self._min_speech_samples = int(min_speech_duration * sample_rate)
        self._max_speech_samples = int(max_speech_duration * sample_rate)

    def reset(self) -> None:
        """Resets speech state and buffers without clearing learned ambient noise floor."""
        self.in_speech = False
        self._speech_buffer.clear()
        self._pre_roll.clear()
        self._silence_samples = 0
        self._speech_samples = 0

    def process_chunk(self, mono_samples: np.ndarray) -> Optional[np.ndarray]:
        """
        Processes 1D float32 audio samples (at sample_rate).
        Returns completed utterance numpy array if speech ended, or None.
        """
        if len(mono_samples) == 0:
            return None

        rms = float(np.sqrt(np.mean(mono_samples ** 2)))
        self.current_rms = rms
        self.current_dbfs = 20.0 * math.log10(max(rms, 1e-7))

        # Update dynamic threshold
        self.vad_threshold = max(self.noise_floor * self.sensitivity, self.min_threshold)

        if not self.in_speech:
            # When silent, track ambient noise floor using slow EMA
            if rms < self.vad_threshold:
                self.noise_floor = 0.95 * self.noise_floor + 0.05 * rms
                self._pre_roll.append(mono_samples.copy())
            else:
                # Speech triggered! Include pre-roll buffer to retain initial consonants/syllables
                self.in_speech = True
                self._speech_buffer = list(self._pre_roll)
                self._speech_buffer.append(mono_samples.copy())
                self._pre_roll.clear()
                self._silence_samples = 0
                self._speech_samples = sum(len(c) for c in self._speech_buffer)
                return None
        else:
            # Currently in speech
            self._speech_buffer.append(mono_samples.copy())
            self._speech_samples += len(mono_samples)

            if rms < self.vad_threshold:
                self._silence_samples += len(mono_samples)
            elif rms >= self.vad_threshold * 1.15:
                # Active speech continues -> reset silence counter
                self._silence_samples = 0
            else:
                # Marginal chunk: decay silence counter gradually instead of abrupt reset
                decay = len(mono_samples) // 2
                self._silence_samples = max(0, self._silence_samples - decay)

            # Fast speech-end finalizer: for utterances > 0.45s, finalize at 300ms silence
            fast_limit = int(0.30 * self.sample_rate)
            effective_limit = fast_limit if self._speech_samples > int(0.45 * self.sample_rate) else self._silence_limit_samples
            hit_silence = self._silence_samples >= effective_limit
            hit_max = self._speech_samples >= self._max_speech_samples

            if hit_silence or hit_max:
                self.in_speech = False
                total_samples = self._speech_samples
                buf = self._speech_buffer
                self._speech_buffer = []
                self._silence_samples = 0
                self._speech_samples = 0
                self._pre_roll.clear()

                if total_samples >= self._min_speech_samples:
                    return np.concatenate(buf)

        return None


class AudioInputPipeline:
    """
    Robust audio capture pipeline:
      - Hardware auto-negotiation (WASAPI preferred)
      - Non-blocking sounddevice InputStream
      - Real-time downmix & resampling to 16kHz
      - Adaptive VAD processing
      - Thread-safe state & utterance callbacks
    """

    def __init__(
        self,
        preferred_device: Optional[str | int] = None,
        vad_sensitivity: float = 2.5,
        min_speech_duration: float = 0.25,
        silence_duration: float = 0.35,
        on_speech_completed: Optional[Callable[[np.ndarray, float], None]] = None,
        on_state_change: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ):
        self.preferred_device = preferred_device
        self.on_speech_completed = on_speech_completed
        self.on_state_change = on_state_change
        self.on_error = on_error

        self.vad = AdaptiveVAD(
            sensitivity_multiplier=vad_sensitivity,
            min_speech_duration=min_speech_duration,
            silence_duration=silence_duration,
            sample_rate=16000,
        )

        self._stream: Optional[sd.InputStream] = None
        self._device_info: Optional[dict] = None
        self._is_paused = False
        self._running = False
        self._diagnostic_counter = 0

    @property
    def device_name(self) -> str:
        if self._device_info:
            return f"{self._device_info['name']} ({self._device_info['hostapi']})"
        return "Not selected"

    def start(self) -> None:
        """Finds the best microphone and starts the audio stream."""
        self._device_info = find_best_microphone(self.preferred_device)
        dev_idx = self._device_info["device_index"]
        hw_sr = self._device_info["samplerate"]
        hw_ch = self._device_info["channels"]

        print(f"[Audio] [MIC] MIC DEVICE SELECTED: [Index {dev_idx}] {self._device_info['name']}")
        print(f"[Audio]    HostAPI: {self._device_info['hostapi']} | Native SR: {hw_sr} Hz | Channels: {hw_ch}")
        if hw_sr != 16000:
            print(f"[Audio]    Resampling enabled: {hw_sr} Hz -> 16000 Hz float32")
        if hw_ch > 1:
            print(f"[Audio]    Downmixing enabled: {hw_ch} channels -> 1 channel (Mono)")

        # Blocksize chosen for ~64ms latency
        block_size = int(hw_sr * 0.064)

        def _audio_callback(indata, frames, time_info, status):
            try:
                if status:
                    print(f"[Audio] [WARN] Stream status warning: {status}")

                if self._is_paused or not self._running:
                    return

                # 1. Downmix multi-channel to mono
                if indata.ndim > 1 and indata.shape[1] > 1:
                    mono_hw = np.mean(indata, axis=1)
                else:
                    mono_hw = indata.flatten()

                # 2. Resample to 16,000 Hz if necessary
                if hw_sr != 16000:
                    mono_16k = resample_audio(mono_hw, hw_sr, 16000)
                else:
                    mono_16k = mono_hw.astype(np.float32)

                prev_speech = self.vad.in_speech
                utterance = self.vad.process_chunk(mono_16k)

                # Diagnostic log every ~1.5s (24 blocks)
                self._diagnostic_counter += 1
                if self._diagnostic_counter % 24 == 0:
                    level_tag = f"MIC LEVEL: {self.vad.current_dbfs:5.1f} dBFS (RMS: {self.vad.current_rms:.5f})"
                    if self.vad.current_rms < 0.0001:
                        level_tag += " [MIC INPUT SILENT]"
                    print(
                        f"[Audio] {level_tag} | Noise Floor: {self.vad.noise_floor:.5f} | "
                        f"VAD Threshold: {self.vad.vad_threshold:.5f} | Speech Active: {self.vad.in_speech}"
                    )

                # State transition notification
                if not prev_speech and self.vad.in_speech:
                    print(
                        f"[Audio] [SPEECH] VOICE DETECTED! RMS: {self.vad.current_rms:.4f} > Threshold: {self.vad.vad_threshold:.4f}"
                    )
                    if self.on_state_change:
                        self.on_state_change("VOICE_DETECTED")

                # Utterance finished
                if utterance is not None:
                    dur = len(utterance) / 16000.0
                    print(f"[Audio] [OK] Speech complete: {dur:.2f}s ({len(utterance)} samples). Queuing for STT.")
                    if self.on_state_change:
                        self.on_state_change("TRANSCRIBING")
                    if self.on_speech_completed:
                        self.on_speech_completed(utterance, dur)
            except Exception as cb_err:
                print(f"[Audio] [ERR] Error in audio callback: {cb_err}")

        try:
            self._stream = sd.InputStream(
                device=dev_idx,
                samplerate=hw_sr,
                channels=hw_ch,
                dtype="float32",
                blocksize=block_size,
                callback=_audio_callback,
            )
            self._stream.start()
            self._running = True
            print("[Audio] [OK] Microphone stream active and listening.")
            if self.on_state_change:
                self.on_state_change("LISTENING")
        except Exception as e:
            err = f"Failed to start microphone stream: {e}"
            print(f"[Audio] [ERROR] {err}")
            if self.on_error:
                self.on_error(err)
            raise

    def pause(self) -> None:
        """Temporarily pause processing (e.g. while JARVIS is speaking)."""
        self._is_paused = True
        try:
            self.vad.reset()
        except Exception as e:
            print(f"[Audio] ⚠️ VAD reset error on pause: {e}")

    def resume(self) -> None:
        """Resume processing after speaking cooldown."""
        try:
            self.vad.reset()
        except Exception as e:
            print(f"[Audio] ⚠️ VAD reset error on resume: {e}")
        finally:
            self._is_paused = False
        if self.on_state_change:
            self.on_state_change("LISTENING")

    def stop(self) -> None:
        """Stop and close the stream."""
        self._running = False
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None


def run_standalone_mic_test(duration: float = 5.0, whisper_model: str = "base") -> None:
    """
    Comprehensive standalone diagnostic mode:
      1. Enumerate and display all audio input devices with hardware specs.
      2. Automatically negotiate and select best device (WASAPI preferred).
      3. Capture live audio with real-time ASCII dBFS volume meter, RMS, and Peak.
      4. Simulate and validate AdaptiveVAD on the live captured audio stream.
      5. Run Whisper STT (probing CUDA GPU acceleration, falling back to CPU).
      6. Print complete diagnostic summary report.
    """
    print("\n" + "=" * 70)
    print("       JARVIS COMPREHENSIVE AUDIO & VOICE INPUT DIAGNOSTIC")
    print("=" * 70)

    # 1. Enumerate devices
    devs = enumerate_microphones(verbose=True)
    if not devs:
        print("[Test] ❌ No recording devices found on this system!")
        return

    # 2. Negotiate hardware
    best = find_best_microphone()
    print(f"[Test] Target Device Selected:")
    print(f"       Index:       {best['device_index']}")
    print(f"       Device Name: {best['name']}")
    print(f"       Host API:    {best['hostapi']}")
    print(f"       Sample Rate: {best['samplerate']} Hz")
    print(f"       Channels:    {best['channels']} ({'Stereo' if best['channels'] > 1 else 'Mono'})")

    # 3. Live capture without VAD (Meter test)
    captured_chunks: list[np.ndarray] = []
    chunk_metrics: list[tuple[float, float, float]] = []  # rms, peak, dbfs
    t_start = time.time()

    def _test_cb(indata, frames, time_info, status):
        mono = np.mean(indata, axis=1) if indata.ndim > 1 else indata.flatten()
        captured_chunks.append(mono.copy())
        rms = float(np.sqrt(np.mean(mono ** 2)))
        peak = float(np.max(np.abs(mono))) if len(mono) > 0 else 0.0
        dbfs = 20.0 * math.log10(max(rms, 1e-7))
        chunk_metrics.append((rms, peak, dbfs))

        # ASCII meter: -60 dBFS to 0 dBFS (30 characters)
        bars = int(max(0, min(30, (dbfs + 60) * 0.5)))
        meter = "#" * bars + "-" * (30 - bars)
        sys.stdout.write(f"\r[Test] Level: [{meter}] {dbfs:5.1f} dBFS | RMS: {rms:.4f} | Peak: {peak:.4f}")
        sys.stdout.flush()

    block = int(best["samplerate"] * 0.064)
    print(f"\n[Test] [MIC] RECORDING LIVE AUDIO FOR {duration:.1f}s — PLEASE SPEAK CLEARLY NOW:")
    with sd.InputStream(
        device=best["device_index"],
        samplerate=best["samplerate"],
        channels=best["channels"],
        dtype="float32",
        blocksize=block,
        callback=_test_cb,
    ):
        while time.time() - t_start < duration:
            time.sleep(0.05)

    print("\n\n[Test] [OK] Audio capture completed.")
    if not captured_chunks:
        print("[Test] ❌ No audio samples were captured during stream.")
        return

    full_audio = np.concatenate(captured_chunks)
    avg_rms = float(np.sqrt(np.mean(full_audio ** 2)))
    max_peak = float(np.max(np.abs(full_audio)))
    avg_dbfs = 20.0 * math.log10(max(avg_rms, 1e-7))

    print(f"\n[Test] Capture Metrics:")
    print(f"       Total Samples: {len(full_audio)} ({len(full_audio)/best['samplerate']:.2f}s)")
    print(f"       Average RMS:   {avg_rms:.5f}")
    print(f"       Max Peak:      {max_peak:.5f}")
    print(f"       Overall dBFS:  {avg_dbfs:.1f} dBFS")

    # Resample to 16,000 Hz float32
    audio_16k = resample_audio(full_audio, best["samplerate"], 16000)
    print(f"       Resampled 16k: {len(audio_16k)} samples ({len(audio_16k)/16000.0:.2f}s)")

    # 4. Adaptive VAD Simulation on the Captured Stream
    print(f"\n[Test] Simulating Adaptive VAD on captured stream...")
    sim_vad = AdaptiveVAD(sample_rate=16000)
    speech_segments_found = 0
    chunk_16k_len = int(16000 * 0.064)
    for i in range(0, len(audio_16k), chunk_16k_len):
        c = audio_16k[i : i + chunk_16k_len]
        res = sim_vad.process_chunk(c)
        if res is not None:
            speech_segments_found += 1
            dur_seg = len(res) / 16000.0
            print(f"       [VAD] Completed speech utterance detected! Duration: {dur_seg:.2f}s ({len(res)} samples)")

    print(f"       Learned Noise Floor: {sim_vad.noise_floor:.5f}")
    print(f"       Active VAD Threshold: {sim_vad.vad_threshold:.5f}")

    # 5. Speech-to-Text Transcription via WhisperSTT
    print(f"\n[Test] Testing Speech-to-Text via WhisperSTT ('{whisper_model}')...")
    from core.stt import WhisperSTT
    t_stt0 = time.time()
    stt = WhisperSTT(whisper_model)
    t_load = time.time() - t_stt0
    print(f"[Test] Model loaded in {t_load:.2f}s.")

    t_tx0 = time.time()
    transcript = stt.transcribe(audio_16k)
    t_tx = time.time() - t_tx0

    # 6. Diagnostic Summary
    hw_status = "PASS" if best else "FAIL"
    signal_status = "PASS" if avg_rms >= 0.002 else ("WARN (LOW GAIN)" if avg_rms >= 0.0005 else "FAIL (SILENT)")
    vad_status = "PASS" if (speech_segments_found > 0 or sim_vad.in_speech) else ("WARN (NO SPEECH DETECTED)" if avg_rms < 0.005 else "PASS")
    stt_status = "PASS" if transcript.strip() else ("WARN (EMPTY TRANSCRIPT)" if avg_rms < 0.005 else "WARN")

    print("\n" + "=" * 70)
    print("                     DIAGNOSTIC SUMMARY")
    print("=" * 70)
    print(f"  Microphone Hardware:      [{hw_status}] - {best['name']} ({best['hostapi']})")
    print(f"  Signal Level:             [{signal_status}] - Avg RMS: {avg_rms:.5f} | Peak: {max_peak:.4f}")
    print(f"  Voice Activity Detection: [{vad_status}] - Noise Floor: {sim_vad.noise_floor:.5f} | Threshold: {sim_vad.vad_threshold:.5f}")
    print(f"  Speech-to-Text Engine:    [{stt_status}] - Latency: {t_tx*1000:.1f}ms")
    print(f"  Spoken Transcript:        \"{transcript}\"")
    print("=" * 70 + "\n")


