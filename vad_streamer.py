"""
demo/vad_streamer.py — Adaptive VAD (Voice Activity Detector) for Continuous Streaming
========================================================================================
Nhận luồng PCM float32 16kHz liên tục và cắt thành các utterance (câu nói hoàn chỉnh)
dựa trên phát hiện khoảng im lặng.

Cơ chế:
  - Sử dụng thuật toán Adaptive Energy VAD: tính RMS của từng chunk PCM.
  - Ngưỡng (threshold) tự động thích nghi theo mức nhiễu nền (noise floor).
  - Phát hiện kết thúc câu khi RMS dưới ngưỡng liên tục ≥ silence_limit_ms.
  - Hỗ trợ cập nhật tham số realtime qua update_config().

Tích hợp với Sherpa-onnx SileroVAD nếu model tồn tại (chính xác hơn),
ngược lại fallback về adaptive RMS VAD (không phụ thuộc model).
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

import numpy as np

log = logging.getLogger("onevoice.vad")

# ─── Đường dẫn model Silero VAD (tùy chọn, không bắt buộc) ──────────────────
BASE_DIR = Path(__file__).parent
SILERO_VAD_MODEL = BASE_DIR / "models" / "silero_vad.onnx"


# ════════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class VADConfig:
    """Cấu hình Voice Activity Detector."""

    # Khoảng im lặng tối thiểu để kết thúc một utterance (ms)
    silence_limit_ms: int = 500

    # Thời gian nói tối thiểu để tính là utterance hợp lệ (ms) — loại bỏ tiếng ho/thở
    min_speech_ms: int = 200

    # Thời gian nói tối đa trước khi cắt cưỡng bức (ms) — chống tràn RAM
    max_speech_ms: int = 10_000

    # Ngưỡng RMS tự động điều chỉnh — khởi đầu (0.01 = rất nhạy, 0.05 = trung bình)
    initial_threshold: float = 0.02

    # Hệ số update noise floor: weighted average với (1 - alpha) * floor + alpha * rms
    noise_adaptation_alpha: float = 0.05

    # Tỉ lệ nhân ngưỡng so với noise floor (speech = floor * ratio)
    speech_threshold_ratio: float = 3.5

    # Cỡ buffer trượt để ước tính noise floor (số chunk)
    noise_window_size: int = 30

    # Dùng Silero VAD nếu model tồn tại (chính xác hơn nhưng cần model)
    use_silero: bool = True

    # Ngưỡng Silero (0.0–1.0): xác suất giọng nói tối thiểu để coi là đang nói
    silero_threshold: float = 0.5


# ════════════════════════════════════════════════════════════════════════════
# STATE
# ════════════════════════════════════════════════════════════════════════════

class VADState(Enum):
    SILENCE = "silence"   # Đang im lặng, chờ giọng nói
    SPEECH  = "speech"    # Đang nghe giọng nói
    ENDED   = "ended"     # Vừa kết thúc câu (phát utterance)


@dataclass
class VADInternalState:
    state: VADState = VADState.SILENCE
    speech_buffer: list = field(default_factory=list)   # Danh sách chunk PCM trong câu
    silence_samples: int = 0   # Số sample im lặng tích lũy từ khi speech kết thúc
    speech_samples: int = 0    # Số sample đang nói
    noise_rms: float = 0.01    # Ước tính noise floor hiện tại
    noise_history: deque = field(default_factory=lambda: deque(maxlen=30))


# ════════════════════════════════════════════════════════════════════════════
# VAD STREAMER
# ════════════════════════════════════════════════════════════════════════════

class VADStreamer:
    """
    Nhận luồng PCM Float32 16kHz theo từng chunk nhỏ.
    Gọi callback on_utterance(samples, uid) mỗi khi phát hiện xong một câu hoàn chỉnh.
    Gọi callback on_speech_start(uid) khi người dùng bắt đầu nói.

    Ví dụ sử dụng:
        vad = VADStreamer(config=VADConfig(), sample_rate=16000,
                         on_speech_start=lambda uid: ...,
                         on_utterance=lambda samples, uid: ...)
        vad.feed(pcm_chunk)   # Gọi liên tục với chunk audio
        vad.flush()           # Kết thúc session — đẩy câu đang dở nếu có
    """

    SAMPLE_RATE = 16_000

    def __init__(
        self,
        config: Optional[VADConfig] = None,
        sample_rate: int = 16_000,
        on_speech_start: Optional[Callable[[int], None]] = None,
        on_utterance: Optional[Callable[[np.ndarray, int], None]] = None,
    ):
        self.config        = config or VADConfig()
        self.sample_rate   = sample_rate
        self.on_speech_start = on_speech_start
        self.on_utterance    = on_utterance

        self._state        = VADInternalState()
        self._state.noise_history = deque(maxlen=self.config.noise_window_size)
        self._uid          = 0
        self._current_uid  = 0
        self._total_fed    = 0        # Tổng số sample đã nhận

        # Thử load Silero VAD ONNX
        self._silero       = None
        self._use_silero   = False
        if self.config.use_silero:
            self._try_load_silero()

        log.info(
            f"VADStreamer initialized — backend={'silero' if self._use_silero else 'adaptive_rms'} "
            f"silence={self.config.silence_limit_ms}ms "
            f"min_speech={self.config.min_speech_ms}ms "
            f"max_speech={self.config.max_speech_ms}ms"
        )

    def _try_load_silero(self):
        """Thử load Silero VAD ONNX. Bỏ qua nếu model không tồn tại."""
        try:
            import sherpa_onnx
            if SILERO_VAD_MODEL.exists():
                config = sherpa_onnx.VadModelConfig(
                    silero_vad=sherpa_onnx.SileroVadModelConfig(
                        model=str(SILERO_VAD_MODEL),
                        threshold=self.config.silero_threshold,
                        min_silence_duration=self.config.silence_limit_ms / 1000.0,
                        min_speech_duration=self.config.min_speech_ms / 1000.0,
                        max_speech_duration=self.config.max_speech_ms / 1000.0,
                    ),
                    sample_rate=self.SAMPLE_RATE,
                    num_threads=2,
                )
                self._silero = sherpa_onnx.VoiceActivityDetector(config, buf_size=60)
                self._use_silero = True
                log.info(f"Silero VAD loaded from {SILERO_VAD_MODEL.name}")
            else:
                log.info("Silero VAD model not found — using Adaptive RMS VAD")
        except Exception as e:
            log.info(f"Silero VAD unavailable ({e}) — using Adaptive RMS VAD")

    def update_config(self, **kwargs):
        """
        Cập nhật tham số VAD trong thời gian thực.
        Ví dụ: vad.update_config(silence_limit_ms=600, silero_threshold=0.4)
        """
        for k, v in kwargs.items():
            if hasattr(self.config, k):
                old_val = getattr(self.config, k)
                setattr(self.config, k, v)
                log.info(f"VAD config updated: {k} = {old_val} → {v}")
            else:
                log.warning(f"VAD config: unknown key '{k}' — skipped")

        # Cập nhật ngưỡng Silero nếu đang dùng và threshold thay đổi
        if self._use_silero and self._silero and "silero_threshold" in kwargs:
            # Silero VAD không hỗ trợ hot-reload, cần reset
            log.info("Silero threshold changed — re-initializing Silero VAD")
            self._try_load_silero()

    def feed(self, samples: np.ndarray):
        """
        Nhận một chunk PCM float32 16kHz. Gọi hàm này liên tục.
        Các callback on_speech_start / on_utterance được gọi tự động.
        """
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)
        if samples.ndim != 1:
            samples = samples.flatten()

        self._total_fed += len(samples)

        if self._use_silero:
            self._feed_silero(samples)
        else:
            self._feed_adaptive_rms(samples)

    def flush(self):
        """
        Kết thúc session. Nếu đang trong trạng thái SPEECH và đủ độ dài,
        đẩy câu cuối dù chưa đạt silence_limit.
        """
        if (
            self._state.state == VADState.SPEECH
            and self._state.speech_samples >= self._ms_to_samples(self.config.min_speech_ms)
        ):
            self._emit_utterance()
            log.debug("VAD flush: emitted trailing utterance")

        self._state = VADInternalState()
        self._state.noise_history = deque(maxlen=self.config.noise_window_size)
        log.debug("VAD flush: state reset")

    def reset(self):
        """Reset hoàn toàn, bỏ mọi buffer đang giữ."""
        self._state = VADInternalState()
        self._state.noise_history = deque(maxlen=self.config.noise_window_size)
        self._current_uid = 0
        log.debug("VAD reset")

    # ─── Adaptive RMS VAD ────────────────────────────────────────────────────

    def _feed_adaptive_rms(self, samples: np.ndarray):
        """Xử lý chunk PCM với thuật toán Adaptive Energy VAD."""
        rms = float(np.sqrt(np.mean(samples ** 2)))
        n_samples = len(samples)
        s = self._state

        # Cập nhật noise floor khi im lặng
        if s.state == VADState.SILENCE:
            s.noise_history.append(rms)
            if len(s.noise_history) >= 5:
                s.noise_rms = float(np.percentile(list(s.noise_history), 20))

        # Tính ngưỡng nói
        threshold = max(
            s.noise_rms * self.config.speech_threshold_ratio,
            self.config.initial_threshold,
        )
        is_speech = rms > threshold

        if s.state == VADState.SILENCE:
            if is_speech:
                # Bắt đầu câu mới
                s.state = VADState.SPEECH
                s.speech_samples = n_samples
                s.silence_samples = 0
                s.speech_buffer = [samples.copy()]
                self._current_uid = self._next_uid()
                log.debug(f"VAD → SPEECH uid={self._current_uid} rms={rms:.4f} thr={threshold:.4f}")
                if self.on_speech_start:
                    try:
                        self.on_speech_start(self._current_uid)
                    except Exception as e:
                        log.error(f"VAD on_speech_start error: {e}")

        elif s.state == VADState.SPEECH:
            s.speech_buffer.append(samples.copy())
            max_samples = self._ms_to_samples(self.config.max_speech_ms)

            if is_speech:
                s.speech_samples += n_samples
                s.silence_samples = 0
                # Cắt cưỡng bức nếu câu quá dài
                if s.speech_samples >= max_samples:
                    log.info(f"VAD: Max speech duration reached ({self.config.max_speech_ms}ms) — force cut uid={self._current_uid}")
                    self._emit_utterance()
            else:
                s.silence_samples += n_samples
                silence_limit_samples = self._ms_to_samples(self.config.silence_limit_ms)

                if s.silence_samples >= silence_limit_samples:
                    min_samples = self._ms_to_samples(self.config.min_speech_ms)
                    if s.speech_samples >= min_samples:
                        log.debug(f"VAD → SILENCE (utterance complete) uid={self._current_uid} speech={s.speech_samples/self.SAMPLE_RATE:.2f}s")
                        self._emit_utterance()
                    else:
                        # Quá ngắn — bỏ qua (tiếng ho/thở)
                        log.debug(f"VAD: utterance too short ({s.speech_samples/self.SAMPLE_RATE:.2f}s < {self.config.min_speech_ms}ms) — discarded")
                        s.state = VADState.SILENCE
                        s.speech_buffer = []
                        s.speech_samples = 0
                        s.silence_samples = 0

    # ─── Silero VAD ──────────────────────────────────────────────────────────

    def _feed_silero(self, samples: np.ndarray):
        """Xử lý chunk PCM với Silero VAD model (sherpa-onnx)."""
        try:
            self._silero.accept_waveform(samples)
            while not self._silero.empty():
                speech_segment = self._silero.front()
                self._silero.pop()
                seg_samples = np.array(speech_segment.samples, dtype=np.float32)
                uid = self._next_uid()
                log.debug(f"Silero VAD utterance uid={uid} dur={len(seg_samples)/self.SAMPLE_RATE:.2f}s")
                if self.on_speech_start:
                    try:
                        self.on_speech_start(uid)
                    except Exception as e:
                        log.error(f"VAD on_speech_start error: {e}")
                if self.on_utterance:
                    try:
                        self.on_utterance(seg_samples, uid)
                    except Exception as e:
                        log.error(f"VAD on_utterance error: {e}")
        except Exception as e:
            log.error(f"Silero VAD feed error: {e} — falling back to adaptive RMS")
            self._use_silero = False
            self._feed_adaptive_rms(samples)

    # ─── Helpers ─────────────────────────────────────────────────────────────

    def _emit_utterance(self):
        """Đóng gói và phát emit câu hiện tại, reset state."""
        s = self._state
        if not s.speech_buffer:
            return

        # Ghép toàn bộ buffer thành mảng liên tục
        audio = np.concatenate(s.speech_buffer)

        # Reset state trước khi callback (tránh re-entrancy)
        uid = self._current_uid
        s.state = VADState.SILENCE
        s.speech_buffer = []
        s.speech_samples = 0
        s.silence_samples = 0

        dur = len(audio) / self.SAMPLE_RATE
        log.info(f"VAD utterance emitted uid={uid} dur={dur:.2f}s samples={len(audio)}")

        if self.on_utterance:
            try:
                self.on_utterance(audio, uid)
            except Exception as e:
                log.error(f"VAD on_utterance callback error: {e}")

    def _next_uid(self) -> int:
        self._uid += 1
        return self._uid

    def _ms_to_samples(self, ms: int) -> int:
        return int(ms * self.SAMPLE_RATE / 1000)

    @property
    def current_threshold(self) -> float:
        """Ngưỡng RMS hiện tại đang dùng để phân loại speech/silence."""
        if self._use_silero:
            return self.config.silero_threshold
        s = self._state
        return max(
            s.noise_rms * self.config.speech_threshold_ratio,
            self.config.initial_threshold,
        )

    @property
    def noise_floor(self) -> float:
        """Ước tính mức nhiễu nền hiện tại."""
        return self._state.noise_rms

    @property
    def backend(self) -> str:
        return "silero" if self._use_silero else "adaptive_rms"
