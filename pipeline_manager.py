"""
demo/pipeline_manager.py — OneVoice Plug-and-Play Pipeline Manager
====================================================================
Quản lý 4 luồng dịch Speech-to-Speech:
  vi-en : VI → EN  (Zipformer 30M + Opus-MT vi-en + Kokoro EN)
  en-vi : EN → VI  (Moonshine Base INT8 + Opus-MT en-vi + Piper/Kokoro VI)
  vi-zh : VI → ZH  (Zipformer 30M + Opus-MT vi-zh + Kokoro ZH)
  zh-vi : ZH → VI  (Paraformer ZH INT8 + Opus-MT zh-vi + Piper/Kokoro VI)

Cơ chế:
  - Lazy Load on Demand: chỉ nạp model khi người dùng chọn luồng.
  - Auto Unload: giải phóng RAM của luồng cũ trước khi nạp luồng mới.
  - RAM & Latency tracking: đo đạc bằng psutil + time.perf_counter.
"""

from __future__ import annotations

import gc
import io
import logging
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

log = logging.getLogger("onevoice.pipeline")

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"


# ════════════════════════════════════════════════════════════════════════════
# PIPELINE CONFIGS
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineConfig:
    """Mô tả cấu hình một luồng dịch cụ thể."""
    direction:   str    # "vi-en", "en-vi", "vi-zh", "zh-vi"
    src_lang:    str    # Ngôn ngữ nguồn
    tgt_lang:    str    # Ngôn ngữ đích
    label:       str    # Nhãn hiển thị

    # ASR
    asr_type:    str    # "zipformer" | "moonshine" | "paraformer"
    asr_dir:     str    # Thư mục model trong MODELS_DIR

    # MT
    mt_type:     str    # "marian" | "vinai"
    mt_dir:      str    # Thư mục model trong MODELS_DIR hoặc HF repo

    # TTS
    tts_type:    str    # "kokoro" | "piper"
    tts_dir:     str    # Thư mục model TTS
    tts_model:   str    # Tên file ONNX
    tts_voices:  str    # File voice / data_dir
    tts_voice:   str    # Voice ID
    tts_lang:    str    # Lang code ("en-us", "vi", "zh")


PIPELINE_CONFIGS: dict[str, PipelineConfig] = {
    "vi-en": PipelineConfig(
        direction="vi-en",
        src_lang="vi", tgt_lang="en",
        label="🇻🇳 Tiếng Việt → 🇬🇧 English",
        # ASR
        asr_type="zipformer",
        asr_dir="sherpa-onnx-zipformer-vi-30M-int8-2026-02-09",
        # MT
        mt_type="marian",
        mt_dir="opus-mt-vi-en",
        # TTS
        tts_type="kokoro",
        tts_dir="kokoro",
        tts_model="kokoro-v1.0.onnx",
        tts_voices="voices-v1.0.bin",
        tts_voice="af_heart",
        tts_lang="en-us",
    ),
    "en-vi": PipelineConfig(
        direction="en-vi",
        src_lang="en", tgt_lang="vi",
        label="🇬🇧 English → 🇻🇳 Tiếng Việt",
        # ASR
        asr_type="moonshine",
        asr_dir="sherpa-onnx-moonshine-base-en-int8",
        # MT: VinAI translate-en2vi-v2 INT8 (fallback về opus-mt-en-vi nếu chưa có)
        mt_type="marian",
        mt_dir="opus-mt-en-vi",
        # TTS: Kokoro-Vietnamese ONNX (Thuần ONNX Runtime, ~168 MB RAM)
        tts_type="kokoro_vi",
        tts_dir="kokoro-vi",
        tts_model="kokoro_vi.int8.onnx",
        tts_voices="voicepacks/diem_trinh.pt",
        tts_voice="diem_trinh",
        tts_lang="vi",
    ),
    "vi-zh": PipelineConfig(
        direction="vi-zh",
        src_lang="vi", tgt_lang="zh",
        label="🇻🇳 Tiếng Việt → 🇨🇳 中文",
        # ASR
        asr_type="zipformer",
        asr_dir="sherpa-onnx-zipformer-vi-30M-int8-2026-02-09",
        # MT
        mt_type="marian",
        mt_dir="opus-mt-vi-zh",
        # TTS
        tts_type="kokoro",
        tts_dir="kokoro-zh",
        tts_model="kokoro-v1.1-zh.onnx",
        tts_voices="voices-v1.1-zh.bin",
        tts_voice="zf_001",
        tts_lang="zh",
    ),
    "zh-vi": PipelineConfig(
        direction="zh-vi",
        src_lang="zh", tgt_lang="vi",
        label="🇨🇳 中文 → 🇻🇳 Tiếng Việt",
        # ASR
        asr_type="paraformer",
        asr_dir="sherpa-onnx-paraformer-zh-int8",
        # MT
        mt_type="marian",
        mt_dir="opus-mt-zh-vi",
        # TTS: Kokoro-Vietnamese (anphunl/Kokoro-Vietnamese)
        tts_type="kokoro_vi",
        tts_dir="kokoro-vi",
        tts_model="kokoro_vi.pth",
        tts_voices="voicepacks/diem_trinh.pt",
        tts_voice="diem_trinh",
        tts_lang="vi",
    ),
}


# ════════════════════════════════════════════════════════════════════════════
# RAM TRACKER
# ════════════════════════════════════════════════════════════════════════════

def _get_ram_mb() -> float:
    """Đo RAM RSS (Resident Set Size) của tiến trình hiện tại (MB)."""
    try:
        import psutil
        proc = psutil.Process()
        return proc.memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


# ════════════════════════════════════════════════════════════════════════════
# MODEL WRAPPERS
# ════════════════════════════════════════════════════════════════════════════

class ModelStatus:
    UNLOADED = "unloaded"
    LOADING  = "loading"
    READY    = "ready"
    ERROR    = "error"


# ─── ASR Wrapper ─────────────────────────────────────────────────────────────

class ASRWrapper:
    """Bọc các engine ASR: Zipformer (VI), Moonshine (EN), Paraformer (ZH)."""

    def __init__(self, cfg: PipelineConfig):
        self.cfg     = cfg
        self.status  = ModelStatus.UNLOADED
        self.error   = ""
        self.load_ms = 0.0
        self._model  = None
        self._lock   = threading.Lock()

    def load(self) -> bool:
        with self._lock:
            if self.status == ModelStatus.READY:
                return True
            self.status = ModelStatus.LOADING

        t0 = time.perf_counter()
        try:
            asr_type = self.cfg.asr_type
            model_dir = MODELS_DIR / self.cfg.asr_dir

            if asr_type == "zipformer":
                self._model = self._load_zipformer(model_dir)
            elif asr_type == "moonshine":
                self._model = self._load_moonshine(model_dir)
            elif asr_type == "paraformer":
                self._model = self._load_paraformer(model_dir)
            else:
                raise ValueError(f"Unknown ASR type: {asr_type}")

            self.load_ms = (time.perf_counter() - t0) * 1000
            self.status  = ModelStatus.READY
            log.info(f"ASR [{asr_type}] ready — {self.load_ms:.0f} ms")
            return True

        except Exception as e:
            self.error  = str(e)
            self.status = ModelStatus.ERROR
            log.error(f"ASR load failed [{self.cfg.asr_type}]: {e}")
            return False

    def _load_zipformer(self, model_dir: Path) -> Any:
        import sherpa_onnx

        if not model_dir.exists():
            raise FileNotFoundError(
                f"Zipformer model dir không tồn tại: {model_dir.name}\n"
                "Chạy: bash demo/download_models.sh"
            )

        def _find(pat):
            matches = list(model_dir.glob(pat))
            return matches[0] if matches else None

        enc = _find("encoder*.int8.onnx") or _find("encoder*.onnx")
        dec = _find("decoder*.onnx")
        joi = _find("joiner*.onnx")
        tok = model_dir / "tokens.txt"

        if not (enc and dec and joi and tok.exists()):
            raise FileNotFoundError(f"Zipformer files không đầy đủ trong {model_dir}")

        return sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(enc), decoder=str(dec), joiner=str(joi),
            tokens=str(tok), num_threads=4,
            decoding_method="greedy_search", debug=False,
        )

    def _load_moonshine(self, model_dir: Path) -> Any:
        import sherpa_onnx

        # Check local folder
        if not model_dir.exists():
            # Thử tìm các folder moonshine khác trong MODELS_DIR
            candidates = list(MODELS_DIR.glob("*moonshine*"))
            if candidates:
                model_dir = candidates[0]

        if not model_dir.exists():
            raise FileNotFoundError(
                f"Moonshine ASR model không tồn tại trong: {model_dir.name}\n"
                "Chạy: bash demo/download_models.sh --pipeline en-vi"
            )

        def _find_file(pat):
            m = list(model_dir.glob(pat))
            return str(m[0]) if m else ""

        prep = _find_file("preprocess*.onnx")
        enc  = _find_file("encode*.int8.onnx") or _find_file("encode*.onnx")
        udec = _find_file("uncached_decode*.int8.onnx") or _find_file("uncached_decode*.onnx")
        cdec = _find_file("cached_decode*.int8.onnx") or _find_file("cached_decode*.onnx")
        tok  = _find_file("tokens.txt")

        if not (prep and enc and udec and cdec and tok):
            raise FileNotFoundError(f"Moonshine ONNX files không đầy đủ trong {model_dir}")

        log.info(f"Loading Moonshine ASR từ {model_dir.name}...")
        return sherpa_onnx.OfflineRecognizer.from_moonshine(
            preprocessor=prep,
            encoder=enc,
            uncached_decoder=udec,
            cached_decoder=cdec,
            tokens=tok,
            num_threads=4,
        )

    def _load_paraformer(self, model_dir: Path) -> Any:
        import sherpa_onnx

        if not model_dir.exists():
            candidates = list(MODELS_DIR.glob("*paraformer*"))
            if candidates:
                model_dir = candidates[0]

        if not model_dir.exists():
            raise FileNotFoundError(
                f"Paraformer Chinese ASR chưa tải: {model_dir.name}\n"
                "Chạy: bash demo/download_models.sh --pipeline zh-vi"
            )

        onnx_files = list(model_dir.glob("model*.int8.onnx")) or list(model_dir.glob("*.onnx"))
        tok = model_dir / "tokens.txt"
        if not (onnx_files and tok.exists()):
            raise FileNotFoundError(f"Paraformer files không đầy đủ trong {model_dir}")

        return sherpa_onnx.OfflineRecognizer.from_paraformer(
            paraformer=str(onnx_files[0]),
            tokens=str(tok),
            num_threads=4,
        )

    def transcribe(self, samples: np.ndarray, sample_rate: int = 16000) -> dict:
        if self.status != ModelStatus.READY or self._model is None:
            raise RuntimeError(f"ASR [{self.cfg.asr_type}] chưa sẵn sàng (status={self.status})")

        if sample_rate != 16000:
            samples = _resample(samples, sample_rate, 16000)
        samples = samples.astype(np.float32)
        duration = len(samples) / 16000

        t0 = time.perf_counter()
        stream = self._model.create_stream()
        stream.accept_waveform(16000, samples)
        self._model.decode_stream(stream)
        text = stream.result.text.strip()
        # Zipformer VI 30M trả về in hoa, lowercase trước khi qua MT và hiển thị
        if self.cfg.asr_type == "zipformer" or self.cfg.src_lang == "vi":
            text = text.lower()

        latency_ms = (time.perf_counter() - t0) * 1000
        rtf = latency_ms / max(duration * 1000, 1)

        log.info(f"ASR [{self.cfg.asr_type}]: '{text[:60]}' | dur={duration:.1f}s lat={latency_ms:.0f}ms RTF={rtf:.3f}")
        return {"text": text, "latency_ms": round(latency_ms, 1), "rtf": round(rtf, 3)}

    def unload(self):
        self._model = None
        self.status = ModelStatus.UNLOADED
        gc.collect()
        log.info(f"ASR [{self.cfg.asr_type}] unloaded")


# ─── MT Wrapper ──────────────────────────────────────────────────────────────

class MTWrapper:
    """Wraps MarianMT (Opus-MT) với auto-download từ Hugging Face Hub."""

    def __init__(self, cfg: PipelineConfig):
        self.cfg     = cfg
        self.status  = ModelStatus.UNLOADED
        self.error   = ""
        self.load_ms = 0.0
        self._model  = None
        self._tok    = None
        self._lock   = threading.Lock()

    def load(self) -> bool:
        with self._lock:
            if self.status == ModelStatus.READY:
                return True
            self.status = ModelStatus.LOADING

        t0 = time.perf_counter()
        try:
            if self.cfg.mt_type == "vinai":
                ok = self._load_vinai(t0)
            else:
                ok = self._load_marian(t0)
            return ok
        except Exception as e:
            self.error  = str(e)
            self.status = ModelStatus.ERROR
            log.error(f"MT load failed [{self.cfg.direction}]: {e}")
            return False

    def _load_vinai(self, t0: float) -> bool:
        """Load VinAI translate-en2vi-v2 INT8 (MBart) trực tiếp, không load FP32."""
        import torch
        from transformers import AutoTokenizer, MBartForConditionalGeneration
        import json

        mt_dir = MODELS_DIR / self.cfg.mt_dir
        full_int8_path = mt_dir / "full_model_int8.pt"
        meta_path = mt_dir / "quantize_meta.json"

        # Trường hợp 1: Đã có model INT8 standalone (Chuẩn nhất - Không đụng đến FP32)
        if full_int8_path.exists():
            log.info(f"Loading VinAI EN→VI INT8 trực tiếp từ {full_int8_path.name} (Zero FP32 Overhead)...")
            self._tok = AutoTokenizer.from_pretrained(str(mt_dir), src_lang="en_XX")
            self._model = torch.load(str(full_int8_path), weights_only=False)
            self._model.eval()
            self._vinai_tgt_token = self._tok.convert_tokens_to_ids("vi_VN")
            self._mt_backend = "vinai_int8"
            self.load_ms = (time.perf_counter() - t0) * 1000
            self.status = ModelStatus.READY
            log.info(f"VinAI EN→VI INT8 ready (Direct INT8) — {self.load_ms:.0f} ms")
            return True

        # Trường hợp 2: Fallback nếu chưa có INT8
        log.warning(f"VinAI INT8 chưa có tại {mt_dir} — fallback về opus-mt-en-vi")
        self.cfg = self.cfg.__class__(
            **{**self.cfg.__dict__, "mt_type": "marian", "mt_dir": "opus-mt-en-vi"}
        )
        return self._load_marian(t0)

    def _load_marian(self, t0: float) -> bool:
        """Load MarianMT (Opus-MT) — local hoặc tải từ HuggingFace Hub."""
        from transformers import MarianMTModel, MarianTokenizer

        mt_dir = MODELS_DIR / self.cfg.mt_dir
        src = str(mt_dir) if mt_dir.exists() and (mt_dir / "config.json").exists() else None

        if src is None:
            hf_id_map = {
                "opus-mt-vi-en": "Helsinki-NLP/opus-mt-vi-en",
                "opus-mt-en-vi": "Helsinki-NLP/opus-mt-en-vi",
                "opus-mt-vi-zh": "Helsinki-NLP/opus-mt-vi-zh",
                "opus-mt-zh-vi": "Helsinki-NLP/opus-mt-zh-vi",
            }
            src = hf_id_map.get(self.cfg.mt_dir, f"Helsinki-NLP/{self.cfg.mt_dir}")

        log.info(f"Loading Marian MT [{self.cfg.direction}] từ: {src}")
        self._tok   = MarianTokenizer.from_pretrained(src)
        self._model = MarianMTModel.from_pretrained(src)
        self._model.eval()
        self._mt_backend = "marian"

        if not mt_dir.exists():
            mt_dir.mkdir(parents=True, exist_ok=True)
            self._tok.save_pretrained(str(mt_dir))
            self._model.save_pretrained(str(mt_dir))

        self.load_ms = (time.perf_counter() - t0) * 1000
        self.status  = ModelStatus.READY
        log.info(f"MT [{self.cfg.direction}] ready — {self.load_ms:.0f} ms")
        return True

    def translate(self, text: str) -> dict:
        if self.status != ModelStatus.READY or self._model is None:
            raise RuntimeError(f"MT [{self.cfg.direction}] chưa sẵn sàng (status={self.status})")
        if not text.strip():
            return {"text": "", "latency_ms": 0}

        # Lowercase tiếng Việt trước khi dịch — Marian nhạy cảm hơn với chữ hoa
        if self.cfg.src_lang == "vi":
            text = text.lower()

        import torch
        backend = getattr(self, "_mt_backend", "marian")
        t0 = time.perf_counter()

        if backend == "vinai_int8":
            # VinAI MBart: cần decoder_start_token_id = vi_VN
            inputs = self._tok(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    decoder_start_token_id=self._vinai_tgt_token,
                    num_beams=4,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.3,
                    max_length=512,
                    early_stopping=True,
                )
            translated = self._tok.batch_decode(outputs, skip_special_tokens=True)[0]
        else:
            # MarianMT / Opus-MT: Tự động tách câu nếu có nhiều câu (tránh bị nuốt câu sau dấu hỏi/chấm)
            import re
            raw_sentences = [s.strip() for s in re.split(r'([.?!]+)', text) if s.strip()]
            sentences = []
            i = 0
            while i < len(raw_sentences):
                s = raw_sentences[i]
                if i + 1 < len(raw_sentences) and re.match(r'^[.?!]+$', raw_sentences[i+1]):
                    s += raw_sentences[i+1]
                    i += 1
                sentences.append(s)
                i += 1

            if len(sentences) <= 1:
                inputs = self._tok(text, return_tensors="pt", truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = self._model.generate(
                        **inputs,
                        max_length=512,
                        num_beams=4,
                        no_repeat_ngram_size=3,
                        repetition_penalty=1.3,
                        early_stopping=True,
                    )
                translated = self._tok.decode(outputs[0], skip_special_tokens=True)
            else:
                # Batch dịch từng câu rồi ghép lại
                inputs = self._tok(sentences, return_tensors="pt", padding=True, truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = self._model.generate(
                        **inputs,
                        max_length=512,
                        num_beams=4,
                        no_repeat_ngram_size=3,
                        repetition_penalty=1.3,
                        early_stopping=True,
                    )
                translated_parts = [self._tok.decode(o, skip_special_tokens=True) for o in outputs]
                translated = " ".join(translated_parts)

        latency_ms = (time.perf_counter() - t0) * 1000
        log.info(f"MT [{backend}]: '{text[:40]}' → '{translated[:40]}' | {latency_ms:.0f}ms")
        return {"text": translated, "latency_ms": round(latency_ms, 1)}

    def unload(self):
        self._model = None
        self._tok   = None
        self.status = ModelStatus.UNLOADED
        gc.collect()
        log.info(f"MT [{self.cfg.direction}] unloaded")


# ─── TTS Wrapper ─────────────────────────────────────────────────────────────

class TTSWrapper:
    """Wraps Kokoro ONNX TTS và Piper VITS TTS."""

    def __init__(self, cfg: PipelineConfig):
        self.cfg        = cfg
        self.status     = ModelStatus.UNLOADED
        self.error      = ""
        self.load_ms    = 0.0
        self._engine    = None  # kokoro instance hoặc sherpa_onnx.OfflineTts
        self._backend   = "kokoro" # "kokoro" | "piper"
        self._lock      = threading.Lock()

    def load(self) -> bool:
        with self._lock:
            if self.status == ModelStatus.READY:
                return True
            self.status = ModelStatus.LOADING

        t0 = time.perf_counter()
        try:
            tts_type = self.cfg.tts_type

            if tts_type == "kokoro_vi" or self.cfg.tgt_lang == "vi":
                try:
                    self._load_kokoro_vi()
                except Exception as vi_err:
                    log.warning(f"Kokoro-Vietnamese không tải được ({vi_err}). Thử Piper VITS...")
                    self._load_piper()
            elif tts_type == "piper":
                self._load_piper()
            else:
                # Kokoro EN / ZH với fallback sang Piper
                try:
                    self._load_kokoro()
                except Exception as kokoro_err:
                    log.warning(f"Kokoro TTS không tải được ({kokoro_err}). Thử Piper VITS...")
                    self._load_piper()

            self.load_ms = (time.perf_counter() - t0) * 1000
            self.status  = ModelStatus.READY
            log.info(f"TTS [{self.cfg.tgt_lang} - {self._backend}] ready — {self.load_ms:.0f} ms")
            return True

        except Exception as e:
            self.error  = str(e)
            self.status = ModelStatus.ERROR
            log.error(f"TTS load failed [{self.cfg.tgt_lang}]: {e}")
            return False

    def _load_kokoro_vi(self):
        """Load Kokoro-Vietnamese — Thuần ONNX Runtime (INT8/FP32), fallback về PyTorch."""
        voice = self.cfg.tts_voice or "diem_trinh"
        tts_dir = MODELS_DIR / self.cfg.tts_dir
        if not tts_dir.exists():
            tts_dir = MODELS_DIR / "kokoro-vi"

        int8_onnx = tts_dir / "kokoro_vi.int8.onnx"
        fp32_onnx = tts_dir / "kokoro_vi.onnx"
        voicepack_path = tts_dir / "voicepacks" / f"{voice}.pt"
        config_path = tts_dir / "config.json"

        # 1. Ưu tiên Kokoro Vietnamese ONNX Runtime (Siêu nhẹ, ~168 MB RAM)
        onnx_file = int8_onnx if int8_onnx.exists() else (fp32_onnx if fp32_onnx.exists() else None)
        if onnx_file is not None and voicepack_path.exists() and config_path.exists():
            from kokoro_vietnamese.onnx_cli import KokoroVietnameseONNX
            log.info(f"Loading Kokoro-Vietnamese ONNX ({onnx_file.name}) — voice: {voice}...")
            self._engine = KokoroVietnameseONNX(
                onnx_path=onnx_file,
                voicepack_path=voicepack_path,
                config_path=config_path,
                device="cpu",
            )
            self._backend = "kokoro_vi_onnx"
            return

        # 2. Fallback: PyTorch engine
        from kokoro_vietnamese import KokoroVietnamese
        log.info(f"Loading Kokoro-Vietnamese PyTorch — voice: {voice} (ONNX chưa sẵn sàng)...")
        self._engine  = KokoroVietnamese(device="cpu", voice=voice)
        self._backend = "kokoro_vi"

    def _load_kokoro(self):
        from kokoro_onnx import Kokoro

        tts_dir = MODELS_DIR / self.cfg.tts_dir
        int8_model = self.cfg.tts_model.replace(".onnx", ".int8.onnx")
        
        model_path = None
        for p in [tts_dir / self.cfg.tts_model, MODELS_DIR / "kokoro" / "kokoro-v1.0.onnx", tts_dir / int8_model, MODELS_DIR / "kokoro" / "kokoro-v1.0.int8.onnx"]:
            if p.exists():
                model_path = p
                break

        voices_path = None
        for p in [tts_dir / self.cfg.tts_voices, MODELS_DIR / "kokoro" / "voices-v1.0.bin"]:
            if p.exists():
                voices_path = p
                break

        if not (model_path and voices_path):
            raise FileNotFoundError(f"Kokoro model files không tìm thấy trong {tts_dir}")

        log.info(f"Loading Kokoro TTS: {model_path.name}")
        self._engine  = Kokoro(str(model_path), str(voices_path))
        self._backend = "kokoro"

    def _load_piper(self):
        import sherpa_onnx

        tts_dir = MODELS_DIR / self.cfg.tts_dir
        if not tts_dir.exists():
            candidates = list(MODELS_DIR.glob(f"*piper*{self.cfg.tgt_lang}*")) or list(MODELS_DIR.glob("*piper*"))
            if candidates:
                tts_dir = candidates[0]

        if not tts_dir.exists():
            raise FileNotFoundError(
                f"Piper TTS model không tồn tại: {tts_dir.name}\n"
                f"Chạy: bash demo/download_models.sh --pipeline {self.cfg.direction}"
            )

        onnx_file = None
        for p in [tts_dir / self.cfg.tts_model, tts_dir / f"{self.cfg.tts_model.replace('.int8.onnx', '.onnx')}"]:
            if p.exists():
                onnx_file = p
                break
        if onnx_file is None:
            onnx_files = list(tts_dir.glob("*.onnx"))
            if onnx_files:
                onnx_file = onnx_files[0]

        tokens_file = tts_dir / "tokens.txt"
        data_dir    = tts_dir / "espeak-ng-data"

        if not (onnx_file and tokens_file.exists()):
            raise FileNotFoundError(f"Piper files không đầy đủ trong {tts_dir}")

        log.info(f"Loading Piper VITS TTS [{self.cfg.tgt_lang}] từ: {tts_dir.name}")
        cfg = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=str(onnx_file),
                    tokens=str(tokens_file),
                    data_dir=str(data_dir) if data_dir.exists() else "",
                    noise_scale=0.667,
                    noise_scale_w=0.8,
                    length_scale=1.0,
                ),
            ),
            rule_fsts="",
            max_num_sentences=1,
        )
        self._engine  = sherpa_onnx.OfflineTts(cfg)
        self._backend = "piper"

    def synthesize(self, text: str, voice: str = None) -> dict:
        if self.status != ModelStatus.READY or self._engine is None:
            raise RuntimeError(f"TTS [{self.cfg.tgt_lang}] chưa sẵn sàng (status={self.status})")
        if not text.strip():
            raise ValueError("Empty text")

        t0 = time.perf_counter()

        if self._backend in ("kokoro_vi", "kokoro_vi_int8", "kokoro_vi_onnx"):
            audio, _ = self._engine.synthesize(text, speed=1.0)
            samples = np.array(audio, dtype=np.float32)
            sr = 24000
        elif self._backend == "kokoro":
            voice = voice or self.cfg.tts_voice
            samples, sr = self._engine.create(text, voice=voice, speed=1.0, lang=self.cfg.tts_lang)
            samples = np.array(samples, dtype=np.float32)
        else:
            # Piper VITS
            audio = self._engine.generate(text, sid=int(voice or self.cfg.tts_voice or 0), speed=1.0)
            samples = np.array(audio.samples, dtype=np.float32)
            sr = audio.sample_rate

        latency_ms   = (time.perf_counter() - t0) * 1000
        duration_sec = len(samples) / sr
        wav_bytes    = _to_wav_bytes(samples, sr)

        log.info(f"TTS [{self.cfg.tgt_lang} - {self._backend}]: '{text[:40]}' | dur={duration_sec:.2f}s lat={latency_ms:.0f}ms")
        return {
            "wav_bytes": wav_bytes, "duration_sec": round(duration_sec, 2),
            "latency_ms": round(latency_ms, 1), "sample_rate": sr,
        }

    def unload(self):
        self._engine = None
        self.status  = ModelStatus.UNLOADED
        gc.collect()
        log.info(f"TTS [{self.cfg.tgt_lang}] unloaded")


# ════════════════════════════════════════════════════════════════════════════
# PIPELINE MANAGER
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineMetrics:
    direction:     str
    asr_latency:   float = 0.0
    mt_latency:    float = 0.0
    tts_latency:   float = 0.0
    total_latency: float = 0.0
    ram_before_mb: float = 0.0
    ram_after_mb:  float = 0.0
    ram_delta_mb:  float = 0.0


class PipelineManager:
    """Điều phối nạp/gỡ và chạy pipeline S2S đa ngôn ngữ."""

    def __init__(self, default_direction: str = "vi-en"):
        self._current_direction: Optional[str] = None
        self._asr: Optional[ASRWrapper] = None
        self._mt:  Optional[MTWrapper]  = None
        self._tts: Optional[TTSWrapper] = None
        self._lock = threading.RLock()
        self._latency_history: list[PipelineMetrics] = []
        self._max_history = 20

    def switch(self, direction: str, preload: bool = True) -> bool:
        if direction not in PIPELINE_CONFIGS:
            log.error(f"Unknown direction: {direction}")
            return False

        with self._lock:
            if self._current_direction == direction and self.is_ready:
                log.info(f"Pipeline [{direction}] đã active và ready")
                return True

            log.info(f"Switching pipeline: {self._current_direction} → {direction}")
            self._unload_current()

            cfg = PIPELINE_CONFIGS[direction]
            self._asr = ASRWrapper(cfg)
            self._mt  = MTWrapper(cfg)
            self._tts = TTSWrapper(cfg)
            self._current_direction = direction

        if preload:
            return self._load_all()
        return True

    def _load_all(self) -> bool:
        if not self._asr:
            return False

        results = {}
        threads = []

        def _load(name, wrapper):
            results[name] = wrapper.load()

        for name, w in [("asr", self._asr), ("mt", self._mt), ("tts", self._tts)]:
            t = threading.Thread(target=_load, args=(name, w), daemon=True)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        ok = all(results.values())
        if ok:
            log.info(f"Pipeline [{self._current_direction}] READY ✅ | RAM: {_get_ram_mb():.0f} MB")
        else:
            failed = [k for k, v in results.items() if not v]
            log.error(f"Pipeline [{self._current_direction}] — Lỗi component: {failed}")
        return ok

    def _unload_current(self):
        for w in [self._asr, self._mt, self._tts]:
            if w is not None:
                try:
                    w.unload()
                except Exception as e:
                    log.warning(f"Lỗi unload: {e}")
        self._asr = None
        self._mt  = None
        self._tts = None
        gc.collect()

    def run(self, samples: np.ndarray, sample_rate: int = 16000) -> dict:
        with self._lock:
            if not self._current_direction:
                raise RuntimeError("Chưa chọn pipeline.")
            for name, w in [("ASR", self._asr), ("MT", self._mt), ("TTS", self._tts)]:
                if w is None or w.status != ModelStatus.READY:
                    err = getattr(w, "error", "")
                    raise RuntimeError(f"{name} chưa sẵn sàng ({getattr(w, 'status', 'None')}) {err}")

        ram_before = _get_ram_mb()
        t_total    = time.perf_counter()

        asr_r = self._asr.transcribe(samples, sample_rate)
        src_text = asr_r["text"]
        if not src_text:
            raise ValueError("ASR trả về text rỗng — vui lòng nói rõ hơn")

        mt_r = self._mt.translate(src_text)
        tgt_text = mt_r["text"]

        tts_r = self._tts.synthesize(tgt_text)

        total_ms  = (time.perf_counter() - t_total) * 1000
        ram_after = _get_ram_mb()

        metrics = PipelineMetrics(
            direction=self._current_direction,
            asr_latency=asr_r["latency_ms"],
            mt_latency=mt_r["latency_ms"],
            tts_latency=tts_r["latency_ms"],
            total_latency=round(total_ms, 1),
            ram_before_mb=ram_before,
            ram_after_mb=ram_after,
            ram_delta_mb=round(ram_after - ram_before, 1),
        )
        self._latency_history.append(metrics)
        if len(self._latency_history) > self._max_history:
            self._latency_history.pop(0)

        import base64
        return {
            "direction": self._current_direction,
            "stages": {
                "asr": {"text": src_text, "latency_ms": asr_r["latency_ms"], "rtf": asr_r["rtf"]},
                "mt":  {"text": tgt_text, "latency_ms": mt_r["latency_ms"]},
                "tts": {
                    "duration_sec": tts_r["duration_sec"],
                    "latency_ms":   tts_r["latency_ms"],
                    "sample_rate":  tts_r["sample_rate"],
                },
            },
            "total_latency_ms": metrics.total_latency,
            "metrics": {
                "ram_before_mb": round(ram_before, 1),
                "ram_after_mb":  round(ram_after, 1),
                "ram_delta_mb":  metrics.ram_delta_mb,
            },
            "audio_b64":  base64.b64encode(tts_r["wav_bytes"]).decode(),
            "audio_mime": "audio/wav",
        }

    def get_status(self) -> dict:
        def _w_status(w):
            if w is None:
                return {"status": "unloaded", "error": "", "load_ms": 0}
            return {"status": w.status, "error": w.error, "load_ms": round(w.load_ms, 0)}

        return {
            "active_direction": self._current_direction,
            "available_pipelines": {k: v.label for k, v in PIPELINE_CONFIGS.items()},
            "ram_mb": round(_get_ram_mb(), 1),
            "models": {
                "asr": _w_status(self._asr),
                "mt":  _w_status(self._mt),
                "tts": _w_status(self._tts),
            },
        }

    def get_metrics(self) -> dict:
        if not self._latency_history:
            return {"count": 0}
        n = len(self._latency_history)
        avg = lambda key: round(sum(getattr(m, key) for m in self._latency_history) / n, 1)
        return {
            "count": n,
            "avg_asr_ms":   avg("asr_latency"),
            "avg_mt_ms":    avg("mt_latency"),
            "avg_tts_ms":   avg("tts_latency"),
            "avg_total_ms": avg("total_latency"),
            "ram_mb":       round(_get_ram_mb(), 1),
        }

    @property
    def active_direction(self) -> Optional[str]:
        return self._current_direction

    @property
    def is_ready(self) -> bool:
        return all(
            w is not None and w.status == ModelStatus.READY
            for w in [self._asr, self._mt, self._tts]
        )


# ════════════════════════════════════════════════════════════════════════════
# UTILS
# ════════════════════════════════════════════════════════════════════════════

def _resample(samples: np.ndarray, orig: int, target: int) -> np.ndarray:
    try:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(orig, target)
        return resample_poly(samples, target // g, orig // g).astype(np.float32)
    except ImportError:
        n_out = int(len(samples) * target / orig)
        return np.interp(
            np.linspace(0, len(samples) - 1, n_out),
            np.arange(len(samples)), samples,
        ).astype(np.float32)


def _to_wav_bytes(samples: np.ndarray, sr: int) -> bytes:
    buf  = io.BytesIO()
    int16 = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(int16.tobytes())
    return buf.getvalue()
