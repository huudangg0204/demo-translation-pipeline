# OneVoice AI — Plug-and-Play Speech-to-Speech Translation Pipeline

> **Hệ thống dịch giọng nói trực tiếp (Speech-to-Speech - S2S) đa ngôn ngữ, thiết kế theo kiến trúc Plug-and-Play (cắm-và-chạy), tối ưu hóa suy luận nhẹ, độ trễ thấp và tiết kiệm bộ nhớ RAM.**

---

## 📌 Tổng quan kiến trúc Pipeline

Pipeline xử lý âm thanh đầu vào qua 3 tầng độc lập được chuẩn hóa:

```
                  ┌──────────────┐      ┌─────────────┐      ┌─────────────┐
Audio Input ────► │  ASR Engine  │ ───► │  MT Engine  │ ───► │  TTS Engine │ ────► Audio Output
(WAV/Mic Stream)  │ (Speech2Text)│ Text │(Translation)│ Text │(Text2Speech)│       (WAV 24kHz)
                  └──────────────┘      └─────────────┘      └─────────────┘
                         │                     │                    │
                         └─────────────────────┴────────────────────┘
                                               │
                                    ┌──────────────────────┐
                                    │   Pipeline Manager   │
                                    │  (Lazy Load & Swapp) │
                                    └──────────────────────┘
```

### 🌐 Các luồng dịch hỗ trợ sẵn

| Luồng | Chiều dịch | ASR Engine | MT Engine | TTS Engine | RAM tiêu thụ |
|:---|:---|:---|:---|:---|:---|
| `vi-en` *(Mặc định)* | 🇻🇳 Tiếng Việt → 🇬🇧 English | Zipformer 30M INT8 | Opus-MT (`Helsinki-NLP/opus-mt-vi-en`) | Kokoro EN ONNX (`af_heart`) | ~350 MB |
| `en-vi` | 🇬🇧 English → 🇻🇳 Tiếng Việt | Moonshine Base INT8 | Opus-MT / VinAI translate-en2vi INT8 | Kokoro-VI ONNX (`diem_trinh`) / Piper | ~450 MB |
| `vi-zh` | 🇻🇳 Tiếng Việt → 🇨🇳 中文 | Zipformer 30M INT8 | Opus-MT (`Helsinki-NLP/opus-mt-vi-zh`) | Kokoro ZH ONNX (`zf_001`) | ~380 MB |
| `zh-vi` | 🇨🇳 中文 → 🇻🇳 Tiếng Việt | Paraformer ZH INT8 | Opus-MT (`Helsinki-NLP/opus-mt-zh-vi`) | Kokoro-VI ONNX (`diem_trinh`) / Piper | ~480 MB |

---

## 🧩 Cơ chế Plug-and-Play (Cắm-và-Chạy)

Hệ thống được thiết kế theo nguyên lý **Decoupled Architecture** và **Config-driven**, giúp việc hoán đổi, thử nghiệm và tích hợp model mới diễn ra nhanh chóng mà không làm ảnh hưởng đến luồng chạy chung:

1. **Chuẩn hóa Interface qua Wrappers (`ASRWrapper`, `MTWrapper`, `TTSWrapper`)**:
   - Mọi engine ASR đều triển khai interface `transcribe(samples, sample_rate) -> dict`.
   - Mọi engine MT đều triển khai interface `translate(text) -> dict`.
   - Mọi engine TTS đều triển khai interface `synthesize(text, voice) -> dict`.
   - Mỗi wrapper tự quản lý vòng đời độc lập qua `load()` và `unload()`.

2. **Declarative Configuration (`PipelineConfig` & `PIPELINE_CONFIGS`)**:
   - Toàn bộ tham số của một luồng dịch (loại model, đường dẫn trọng số ONNX/PT, voice id, ngôn ngữ) được khai báo tập trung trong `PIPELINE_CONFIGS`.

3. **Lazy Loading on Demand & Auto Unload**:
   - **Lazy Load**: Model chỉ được nạp vào RAM khi người dùng chuyển sang luồng dịch tương ứng.
   - **Auto Unload & GC**: Khi chuyển luồng (Switch Pipeline), hệ thống tự động giải phóng toàn bộ model cũ và gọi `gc.collect()`, tránh tình trạng tràn RAM (OOM) khi có nhiều cặp ngôn ngữ.
   - **Parallel Loading**: Quá trình nạp 3 thành phần ASR, MT, TTS chạy song song đa luồng (multi-threading), rút ngắn 60% thời gian khởi tạo.

4. **Multi-tier Fallback**:
   - Tự động chuyển đổi giữa các engine thay thế nếu engine ưu tiên chưa tải trọng số (ví dụ: Kokoro-VI ONNX → Kokoro PyTorch → Piper VITS; hoặc VinAI INT8 → Opus-MT).

---

## 🛠️ Hướng dẫn Developer: Thay đổi & Mở rộng Model trong Pipeline

### Trường hợp 1: Đổi model/checkpoint có sẵn hoặc đổi Voice

Để đổi voice hoặc đổi đường dẫn model trong pipeline hiện có, chỉ cần chỉnh sửa dictionary `PIPELINE_CONFIGS` trong [`pipeline_manager.py`](file:///home/huudang/Documents/projects/OneVoice_Ai/demo/pipeline_manager.py):

```python
PIPELINE_CONFIGS["vi-en"] = PipelineConfig(
    direction="vi-en",
    src_lang="vi", tgt_lang="en",
    label="🇻🇳 Tiếng Việt → 🇬🇧 English",
    # Đổi thư mục ASR
    asr_type="zipformer",
    asr_dir="sherpa-onnx-zipformer-vi-30M-int8-2026-02-09",
    # Đổi model MT (Hugging Face ID hoặc folder local trong demo/models)
    mt_type="marian",
    mt_dir="opus-mt-vi-en",
    # Đổi giọng TTS hoặc model TTS
    tts_type="kokoro",
    tts_dir="kokoro",
    tts_model="kokoro-v1.0.onnx",
    tts_voices="voices-v1.0.bin",
    tts_voice="am_adam",  # Đổi voice thành giọng nam (ví dụ: am_adam, af_bella,...)
    tts_lang="en-us",
)
```

---

### Trường hợp 2: Thêm một Model / Engine mới vào Wrapper

Nếu bạn muốn tích hợp một thuật toán/engine hoàn toàn mới (ví dụ: Whisper ASR, NLLB MT, hay MeloTTS):

#### 1. Thêm ASR Engine mới vào `ASRWrapper`:
Trong [`pipeline_manager.py`](file:///home/huudang/Documents/projects/OneVoice_Ai/demo/pipeline_manager.py) > `ASRWrapper`:
- Thêm phương thức nạp model `_load_whisper(self, model_dir: Path)`:
  ```python
  def _load_whisper(self, model_dir: Path):
      import sherpa_onnx
      return sherpa_onnx.OfflineRecognizer.from_whisper(
          encoder=str(model_dir / "tiny-encoder.int8.onnx"),
          decoder=str(model_dir / "tiny-decoder.int8.onnx"),
          tokens=str(model_dir / "tiny-tokens.txt"),
          language=self.cfg.src_lang,
          num_threads=4,
      )
  ```
- Trong hàm `load()`, thêm nhánh kiểm tra `elif asr_type == "whisper": self._model = self._load_whisper(model_dir)`.

#### 2. Thêm MT Engine mới vào `MTWrapper`:
Trong [`pipeline_manager.py`](file:///home/huudang/Documents/projects/OneVoice_Ai/demo/pipeline_manager.py) > `MTWrapper`:
- Thêm loader (ví dụ: NLLB, CTranslate2, vLLM hoặc Custom ONNX Model) trong `_load_xxx()`.
- Trong hàm `translate(self, text: str)`, gọi hàm suy luận tương ứng và trả về `{"text": translated_text, "latency_ms": round(latency_ms, 1)}`.

#### 3. Thêm TTS Engine mới vào `TTSWrapper`:
Trong [`pipeline_manager.py`](file:///home/huudang/Documents/projects/OneVoice_Ai/demo/pipeline_manager.py) > `TTSWrapper`:
- Thêm loader engine trong `_load_xxx()`.
- Trong hàm `synthesize(self, text: str, voice: str = None)`, tạo mảng âm thanh `samples` (`np.float32`), tần số lấy mẫu `sr`, sau đó chuyển thành WAV bytes thông qua `_to_wav_bytes(samples, sr)`.

---

### Trường hợp 3: Thêm một Hướng dịch mới (Language Pair)

Ví dụ thêm luồng **Tiếng Nhật → Tiếng Việt (`ja-vi`)**:

1. **Định nghĩa trong `PIPELINE_CONFIGS`** ([`pipeline_manager.py`](file:///home/huudang/Documents/projects/OneVoice_Ai/demo/pipeline_manager.py)):
   ```python
   PIPELINE_CONFIGS["ja-vi"] = PipelineConfig(
       direction="ja-vi",
       src_lang="ja", tgt_lang="vi",
       label="🇯🇵 日本語 → 🇻🇳 Tiếng Việt",
       asr_type="paraformer", # hoặc sense_voice / whisper
       asr_dir="sherpa-onnx-paraformer-ja-int8",
       mt_type="marian",
       mt_dir="opus-mt-ja-vi",
       tts_type="kokoro_vi",
       tts_dir="kokoro-vi",
       tts_model="kokoro_vi.int8.onnx",
       tts_voices="voicepacks/diem_trinh.pt",
       tts_voice="diem_trinh",
       tts_lang="vi",
   )
   ```

2. **Thêm script tải model vào [`download_models.sh`](file:///home/huudang/Documents/projects/OneVoice_Ai/demo/download_models.sh)** để team members có thể tự động tải trọng số.

---

## 🚀 Hướng dẫn Cài đặt & Chạy Demo

### 1. Khởi tạo môi trường ảo
```bash
cd demo
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Cài đặt thư viện phụ thuộc
```bash
pip install -r requirements.txt
```

### 3. Tải Model Weights
```bash
# Tải pipeline mặc định (vi-en ~350MB):
bash download_models.sh

# Hoặc tải một luồng cụ thể:
bash download_models.sh --pipeline en-vi
bash download_models.sh --pipeline vi-zh
bash download_models.sh --pipeline zh-vi

# Hoặc tải toàn bộ tất cả model của 4 luồng:
bash download_models.sh --all
```

### 4. Khởi chạy Server
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Truy cập giao diện Web tại: **`http://localhost:8000`**

---

## 📡 API Endpoints Reference

### Quản lý Pipeline & Metrics
- `GET /api/status`: Lấy trạng thái hoạt động của pipeline hiện tại, các model đang load và RAM RSS.
- `GET /api/pipelines`: Lấy danh sách toàn bộ các luồng dịch sẵn có.
- `POST /api/pipeline/select`: Chuyển đổi luồng dịch active (Body: `{"direction": "en-vi"}`).
- `GET /api/metrics`: Thống kê độ trễ trung bình của từng tầng ASR, MT, TTS qua các lượt chạy.

### Xử lý Âm thanh & Dịch thuật
- `POST /api/pipeline`: Gửi file audio qua form-data (`file`), chạy toàn bộ pipeline S2S và nhận kết quả văn bản + audio WAV (Base64).
- `POST /api/asr`: Chạy riêng tầng ASR (Input: file audio, Output: văn bản nhận diện).
- `POST /api/mt`: Chạy riêng tầng MT (JSON Body: `{"text": "xin chào"}`).
- `POST /api/tts`: Chạy riêng tầng TTS (JSON Body: `{"text": "hello", "voice": "af_heart"}`).
- `WS /ws/pipeline`: WebSocket truyền nhận âm thanh real-time độ trễ thấp.

---

## 📂 Cấu trúc Thư mục

```
demo/
├── app.py                # FastAPI Server, REST API & WebSocket Handler
├── pipeline_manager.py   # Core Plug-and-Play Engine, Wrappers, Model Lifecycle
├── download_models.sh    # Script tự động hóa tải & giải nén trọng số
├── requirements.txt      # Danh sách thư viện cần thiết
├── README.md             # Tài liệu dự án & Hướng dẫn kỹ thuật
├── static/
│   └── index.html        # Giao diện Web SPA (Real-time Waveform, VAD, S2S UI)
└── models/               # Thư mục lưu trữ trọng số (được ignore bởi Git)
    ├── sherpa-onnx-zipformer-vi-30M-int8-2026-02-09/
    ├── sherpa-onnx-moonshine-base-en-int8/
    ├── opus-mt-vi-en/
    ├── opus-mt-en-vi/
    ├── kokoro/
    └── kokoro-vi/
```

---

## ⚡ Hiệu năng Thực tế (Benchmark trên CPU phổ thông)

- **ASR Latency**: ~250 – 500 ms (RTF ~0.08–0.15 với Zipformer INT8).
- **MT Latency**: ~80 – 200 ms (MarianMT / VinAI MBart INT8).
- **TTS Latency**: ~150 – 400 ms (Kokoro ONNX Runtime).
- **Tổng độ trễ End-to-End**: **~0.6 – 1.2 giây** (Nhanh hơn tốc độ nói tự nhiên).
- **RAM Footprint**: ~350MB – 500MB cho 1 luồng hoạt động (nhờ cơ chế giải phóng bộ nhớ tự động).
