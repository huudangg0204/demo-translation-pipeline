# OneVoice AI — Speech-to-Speech Demo (VI → EN)

Ứng dụng demo dịch giọng nói Tiếng Việt sang Tiếng Anh qua pipeline 3 giai đoạn:

```
Audio (VI) ──► ASR ──► MT ──► TTS ──► Audio (EN)
```

| Giai đoạn | Model | Mô tả |
|---|---|---|
| **ASR** | `sherpa-onnx-zipformer-vi-30M-int8-2026-02-09` | Nhận dạng giọng nói tiếng Việt (offline) |
| **MT** | `Helsinki-NLP/opus-mt-vi-en` | Dịch Tiếng Việt → Tiếng Anh (MarianMT) |
| **TTS** | `Kokoro-82M ONNX` | Tổng hợp giọng nói tiếng Anh (chất lượng cao) |

---

## Cài đặt nhanh

### 1. Tạo môi trường ảo (khuyến nghị)
```bash
cd demo
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Cài thư viện
```bash
pip install -r requirements.txt
```

### 3. Tải models
```bash
bash download_models.sh
```

### 4. Chạy server
```bash
uvicorn app:app --reload --port 8000
```

### 5. Mở demo
Mở trình duyệt tại: **http://localhost:8000**

---

## Tính năng

- **🎤 Ghi âm trực tiếp**: VAD hands-free — tự động phát hiện giọng nói và dừng sau 1.5s im lặng
- **📁 Upload file**: Kéo thả hoặc chọn file audio (WAV, MP3, OGG, FLAC, M4A)
- **📊 Pipeline visualization**: Hiển thị từng giai đoạn ASR → MT → TTS với latency
- **🔊 Phát audio ngay**: Output tiếng Anh tự động phát sau khi xử lý xong
- **📈 Waveform realtime**: Hiển thị sóng âm khi ghi âm

---

## API Endpoints

| Method | Path | Mô tả |
|---|---|---|
| `GET` | `/` | Giao diện web |
| `GET` | `/api/status` | Trạng thái các model |
| `POST` | `/api/pipeline` | Full pipeline (file upload) |
| `POST` | `/api/asr` | Chỉ ASR |
| `POST` | `/api/mt` | Chỉ MT (JSON body) |
| `POST` | `/api/tts` | Chỉ TTS (JSON body) |
| `WS` | `/ws/pipeline` | WebSocket real-time pipeline |

---

## Cấu trúc thư mục

```
demo/
├── app.py               # FastAPI backend (toàn bộ logic)
├── requirements.txt     # Thư viện Python
├── download_models.sh   # Script tải model
├── README.md            # File này
├── models/              # Models được tải vào đây
│   ├── sherpa-onnx-zipformer-vi-30M-int8-2026-02-09/
│   ├── opus-mt-vi-en/
│   └── kokoro/
└── static/
    └── index.html       # Giao diện web
```

---

## Yêu cầu hệ thống

- Python 3.10+
- RAM ≥ 4 GB (ASR ~200MB + MT ~300MB + TTS ~300MB)
- Microphone (cho chế độ ghi âm)
- Internet (để tải model lần đầu)

---

## Lưu ý

- Các model được load **song song** khi server khởi động (background threads)
- **Latency thực tế** (lần đầu tiên sau khi models load):
  - ASR: ~300–800 ms (tuỳ độ dài câu)
  - MT: ~100–300 ms
  - TTS: ~200–600 ms
  - **Tổng: ~0.8–2.0 giây** (không tính load model)
