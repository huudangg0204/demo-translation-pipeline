"""
demo/app.py — OneVoice AI Speech-to-Speech Demo (Plug-and-Play)
================================================================
Pipeline: Audio → ASR → MT → TTS → Audio

Hỗ trợ 4 luồng dịch:
  VI → EN : Zipformer 30M + Opus-MT vi-en + Kokoro EN
  EN → VI : Moonshine Base INT8 + Opus-MT en-vi + Kokoro VI
  VI → ZH : Zipformer 30M + Opus-MT vi-zh + Kokoro ZH
  ZH → VI : Paraformer ZH INT8 + Opus-MT zh-vi + Kokoro VI

Run:
  cd demo
  uvicorn app:app --reload --port 8000
"""

import asyncio
import base64
import io
import json
import logging
import threading
import time
import wave
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pipeline_manager import (
    PipelineManager,
    PIPELINE_CONFIGS,
    ModelStatus,
    _resample,
    _to_wav_bytes,
    _get_ram_mb,
)

# ─── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("onevoice.app")

# ─── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="OneVoice AI — Plug-and-Play S2S Demo",
    description="Speech-to-Speech đa ngôn ngữ (VI ↔ EN, VI ↔ ZH)",
    version="2.0.0",
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ─── Global Pipeline Manager ────────────────────────────────────────────────
pm = PipelineManager()
_switch_lock = threading.Lock()


# ════════════════════════════════════════════════════════════════════════════
# STARTUP
# ════════════════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup_event():
    log.info("OneVoice AI v2.0 starting — Plug-and-Play Pipeline")

    def _preload():
        # Load pipeline mặc định VI → EN khi khởi động
        pm.switch("vi-en", preload=True)
        log.info("Default pipeline [vi-en] loaded ✅")

    threading.Thread(target=_preload, daemon=True).start()


# ════════════════════════════════════════════════════════════════════════════
# REST ENDPOINTS
# ════════════════════════════════════════════════════════════════════════════

# ─── Frontend ────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = STATIC_DIR / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h1>index.html not found</h1>", status_code=404)
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


# ─── Status ──────────────────────────────────────────────────────────────────
@app.get("/api/status")
async def get_status():
    """Trạng thái pipeline hiện tại + danh sách pipeline khả dụng."""
    return pm.get_status()


# ─── Metrics ─────────────────────────────────────────────────────────────────
@app.get("/api/metrics")
async def get_metrics():
    """Thống kê latency và RAM (rolling average 20 lần gần nhất)."""
    return pm.get_metrics()


# ─── Danh sách pipelines ─────────────────────────────────────────────────────
@app.get("/api/pipelines")
async def list_pipelines():
    """Danh sách 4 luồng dịch và trạng thái load."""
    status = pm.get_status()
    return {
        "active": status["active_direction"],
        "ram_mb": status["ram_mb"],
        "pipelines": [
            {
                "direction": k,
                "label":     v.label,
                "src_lang":  v.src_lang,
                "tgt_lang":  v.tgt_lang,
                "asr":       v.asr_type,
                "mt":        v.mt_dir,
                "tts":       f"kokoro-{v.tgt_lang}",
                "is_active": k == status["active_direction"],
            }
            for k, v in PIPELINE_CONFIGS.items()
        ],
    }


# ─── Chọn pipeline ───────────────────────────────────────────────────────────
class SelectPipelineIn(BaseModel):
    direction: str

@app.post("/api/pipeline/select")
async def select_pipeline(body: SelectPipelineIn):
    """
    Chuyển đổi luồng dịch active.
    Hệ thống sẽ unload pipeline cũ và load pipeline mới (có thể mất 5–30s).
    """
    if body.direction not in PIPELINE_CONFIGS:
        raise HTTPException(
            400,
            f"Invalid direction '{body.direction}'. "
            f"Chọn từ: {list(PIPELINE_CONFIGS.keys())}"
        )

    if pm.active_direction == body.direction:
        return {"status": "already_active", "direction": body.direction}

    # Switch trong background thread
    def _do_switch():
        with _switch_lock:
            pm.switch(body.direction, preload=True)

    threading.Thread(target=_do_switch, daemon=True).start()
    return {
        "status":    "switching",
        "direction": body.direction,
        "label":     PIPELINE_CONFIGS[body.direction].label,
        "message":   "Đang tải pipeline mới — poll /api/status để biết khi nào sẵn sàng",
    }


# ─── Full Pipeline (file upload) ─────────────────────────────────────────────
@app.post("/api/pipeline")
async def api_pipeline(file: UploadFile = File(...)):
    """
    Full pipeline: Audio → ASR → MT → TTS → Audio WAV.
    Sử dụng luồng đang active trong PipelineManager.
    """
    if not pm.is_ready:
        status = pm.get_status()
        raise HTTPException(
            503,
            f"Pipeline [{pm.active_direction}] chưa sẵn sàng. "
            f"Models: {status['models']}"
        )

    data = await file.read()
    try:
        samples, sr = _read_audio_file(data, file.filename)
    except Exception as e:
        raise HTTPException(400, f"Không đọc được audio: {e}")

    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: pm.run(samples, sr)
        )
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        raise HTTPException(500, f"Pipeline error: {e}")

    return result


# ─── ASR only ────────────────────────────────────────────────────────────────
@app.post("/api/asr")
async def api_asr(file: UploadFile = File(...)):
    """Chỉ chạy ASR. Trả về text nhận dạng + latency."""
    if not pm._asr or pm._asr.status != ModelStatus.READY:
        raise HTTPException(503, "ASR not ready")

    data = await file.read()
    try:
        samples, sr = _read_audio_file(data, file.filename)
    except Exception as e:
        raise HTTPException(400, f"Cannot read audio: {e}")

    return pm._asr.transcribe(samples, sr)


# ─── MT only ─────────────────────────────────────────────────────────────────
class TextIn(BaseModel):
    text: str

@app.post("/api/mt")
async def api_mt(body: TextIn):
    """Chỉ chạy MT. Trả về text dịch + latency."""
    if not pm._mt or pm._mt.status != ModelStatus.READY:
        raise HTTPException(503, "MT not ready")
    return pm._mt.translate(body.text)


# ─── TTS only ────────────────────────────────────────────────────────────────
class TTSIn(BaseModel):
    text:  str
    voice: Optional[str] = None

@app.post("/api/tts")
async def api_tts(body: TTSIn):
    """Chỉ chạy TTS. Trả về audio WAV."""
    if not pm._tts or pm._tts.status != ModelStatus.READY:
        raise HTTPException(503, "TTS not ready")
    try:
        result = pm._tts.synthesize(body.text, voice=body.voice)
    except Exception as e:
        raise HTTPException(500, str(e))
    return Response(
        content=result["wav_bytes"],
        media_type="audio/wav",
        headers={
            "X-Duration-Sec": str(result["duration_sec"]),
            "X-Latency-Ms":   str(result["latency_ms"]),
        },
    )


# ════════════════════════════════════════════════════════════════════════════
# WEBSOCKET — Real-time mic pipeline
# ════════════════════════════════════════════════════════════════════════════

@app.websocket("/ws/pipeline")
async def ws_pipeline(ws: WebSocket):
    """
    WebSocket protocol:
      Client → Server:
        binary: raw PCM float32 LE, 16kHz mono
        text {type: "stop"}
        text {type: "status"}
        text {type: "switch", direction: "vi-en"}
      Server → Client:
        {type: "asr",    text, latency_ms}
        {type: "mt",     text, latency_ms}
        {type: "tts",    audio_b64, duration_sec, latency_ms}
        {type: "status", direction, models, ram_mb}
        {type: "error",  message}
        {type: "switching", direction}
    """
    await ws.accept()
    log.info("WebSocket client connected")

    pcm_buffer: list[bytes] = []

    async def send_json(obj: dict):
        await ws.send_text(json.dumps(obj))

    # Gửi trạng thái ngay khi kết nối
    await send_json({"type": "status", **pm.get_status()})

    try:
        while True:
            try:
                msg = await asyncio.wait_for(ws.receive(), timeout=60.0)
            except asyncio.TimeoutError:
                await send_json({"type": "ping"})
                continue

            # ─── Text messages (control) ──────────────────────────────────
            if "text" in msg:
                payload = json.loads(msg["text"])

                if payload.get("type") == "switch":
                    direction = payload.get("direction", "vi-en")
                    if direction not in PIPELINE_CONFIGS:
                        await send_json({"type": "error", "message": f"Invalid direction: {direction}"})
                        continue
                    await send_json({"type": "switching", "direction": direction})

                    def _do_switch():
                        with _switch_lock:
                            pm.switch(direction, preload=True)

                    threading.Thread(target=_do_switch, daemon=True).start()

                    # Poll cho tới khi ready (tối đa 60s)
                    for _ in range(120):
                        await asyncio.sleep(0.5)
                        if pm.is_ready and pm.active_direction == direction:
                            break
                    await send_json({"type": "status", **pm.get_status()})

                elif payload.get("type") == "stop":
                    if not pcm_buffer:
                        await send_json({"type": "error", "message": "No audio received"})
                        continue

                    raw = b"".join(pcm_buffer)
                    pcm_buffer.clear()

                    n = len(raw) // 4
                    samples = np.frombuffer(raw[:n * 4], dtype="<f4").copy()
                    log.info(f"WS: Processing {len(samples)/16000:.2f}s audio [{pm.active_direction}]")

                    loop = asyncio.get_event_loop()
                    try:
                        result = await loop.run_in_executor(None, lambda: pm.run(samples, 16000))
                        await send_json({"type": "asr", **result["stages"]["asr"]})
                        await send_json({"type": "mt",  **result["stages"]["mt"]})
                        await send_json({
                            "type":         "tts",
                            "audio_b64":    result["audio_b64"],
                            "duration_sec": result["stages"]["tts"]["duration_sec"],
                            "latency_ms":   result["stages"]["tts"]["latency_ms"],
                            "total_ms":     result["total_latency_ms"],
                            "metrics":      result["metrics"],
                        })
                    except Exception as e:
                        log.error(f"WS pipeline error: {e}")
                        await send_json({"type": "error", "message": str(e)})

                elif payload.get("type") == "status":
                    await send_json({"type": "status", **pm.get_status()})

                elif payload.get("type") == "metrics":
                    await send_json({"type": "metrics", **pm.get_metrics()})

            # ─── Binary: PCM audio chunks ─────────────────────────────────
            elif "bytes" in msg and msg["bytes"]:
                pcm_buffer.append(msg["bytes"])

    except WebSocketDisconnect:
        log.info("WebSocket client disconnected")
    except Exception as e:
        log.error(f"WebSocket error: {e}")


# ════════════════════════════════════════════════════════════════════════════
# UTILS
# ════════════════════════════════════════════════════════════════════════════

def _read_audio_file(data: bytes, filename: str = "") -> tuple[np.ndarray, int]:
    """Đọc file audio → (float32 mono, sample_rate)."""
    try:
        import soundfile as sf
        arr, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        if arr.ndim == 2:
            arr = arr.mean(axis=1)
        return arr.astype(np.float32), sr
    except Exception:
        with wave.open(io.BytesIO(data)) as wf:
            sr     = wf.getframerate()
            raw    = wf.readframes(wf.getnframes())
            nch    = wf.getnchannels()
            swidth = wf.getsampwidth()
        dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(swidth, np.int16)
        arr = np.frombuffer(raw, dtype=dtype).astype(np.float32)
        arr /= float(np.iinfo(dtype).max)
        if nch > 1:
            arr = arr.reshape(-1, nch).mean(axis=1)
        return arr, sr


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
