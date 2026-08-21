"""
demo/app.py — OneVoice AI Speech-to-Speech Demo (Plug-and-Play)
================================================================
Pipeline: Audio → ASR → MT → TTS → Audio

Hỗ trợ 4 luồng dịch:
  VI → EN : Zipformer 30M + Opus-MT vi-en + Kokoro EN
  EN → VI : Moonshine Base INT8 + Opus-MT en-vi + Kokoro VI
  VI → ZH : Zipformer 30M + Opus-MT vi-zh + Kokoro ZH
  ZH → VI : Paraformer ZH INT8 + Opus-MT zh-vi + Kokoro VI

Chế độ:
  1. Turn-based (Batch)     : /ws/pipeline          — Thu âm → bấm Stop → ASR → MT → TTS
  2. Free-hand (Continuous) : /ws/pipeline-continuous — Mic liên tục → VAD cắt câu → ASR/MT/TTS async queue

Run:
  cd demo
  uvicorn app:app --reload --port 8000
"""

import asyncio
import base64
import concurrent.futures
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
from vad_streamer import VADStreamer, VADConfig

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
    version="3.0.0",
)

# ─── Thread pool cho ASR/MT/TTS blocking calls ───────────────────────────────
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="onevoice")
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
# WEBSOCKET — Free-hand Continuous Streaming Pipeline
# ════════════════════════════════════════════════════════════════════════════

@app.websocket("/ws/pipeline-continuous")
async def ws_pipeline_continuous(ws: WebSocket):
    """
    WebSocket protocol (Continuous / Free-hand mode):

    Client → Server:
      binary:  raw PCM float32 LE, 16kHz mono chunks (~64ms each)
      text {type: "set_vad", silence_ms: 500, threshold: 0.5}   — đổi tham số VAD realtime
      text {type: "switch", direction: "vi-en"}                 — đổi cặp ngôn ngữ
      text {type: "pause"}                                       — tạm dừng
      text {type: "resume"}                                      — tiếp tục
      text {type: "status"}                                      — truy vấn trạng thái

    Server → Client:
      {type: "status",   direction, models, ram_mb}              — trạng thái pipeline
      {type: "vad_start", uid}                                   — người dùng bắt đầu nói câu mới
      {type: "asr",      uid, text, latency_ms}                  — kết quả ASR
      {type: "mt",       uid, text, latency_ms}                  — kết quả MT
      {type: "tts",      uid, audio_b64, duration_sec, latency_ms, total_ms} — audio TTS
      {type: "vad_info", threshold, noise_floor, silence_ms, backend}        — thông số VAD hiện tại
      {type: "error",    uid, stage, message}                    — lỗi 1 câu (không dừng stream)
    """
    await ws.accept()
    log.info("Continuous WS client connected")

    loop = asyncio.get_event_loop()
    paused = False
    session_start = time.perf_counter()

    # ─── Hàng đợi nội bộ ──────────────────────────────────────────────────
    mt_queue:  asyncio.Queue = asyncio.Queue(maxsize=10)
    tts_queue: asyncio.Queue = asyncio.Queue(maxsize=10)

    async def send_json(obj: dict):
        try:
            await ws.send_text(json.dumps(obj, ensure_ascii=False))
        except Exception:
            pass

    # ─── Gửi trạng thái khởi tạo ──────────────────────────────────────────
    await send_json({"type": "status", **pm.get_status()})

    # ─── VAD callback: người dùng bắt đầu nói ─────────────────────────────
    def on_speech_start(uid: int):
        asyncio.run_coroutine_threadsafe(
            send_json({"type": "vad_start", "uid": uid}),
            loop,
        )

    # ─── VAD callback: utterance hoàn chỉnh → đẩy vào ASR ─────────────────
    def on_utterance(samples: np.ndarray, uid: int):
        if paused:
            log.debug(f"Continuous WS paused — utterance {uid} dropped")
            return
        if not pm.is_ready:
            asyncio.run_coroutine_threadsafe(
                send_json({"type": "error", "uid": uid, "stage": "asr",
                           "message": "Pipeline chưa sẵn sàng"}),
                loop,
            )
            return

        t_utterance_start = time.perf_counter()

        # ASR chạy trong thread pool (blocking)
        def _run_asr():
            try:
                asr_r = pm._asr.transcribe(samples, 16000)
                src_text = asr_r["text"].strip()
                if not src_text:
                    log.debug(f"ASR uid={uid} returned empty text — skipped")
                    return
                # Phát event ASR về client
                asyncio.run_coroutine_threadsafe(
                    send_json({"type": "asr", "uid": uid,
                               "text": src_text, "latency_ms": asr_r["latency_ms"]}),
                    loop,
                )
                # Đẩy vào MT queue
                mt_item = (uid, src_text, t_utterance_start, asr_r["latency_ms"])
                asyncio.run_coroutine_threadsafe(
                    mt_queue.put(mt_item), loop
                )
            except Exception as e:
                log.error(f"ASR error uid={uid}: {e}")
                asyncio.run_coroutine_threadsafe(
                    send_json({"type": "error", "uid": uid, "stage": "asr", "message": str(e)}),
                    loop,
                )

        _executor.submit(_run_asr)

    # ─── Khởi tạo VAD Streamer ─────────────────────────────────────────────
    vad_config = VADConfig(
        silence_limit_ms=500,
        min_speech_ms=200,
        max_speech_ms=10_000,
    )
    vad = VADStreamer(
        config=vad_config,
        sample_rate=16000,
        on_speech_start=on_speech_start,
        on_utterance=on_utterance,
    )

    # ─── MT Worker ────────────────────────────────────────────────────────
    async def mt_worker():
        while True:
            try:
                item = await asyncio.wait_for(mt_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            uid, src_text, t_start, asr_ms = item
            try:
                mt_r = await loop.run_in_executor(
                    _executor, lambda: pm._mt.translate(src_text)
                )
                tgt_text = mt_r["text"].strip()
                await send_json({"type": "mt", "uid": uid,
                                 "text": tgt_text, "latency_ms": mt_r["latency_ms"]})
                # Đẩy vào TTS queue
                await tts_queue.put((uid, tgt_text, t_start, asr_ms, mt_r["latency_ms"]))
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"MT error uid={uid}: {e}")
                await send_json({"type": "error", "uid": uid, "stage": "mt", "message": str(e)})
            finally:
                mt_queue.task_done()

    # ─── TTS Worker ───────────────────────────────────────────────────────
    async def tts_worker():
        while True:
            try:
                item = await asyncio.wait_for(tts_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            uid, tgt_text, t_start, asr_ms, mt_ms = item
            if not tgt_text:
                tts_queue.task_done()
                continue

            try:
                tts_r = await loop.run_in_executor(
                    _executor, lambda: pm._tts.synthesize(tgt_text)
                )
                total_ms = round((time.perf_counter() - t_start) * 1000, 1)
                audio_b64 = base64.b64encode(tts_r["wav_bytes"]).decode()
                await send_json({
                    "type":         "tts",
                    "uid":          uid,
                    "audio_b64":    audio_b64,
                    "duration_sec": tts_r["duration_sec"],
                    "latency_ms":   tts_r["latency_ms"],
                    "total_ms":     total_ms,
                    "sample_rate":  tts_r.get("sample_rate", 24000),
                    "metrics": {
                        "asr_ms":   asr_ms,
                        "mt_ms":    mt_ms,
                        "tts_ms":   tts_r["latency_ms"],
                        "total_ms": total_ms,
                    },
                })
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"TTS error uid={uid}: {e}")
                await send_json({"type": "error", "uid": uid, "stage": "tts", "message": str(e)})
            finally:
                tts_queue.task_done()

    # ─── Khởi động workers ────────────────────────────────────────────────
    mt_task  = asyncio.create_task(mt_worker(),  name="mt_worker")
    tts_task = asyncio.create_task(tts_worker(), name="tts_worker")

    try:
        while True:
            try:
                msg = await asyncio.wait_for(ws.receive(), timeout=30.0)
            except asyncio.TimeoutError:
                await send_json({"type": "ping"})
                continue

            # ─── Xử lý text control messages ──────────────────────────────
            if "text" in msg:
                try:
                    payload = json.loads(msg["text"])
                except Exception:
                    continue

                msg_type = payload.get("type")

                if msg_type == "set_vad":
                    # Cập nhật tham số VAD realtime
                    updates = {}
                    if "silence_ms" in payload:
                        updates["silence_limit_ms"] = int(payload["silence_ms"])
                    if "threshold" in payload:
                        updates["silero_threshold"] = float(payload["threshold"])
                        updates["initial_threshold"] = float(payload["threshold"])
                    if "min_speech_ms" in payload:
                        updates["min_speech_ms"] = int(payload["min_speech_ms"])
                    if updates:
                        vad.update_config(**updates)
                    # Trả về thông số VAD hiện tại
                    await send_json({
                        "type":         "vad_info",
                        "silence_ms":   vad.config.silence_limit_ms,
                        "threshold":    round(vad.current_threshold, 4),
                        "noise_floor":  round(vad.noise_floor, 4),
                        "min_speech_ms": vad.config.min_speech_ms,
                        "backend":      vad.backend,
                    })

                elif msg_type == "pause":
                    paused = True
                    vad.reset()
                    await send_json({"type": "paused"})

                elif msg_type == "resume":
                    paused = False
                    await send_json({"type": "resumed"})

                elif msg_type == "switch":
                    direction = payload.get("direction", "vi-en")
                    if direction not in PIPELINE_CONFIGS:
                        await send_json({"type": "error", "uid": -1, "stage": "switch",
                                         "message": f"Invalid direction: {direction}"})
                        continue
                    paused = True
                    vad.reset()
                    await send_json({"type": "switching", "direction": direction})

                    def _do_switch():
                        with _switch_lock:
                            pm.switch(direction, preload=True)

                    threading.Thread(target=_do_switch, daemon=True).start()
                    for _ in range(120):
                        await asyncio.sleep(0.5)
                        if pm.is_ready and pm.active_direction == direction:
                            break
                    paused = False
                    await send_json({"type": "status", **pm.get_status()})

                elif msg_type == "status":
                    await send_json({"type": "status", **pm.get_status()})
                    await send_json({
                        "type":         "vad_info",
                        "silence_ms":   vad.config.silence_limit_ms,
                        "threshold":    round(vad.current_threshold, 4),
                        "noise_floor":  round(vad.noise_floor, 4),
                        "min_speech_ms": vad.config.min_speech_ms,
                        "backend":      vad.backend,
                    })

                elif msg_type == "flush":
                    # Buộc kết thúc câu đang nói dở
                    vad.flush()

            # ─── Xử lý binary: PCM audio chunk ────────────────────────────
            elif "bytes" in msg and msg["bytes"]:
                if paused:
                    continue
                raw = msg["bytes"]
                n = len(raw) // 4
                if n == 0:
                    continue
                chunk = np.frombuffer(raw[:n * 4], dtype="<f4").copy()
                vad.feed(chunk)

    except WebSocketDisconnect:
        log.info("Continuous WS client disconnected")
    except Exception as e:
        log.error(f"Continuous WS error: {e}")
    finally:
        # Cleanup
        vad.flush()
        mt_task.cancel()
        tts_task.cancel()
        try:
            await asyncio.gather(mt_task, tts_task, return_exceptions=True)
        except Exception:
            pass
        duration = round(time.perf_counter() - session_start, 1)
        log.info(f"Continuous WS session ended — duration={duration}s")


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
