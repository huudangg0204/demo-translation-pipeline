#!/usr/bin/env bash
# demo/download_models.sh — Tải model cho các luồng dịch OneVoice S2S
#
# Cách dùng:
#   bash download_models.sh                  # Mặc định: tải pipeline vi-en
#   bash download_models.sh --all            # Tải tất cả 4 luồng (vi-en, en-vi, vi-zh, zh-vi)
#   bash download_models.sh --pipeline en-vi # Tải riêng luồng en-vi
#   bash download_models.sh --pipeline vi-zh # Tải riêng luồng vi-zh
#   bash download_models.sh --pipeline zh-vi # Tải riêng luồng zh-vi

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/models"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_PYTHON="$ROOT_DIR/.venv/bin/python3"

if [ ! -f "$VENV_PYTHON" ]; then
    VENV_PYTHON="python3"
fi

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
ok()   { echo -e "${GREEN}✅ $*${NC}"; }
info() { echo -e "${BLUE}📥 $*${NC}"; }
warn() { echo -e "${YELLOW}⚠  $*${NC}"; }

mkdir -p "$MODELS_DIR"

MODE="${1:---default}"
TARGET_PIPELINE="${2:-vi-en}"

echo ""
echo -e "${BLUE}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   OneVoice AI — Multi-language Model Downloader        ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""

# ───────────────────────────────────────────────────────────────────
# Helper Functions
# ───────────────────────────────────────────────────────────────────

download_zipformer_vi() {
    local ASR_DIR="$MODELS_DIR/sherpa-onnx-zipformer-vi-30M-int8-2026-02-09"
    local ASR_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-vi-30M-int8-2026-02-09.tar.bz2"
    if [ -f "$ASR_DIR/tokens.txt" ] && [ -f "$ASR_DIR/encoder.int8.onnx" ]; then
        ok "Zipformer VI 30M already present"
    else
        info "Downloading Zipformer VI 30M INT8 (~25 MB)..."
        curl -fL --progress-bar "$ASR_URL" -o "$MODELS_DIR/asr_tmp.tar.bz2"
        tar xjf "$MODELS_DIR/asr_tmp.tar.bz2" -C "$MODELS_DIR"
        rm "$MODELS_DIR/asr_tmp.tar.bz2"
        ok "Zipformer VI 30M ready"
    fi
}

download_paraformer_zh() {
    local ZH_DIR="$MODELS_DIR/sherpa-onnx-paraformer-zh-int8"
    local ZH_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2024-03-09.tar.bz2"
    if [ -d "$ZH_DIR" ] && [ -n "$(ls -A "$ZH_DIR" 2>/dev/null)" ]; then
        ok "Paraformer ZH INT8 already present"
    else
        info "Downloading Paraformer ZH (~120 MB)..."
        curl -fL --progress-bar "$ZH_URL" -o "$MODELS_DIR/zh_tmp.tar.bz2" || warn "Download failed, check connection"
        if [ -f "$MODELS_DIR/zh_tmp.tar.bz2" ]; then
            tar xjf "$MODELS_DIR/zh_tmp.tar.bz2" -C "$MODELS_DIR"
            rm "$MODELS_DIR/zh_tmp.tar.bz2"
            if [ -d "$MODELS_DIR/sherpa-onnx-paraformer-zh-2024-03-09" ]; then
                mv "$MODELS_DIR/sherpa-onnx-paraformer-zh-2024-03-09" "$ZH_DIR"
            fi
            ok "Paraformer ZH ready"
        fi
    fi
}

download_marian_mt() {
    local pair="$1"
    local MT_DIR="$MODELS_DIR/opus-mt-$pair"
    if [ -f "$MT_DIR/config.json" ]; then
        ok "Opus-MT $pair already present"
    else
        info "Downloading Opus-MT $pair (~300 MB)..."
        $VENV_PYTHON - <<PYEOF
from transformers import MarianMTModel, MarianTokenizer
from pathlib import Path
out = Path("$MT_DIR")
out.mkdir(parents=True, exist_ok=True)
model_id = "Helsinki-NLP/opus-mt-$pair"
print(f"  Downloading {model_id}...")
tok = MarianTokenizer.from_pretrained(model_id)
tok.save_pretrained(str(out))
mod = MarianMTModel.from_pretrained(model_id)
mod.save_pretrained(str(out))
print(f"  Saved to: {out}")
PYEOF
        ok "Opus-MT $pair ready"
    fi
}

download_kokoro_en() {
    local KOKORO_DIR="$MODELS_DIR/kokoro"
    local BASE="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
    mkdir -p "$KOKORO_DIR"
    if [ -f "$KOKORO_DIR/kokoro-v1.0.onnx" ]; then
        ok "Kokoro EN ONNX already present"
    else
        info "Downloading Kokoro EN ONNX (~325 MB)..."
        curl -fL --progress-bar "$BASE/kokoro-v1.0.onnx" -o "$KOKORO_DIR/kokoro-v1.0.onnx"
        ok "Kokoro EN ONNX ready"
    fi
    if [ ! -f "$KOKORO_DIR/voices-v1.0.bin" ]; then
        info "Downloading Kokoro EN voices..."
        curl -fL --progress-bar "$BASE/voices-v1.0.bin" -o "$KOKORO_DIR/voices-v1.0.bin"
        ok "Kokoro EN voices ready"
    fi
}

download_kokoro_zh() {
    local KOKORO_ZH="$MODELS_DIR/kokoro-zh"
    mkdir -p "$KOKORO_ZH"
    info "Checking Kokoro ZH model..."
    $VENV_PYTHON - <<PYEOF || warn "Kokoro ZH auto-download via huggingface_hub failed"
try:
    from huggingface_hub import hf_hub_download
    from pathlib import Path
    out = Path("$KOKORO_ZH")
    for f in ["kokoro-v1.1-zh.onnx", "voices-v1.1-zh.bin"]:
        p = out / f
        if not p.exists():
            print(f"  Downloading {f}...")
            hf_hub_download("hexgrad/Kokoro-82M-v1.1-zh", f, local_dir=str(out))
    print("  Kokoro ZH ready")
except Exception as e:
    print(f"  Skip Kokoro ZH download: {e}")
PYEOF
}

download_kokoro_vi() {
    local KOKORO_VI="$MODELS_DIR/kokoro-vi"
    mkdir -p "$KOKORO_VI"
    info "Checking Kokoro VI model..."
    $VENV_PYTHON - <<PYEOF || warn "Kokoro VI auto-download via huggingface_hub failed"
try:
    from huggingface_hub import hf_hub_download
    from pathlib import Path
    out = Path("$KOKORO_VI")
    for f in ["kokoro-vi.onnx", "voices-vi.bin"]:
        p = out / f
        if not p.exists():
            print(f"  Downloading {f}...")
            hf_hub_download("anphunl/Kokoro-Vietnamese", f, local_dir=str(out))
    print("  Kokoro VI ready")
except Exception as e:
    print(f"  Skip Kokoro VI download: {e}")
PYEOF
}

# ───────────────────────────────────────────────────────────────────
# Execution based on CLI args
# ───────────────────────────────────────────────────────────────────

if [ "$MODE" == "--all" ]; then
    echo "Tải toàn bộ model cho 4 luồng dịch..."
    download_zipformer_vi
    download_paraformer_zh
    download_marian_mt "vi-en"
    download_marian_mt "en-vi"
    download_marian_mt "vi-zh"
    download_marian_mt "zh-vi"
    download_kokoro_en
    download_kokoro_zh
    download_kokoro_vi

elif [ "$MODE" == "--pipeline" ]; then
    case "$TARGET_PIPELINE" in
        "vi-en")
            download_zipformer_vi
            download_marian_mt "vi-en"
            download_kokoro_en
            ;;
        "en-vi")
            download_marian_mt "en-vi"
            download_kokoro_vi
            info "Moonshine Base sẽ được tải tự động khi chạy hoặc qua: python scripts/quantize_models.py --model moonshine"
            ;;
        "vi-zh")
            download_zipformer_vi
            download_marian_mt "vi-zh"
            download_kokoro_zh
            ;;
        "zh-vi")
            download_paraformer_zh
            download_marian_mt "zh-vi"
            download_kokoro_vi
            ;;
        *)
            warn "Unknown pipeline: $TARGET_PIPELINE. Chọn: vi-en, en-vi, vi-zh, zh-vi"
            ;;
    esac
else
    # Default: vi-en
    echo "Tải pipeline mặc định (VI → EN)..."
    download_zipformer_vi
    download_marian_mt "vi-en"
    download_kokoro_en
fi

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  ✅  Download hoàn tất!                           ${NC}"
echo -e "${GREEN}                                                   ${NC}"
echo -e "${GREEN}  Để nén các model về INT8, chạy:                  ${NC}"
echo -e "${GREEN}    python scripts/quantize_models.py --model all  ${NC}"
echo -e "${GREEN}                                                   ${NC}"
echo -e "${GREEN}  Khởi chạy demo:                                  ${NC}"
echo -e "${GREEN}    cd demo && uvicorn app:app --port 8000         ${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════${NC}"
echo ""
