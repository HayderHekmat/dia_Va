# main.py
import io
import os
import re
import time
import uuid
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# Use the Dia class exactly like your Gradio app
from dia.model import Dia

# -------------------- Model bootstrap --------------------
DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else torch.device("cpu")
)

DTYPE = {"cpu": "float32", "mps": "float32", "cuda": "float16"}[DEVICE.type]
MODEL_ID = os.getenv("DIA_MODEL_ID", "nari-labs/Dia-1.6B-0626")

_model: Optional[Dia] = None

def load_model():
    global _model
    if _model is None:
        _model = Dia.from_pretrained(MODEL_ID, compute_dtype=DTYPE, device=DEVICE)

# -------------------- Vapi payload models --------------------
class VapiMessage(BaseModel):
    timestamp: int
    type: str
    status: str
    role: str
    turn: int
    artifact: Dict[str, Any] = {}
    call: Dict[str, Any] = {}
    assistant: Dict[str, Any] = {}

class VapiRequest(BaseModel):
    message: VapiMessage

# -------------------- Helpers --------------------
AUDIO_DIR = "audio_files"
os.makedirs(AUDIO_DIR, exist_ok=True)

# Extract assistant text from the Vapi payload (messages or messagesOpenAIFormatted)
def _extract_text(v: VapiRequest) -> str:
    art = v.message.artifact or {}

    # Prefer OpenAI-formatted assistant messages (they’re common in your logs)
    msgs = art.get("messagesOpenAIFormatted") or []
    for msg in reversed(msgs):
        if (msg.get("role") or "").lower() in ("assistant", "bot"):
            content = msg.get("content")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                parts = []
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "text":
                        parts.append(c.get("text", ""))
                s = " ".join(parts).strip()
                if s:
                    return s

    # Fallback: raw messages
    raw = art.get("messages") or []
    for msg in reversed(raw):
        if (msg.get("role") or "").lower() in ("assistant", "bot"):
            s = str(msg.get("message", "")).strip()
            if s:
                return s

    # First message fallback
    if v.message.turn == 0:
        fm = (v.message.assistant or {}).get("firstMessage")
        if fm:
            return str(fm).strip()

    return ""

# Strip inline effects like "(laughs)" and return clean text + first effect (if any)
_EFFECTS = {"laughs","sighs","clears throat","coughs","gasps","humming","exhales","mumbles","groans","sneezes","burps","screams"}
def _strip_effects(text: str) -> Tuple[str, Optional[str]]:
    # match things inside parentheses
    raw = re.findall(r"\(([^)]+)\)", text)
    clean = re.sub(r"\(([^)]+)\)", "", text).strip()
    effect = None
    for r in raw:
        key = r.strip().lower()
        if key in _EFFECTS:
            effect = key
            break
    return clean, effect

# Very small “speed” post-process (optional)
def _apply_speed(audio: np.ndarray, factor: float) -> np.ndarray:
    factor = max(0.5, min(2.0, float(factor)))
    if abs(factor - 1.0) < 1e-3:
        return audio
    n = len(audio)
    tgt = int(n / factor)
    if tgt < 2:
        return audio
    x = np.arange(n)
    xi = np.linspace(0, n - 1, tgt)
    return np.interp(xi, x, audio).astype(np.float32)

# -------------------- FastAPI app --------------------
app = FastAPI(title="Dia TTS API", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

@app.on_event("startup")
def _startup():
    load_model()

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "dtype": DTYPE,
        "model_loaded": _model is not None,
        "effects_supported": sorted(list(_EFFECTS)),
    }

@app.post("/api/generate")
def generate(v: VapiRequest):
    if _model is None:
        raise HTTPException(500, "Model not loaded")

    if v.message.status not in {"stopped", "completed", "done"}:
        # Mirror your webhook flow: only synthesize when the assistant finished speaking
        return JSONResponse({"status": "ack", "note": f"status '{v.message.status}' ignored"})

    text = _extract_text(v)
    if not text:
        return JSONResponse({"status": "ignored", "reason": "no assistant text"})

    # Remove inline effect tags — model won’t render them; you can mix SFX later if you want
    clean_text, _ = _strip_effects(text)
    if not clean_text:
        return JSONResponse({"status": "ignored", "reason": "no clean text"})

    # Defaults — you can wire {#tts ... #} parsing later if you like
    max_tokens   = int(os.getenv("DIA_MAX_TOKENS", 1024))
    cfg_scale    = float(os.getenv("DIA_CFG_SCALE", 3.0))
    temperature  = float(os.getenv("DIA_TEMPERATURE", 1.3))
    top_p        = float(os.getenv("DIA_TOP_P", 0.95))
    top_k        = int(os.getenv("DIA_CFG_FILTER_TOP_K", 35))
    speed        = float(os.getenv("DIA_SPEED", 1.0))

    # Synthesize
    try:
        start = time.time()
        with torch.inference_mode():
            wav = _model.generate(
                clean_text,
                max_tokens=max_tokens,
                cfg_scale=cfg_scale,
                temperature=temperature,
                top_p=top_p,
                cfg_filter_top_k=top_k,
                audio_prompt=None,
                verbose=False,
            )
        if wav is None or len(wav) == 0:
            raise RuntimeError("No audio returned from model.generate")

        # Dia DAC sample rate
        sr = 44100
        wav = wav.astype(np.float32)
        wav = _apply_speed(wav, speed)

        # Save to a temp file and return it
        tmp = os.path.join(AUDIO_DIR, f"{int(time.time())}_{uuid.uuid4().hex[:8]}.wav")
        sf.write(tmp, wav, sr)
        dur = time.time() - start

        return FileResponse(
            tmp, media_type="audio/wav", filename=os.path.basename(tmp),
            headers={"X-Gen-Time": f"{dur:.2f}", "X-Text-Used": clean_text[:120]}
        )
    except Exception as e:
        raise HTTPException(500, f"TTS error: {e}")
