# main.py
import os
import io
import re
import uuid
import time
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager

import torch

# Dia official API
# repo: https://github.com/nari-labs/dia
from dia.model import Dia

APP_NAME = "Dia Text-to-Speech (Vapi-ready)"
AUDIO_DIR = Path("audio_files"); AUDIO_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR = Path("upload_files"); UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = os.getenv("DIA_MODEL_ID", "nari-labs/Dia-1.6B-0626")
COMPUTE_DTYPE = os.getenv("DIA_COMPUTE_DTYPE", "float16")  # "float16" or "float32"
USE_TORCH_COMPILE = os.getenv("DIA_TORCH_COMPILE", "false").lower() == "true"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------
# Helpers / extraction
# ----------------------
def _extract_text_from_openai_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                out.append(str(part.get("text", "")))
        return " ".join(out).strip()
    return ""

def extract_text_from_vapi_message(vapi_message: Dict[str, Any]) -> str:
    """
    Pull the most recent assistant/bot text from a Vapi speech-update payload.
    Falls back to assistant.firstMessage when turn == 0.
    """
    try:
        art = (vapi_message.get("artifact") or {})
        # Newest assistant/bot message in messages
        msgs = (art.get("messages") or [])
        for msg in reversed(msgs):
            role = (msg.get("role") or "").lower()
            if role in ("assistant", "bot") and msg.get("message"):
                return str(msg["message"]).strip()

        # Newest assistant content in messagesOpenAIFormatted
        fmts = (art.get("messagesOpenAIFormatted") or [])
        for msg in reversed(fmts):
            role = (msg.get("role") or "").lower()
            if role in ("assistant", "bot") and "content" in msg:
                s = _extract_text_from_openai_content(msg["content"])
                if s:
                    return s.strip()

        # turn == 0: use assistant.firstMessage
        if (vapi_message.get("turn") == 0):
            a = vapi_message.get("assistant") or {}
            fm = a.get("firstMessage")
            if fm:
                return str(fm).strip()
        return ""
    except Exception:
        return ""

# Parse optional inline TTS controls: {#tts temperature=1.1 top_p=0.9 top_k=40 cfg=3.5 max_new_tokens=800 #}
_TTS_KEYS = {
    "temperature": float,
    "top_p": float,
    "top_k": int,
    "cfg": float,              # guidance scale
    "max_new_tokens": int,
}
def parse_tts_controls(text: str) -> Tuple[str, Dict[str, Any]]:
    pattern = r"\{\#tts\s+([^}]*)\#\}"
    m = list(re.finditer(pattern, text, flags=re.IGNORECASE))
    if not m:
        return text, {}
    last = m[-1]
    inner = last.group(1)
    clean = (text[:last.start()] + text[last.end():]).strip()
    pairs = re.findall(r"([a-zA-Z_]+)\s*=\s*([-\d\.]+)", inner)
    out: Dict[str, Any] = {}
    for k, v in pairs:
        kl = k.lower()
        if kl in _TTS_KEYS:
            try:
                out[kl] = _TTS_KEYS[kl](v)
            except Exception:
                pass
    return clean, out

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def apply_overrides(defaults: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    d = dict(defaults)
    if "temperature" in overrides:
        d["temperature"] = clamp(overrides["temperature"], 0.1, 2.0)
    if "top_p" in overrides:
        d["top_p"] = clamp(overrides["top_p"], 0.1, 1.0)
    if "top_k" in overrides:
        d["cfg_filter_top_k"] = int(clamp(overrides["top_k"], 1, 200))
    if "cfg" in overrides:
        d["cfg_scale"] = clamp(overrides["cfg"], 0.1, 10.0)
    if "max_new_tokens" in overrides:
        d["max_tokens"] = int(clamp(overrides["max_new_tokens"], 1, 4096))
    return d

# Optional language hint injection:
# Dia mainly infers from text; if you pass German text, it will speak German.
# If you want a nudge, we prepend a neutral hint. You can disable by not sending `language`.
def add_language_hint(text: str, language: Optional[str]) -> str:
    if not language:
        return text
    # Minimal, safe hint; does not pollute audible output (kept brief).
    # You can change to your convention if you already use one.
    return f"[LANG={language.lower()}] {text}"

# Voice (speaker) control:
# Dia uses speaker tags like [S1], [S2] to switch voices.
# If caller passes voice="S2", we wrap as "[S2] <text>"
def add_voice_tag(text: str, voice: Optional[str]) -> str:
    if not voice:
        return text
    vv = voice.strip().upper()
    if not vv.startswith("S"):
        # allow "1" -> "S1"
        if vv.isdigit():
            vv = f"S{vv}"
    return f"[{vv}] {text}"

# ----------------------
# Model manager
# ----------------------
class DiaModelManager:
    def __init__(self):
        self.model: Optional[Dia] = None
        self.loaded = False

    def load(self):
        if self.loaded and self.model is not None:
            return
        # Dia handles device internally; we give compute dtype
        self.model = Dia.from_pretrained(MODEL_ID, compute_dtype=COMPUTE_DTYPE)
        self.loaded = True

    def unload(self):
        try:
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        finally:
            self.loaded = False

    def get(self) -> Dia:
        if not self.loaded or self.model is None:
            raise RuntimeError("Dia model not loaded")
        return self.model

model_mgr = DiaModelManager()

# ----------------------
# FastAPI app
# ----------------------
@asynccontextmanager
async def lifespan(_: FastAPI):
    model_mgr.load()
    yield
    model_mgr.unload()

app = FastAPI(title=APP_NAME, version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------
# Schemas
# -------------
class GenerateDirectRequest(BaseModel):
    text: str
    # Optional tuning
    voice: Optional[str] = None           # "S1", "S2", or "1"/"2"
    language: Optional[str] = None        # e.g. "de", "en", "fr"
    temperature: float = 1.3
    top_p: float = 0.95
    top_k: int = 35                       # maps to cfg_filter_top_k
    cfg: float = 3.0                      # maps to cfg_scale
    max_new_tokens: int = 1024            # maps to max_tokens
    # Optional path to an audio prompt already on disk (advanced)
    audio_prompt_path: Optional[str] = None

# -------------
# Endpoints
# -------------
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "cuda": torch.cuda.is_available(),
        "model_id": MODEL_ID,
        "loaded": model_mgr.loaded,
    }

def _normalize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    # Dia expects: cfg_scale, temperature, top_p, cfg_filter_top_k, max_tokens
    defaults = dict(
        cfg_scale=3.0,
        temperature=1.3,
        top_p=0.95,
        cfg_filter_top_k=35,
        max_tokens=1024,
        use_torch_compile=USE_TORCH_COMPILE,
        verbose=False,
    )
    return {**defaults, **params}

def _save_temp_audio(output_bytes: bytes, suffix=".wav") -> str:
    # Dia.save_audio writes directly from model output; if we call here,
    # we create a file path and later write via model.save_audio
    name = f"{int(time.time())}_{uuid.uuid4().hex[:8]}{suffix}"
    path = AUDIO_DIR / name
    with open(path, "wb") as f:
        f.write(output_bytes)
    return str(path)

def _write_model_audio_to(path: str, model: Dia, audio_obj: Any):
    # Dia exposes model.save_audio(filepath, audio)
    model.save_audio(path, audio_obj)

def _clean_text_and_overrides(raw_text: str) -> Tuple[str, Dict[str, Any]]:
    # pull inline {#tts ... #}
    text, overrides = parse_tts_controls(raw_text)
    # strip any stray whitespace
    return text.strip(), overrides

def _finalize_text(text: str, voice: Optional[str], language: Optional[str]) -> str:
    t = add_language_hint(text, language)
    t = add_voice_tag(t, voice)
    return t

def _maybe_load_audio_prompt(temp_file: Optional[str]) -> Optional[str]:
    if temp_file and os.path.exists(temp_file):
        return temp_file
    return None

@app.post("/api/generate")
async def vapi_generate(payload: Dict[str, Any]):
    """
    Vapi webhook endpoint.
    We synthesize when status == "stopped" (i.e., the assistant message is complete).
    """
    msg = payload.get("message") or {}
    status = (msg.get("status") or "").lower()

    # Quick ack for 'started'
    if status == "started":
        return JSONResponse({"status": "acknowledged", "reason": "generation will run on 'stopped'"})

    if status != "stopped":
        return JSONResponse({"status": "ignored", "reason": f"status '{status}' not handled"})

    text = extract_text_from_vapi_message(msg)
    if not text:
        return JSONResponse({"status": "ignored", "reason": "no assistant text found"})

    # Inline overrides
    text, inline_overrides = _clean_text_and_overrides(text)

    # Optional assistant overrides (voice/lang) via assistant metadata (if you store them there).
    # For now, we do not rely on it; you can add extraction if you pass such fields via msg["assistant"].
    voice = None
    language = None

    # Build final text
    final_text = _finalize_text(text, voice, language)

    # Hyperparameters
    params = _normalize_params({})
    params = apply_overrides(params, inline_overrides)

    model = model_mgr.get()

    try:
        # Generate
        audio = model.generate(
            final_text,
            use_torch_compile=params["use_torch_compile"],
            verbose=False,
            cfg_scale=params["cfg_scale"],
            temperature=params["temperature"],
            top_p=params["top_p"],
            cfg_filter_top_k=params["cfg_filter_top_k"],
            max_tokens=params["max_tokens"],
        )

        # Save to WAV
        out_name = f"{int(time.time())}_{uuid.uuid4().hex[:8]}.wav"
        out_path = AUDIO_DIR / out_name
        _write_model_audio_to(str(out_path), model, audio)

        return FileResponse(
            path=str(out_path),
            media_type="audio/wav",
            filename=out_name,
            headers={
                "X-TTS-Overrides": ",".join(f"{k}={inline_overrides[k]}" for k in inline_overrides),
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"generation failed: {e}")

@app.post("/api/generate-direct")
async def generate_direct(req: GenerateDirectRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="text cannot be empty")

    # Parse inline overrides inside text
    text, inline_overrides = _clean_text_and_overrides(req.text)
    final_text = _finalize_text(text, req.voice, req.language)

    # Map JSON params -> Dia params
    params = _normalize_params(
        dict(
            cfg_scale=req.cfg,
            temperature=req.temperature,
            top_p=req.top_p,
            cfg_filter_top_k=req.top_k,
            max_tokens=req.max_new_tokens,
        )
    )
    params = apply_overrides(params, inline_overrides)

    # Optional voice clone audio prompt path
    audio_prompt = _maybe_load_audio_prompt(req.audio_prompt_path)

    model = model_mgr.get()
    try:
        audio = model.generate(
            final_text,
            audio_prompt=audio_prompt,
            use_torch_compile=params["use_torch_compile"],
            verbose=True,
            cfg_scale=params["cfg_scale"],
            temperature=params["temperature"],
            top_p=params["top_p"],
            cfg_filter_top_k=params["cfg_filter_top_k"],
            max_tokens=params["max_tokens"],
        )
        out_name = f"{int(time.time())}_{uuid.uuid4().hex[:8]}.wav"
        out_path = AUDIO_DIR / out_name
        _write_model_audio_to(str(out_path), model, audio)

        return FileResponse(
            path=str(out_path),
            media_type="audio/wav",
            filename=out_name,
            headers={
                "X-Voice": req.voice or "",
                "X-Language": req.language or "",
                "X-TTS-Overrides": ",".join(f"{k}={inline_overrides[k]}" for k in inline_overrides),
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"generation failed: {e}")

@app.post("/api/generate-direct-multipart")
async def generate_direct_multipart(
    text: str = Form(...),
    voice: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    temperature: float = Form(1.3),
    top_p: float = Form(0.95),
    top_k: int = Form(35),
    cfg: float = Form(3.0),
    max_new_tokens: int = Form(1024),
    audio_prompt: Optional[UploadFile] = File(None),
):
    # Save uploaded prompt (if any)
    audio_prompt_path = None
    if audio_prompt is not None:
        suffix = Path(audio_prompt.filename or "prompt.wav").suffix or ".wav"
        tmp = UPLOADS_DIR / f"prompt_{uuid.uuid4().hex[:8]}{suffix}"
        content = await audio_prompt.read()
        with open(tmp, "wb") as f:
            f.write(content)
        audio_prompt_path = str(tmp)

    # Apply same flow as JSON version
    req = GenerateDirectRequest(
        text=text,
        voice=voice,
        language=language,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        cfg=cfg,
        max_new_tokens=max_new_tokens,
        audio_prompt_path=audio_prompt_path,
    )
    return await generate_direct(req)
