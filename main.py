# main.py
import os
import re
import time
import uuid
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, Response, Request, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import anyio

# Try preferred high-level API; fall back to transformers wrapper if missing
_DIA_HIGHLEVEL = True
try:
    from dia.model import Dia  # type: ignore
except Exception:
    _DIA_HIGHLEVEL = False
    from transformers import AutoProcessor, DiaForConditionalGeneration  # type: ignore

# ----------------------------- Config -----------------------------
MODEL_ID = os.getenv("DIA_MODEL_ID", "nari-labs/Dia-1.6B-0626")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
DEFAULT_SR = int(os.getenv("DIA_SAMPLE_RATE", "24000"))

# Security (optional). If set, requests must include Authorization: Bearer <token>
REQUIRE_AUTH = os.getenv("AUTH_REQUIRED", "false").lower() in ("1", "true", "yes")
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "")

# Files
AUDIO_DIR = Path("./audio_files"); AUDIO_DIR.mkdir(exist_ok=True, parents=True)

# Logging helper
def log(msg: str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

# ------------------------ Inline Sound Effects --------------------
# Lightweight overlays to guarantee an audible cue even if the model
# ignores the textual effect token.
SFX = {
    "laughs":   {"f": 240, "dur": 2.0},
    "sighs":    {"f": 160, "dur": 1.6},
    "humming":  {"f": 220, "dur": 1.8},
    "exhales":  {"f": 140, "dur": 1.2},
    "gasps":    {"f": 400, "dur": 0.8},
    "groans":   {"f": 120, "dur": 1.4},
}
def synth_sfx(name: str, sr: int) -> np.ndarray:
    cfg = SFX.get(name, {"f": 200, "dur": 1.0})
    t = np.linspace(0, cfg["dur"], int(cfg["dur"] * sr), endpoint=False)
    # decaying sinusoid + a sprinkle of noise
    y = (np.sin(2*np.pi*cfg["f"]*t) * np.exp(-2.5*t) + 0.01*np.random.randn(t.shape[0])).astype(np.float32)
    y /= (np.max(np.abs(y)) + 1e-9)
    return 0.45 * y

def mix_effect(wav_path: str, effect: Optional[str]) -> str:
    if not effect:
        return wav_path
    try:
        y, sr = sf.read(wav_path)
        if y.ndim > 1:
            y = y[:, 0]
        sfx = synth_sfx(effect, sr)
        # position near the end, duck slightly
        insert = max(0, min(len(y) - int(0.25 * sr), len(y)))
        out = np.zeros(max(len(y), insert + len(sfx)), dtype=np.float32)
        out[:len(y)] = y.astype(np.float32)
        s, e = insert, insert + len(sfx)
        out[s:e] *= 0.75
        out[s:e] += sfx
        out /= (np.max(np.abs(out)) + 1e-9)
        sf.write(wav_path, out, sr)
    except Exception as e:
        log(f"[SFX] mix failed: {e}")
    return wav_path

def to_mp3_if_needed(path_wav: str, want_mp3: bool) -> str:
    if not want_mp3:
        return path_wav
    mp3 = path_wav.replace(".wav", ".mp3")
    # ffmpeg should be present in your container
    code = os.system(f'ffmpeg -y -loglevel error -i "{path_wav}" -ar {DEFAULT_SR} -ac 1 "{mp3}"')
    return mp3 if code == 0 and os.path.exists(mp3) else path_wav

# --------------------------- Text Parsing -------------------------
# (laughs) → capture 1st supported effect, strip all parentheses blocks
def strip_effects(text: str) -> Tuple[str, Optional[str]]:
    matches = re.findall(r"\(([^)]+)\)", text)
    clean = re.sub(r"\([^)]+\)", "", text).strip()
    eff = None
    for m in matches:
        k = m.strip().lower()
        if k in SFX:
            eff = k
            break
    return clean, eff

# {#tts temperature=1.1 top_p=0.9 cfg=3.4 max_new_tokens=400 #}
_TTS_KEYS = {"temperature": float, "top_p": float, "top_k": int, "cfg": float, "max_new_tokens": int}
def strip_overrides(text: str) -> Tuple[str, Dict[str, Any]]:
    m = re.search(r"\{\#tts\s+([^}]*)\#\}", text, flags=re.I)
    if not m:
        return text.strip(), {}
    inner = m.group(1)
    pairs = dict(re.findall(r"([a-zA-Z_]+)\s*=\s*([-\d\.]+)", inner))
    ov = {}
    for k, cast in _TTS_KEYS.items():
        if k in pairs:
            try:
                ov[k] = cast(pairs[k])
            except Exception:
                pass
    clean = (text[:m.start()] + text[m.end():]).strip()
    return clean, ov

# --------------------------- Dia Manager -------------------------
class DiaManager:
    def __init__(self):
        self.loaded = False
        self.hi_model = None  # dia.model.Dia
        self.tf_model = None  # transformers model
        self.tf_proc = None

    def load(self):
        if self.loaded:
            return
        log(f"[DIA] Loading model={MODEL_ID} on {DEVICE} (dtype={DTYPE}) high_level={_DIA_HIGHLEVEL}")
        try:
            if _DIA_HIGHLEVEL:
                # High-level API preferred
                compute_dtype = "float16" if DEVICE == "cuda" else "float32"
                self.hi_model = Dia.from_pretrained(MODEL_ID, compute_dtype=compute_dtype)
            else:
                # Fallback to transformers wrapper
                self.tf_proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
                self.tf_model = DiaForConditionalGeneration.from_pretrained(
                    MODEL_ID,
                    torch_dtype=DTYPE,
                    device_map="auto" if DEVICE == "cuda" else None,
                    trust_remote_code=True,
                )
            self.loaded = True
            log("[DIA] Model loaded.")
        except Exception as e:
            log(f"[DIA] Load failed: {e}")
            raise

    async def tts(self, text: str, overrides: Dict[str, Any]) -> str:
        """
        Generate to a temp WAV file and return its path.
        """
        if not self.loaded:
            self.load()

        # Defaults close to repo examples
        gen = {
            "max_new_tokens": int(overrides.get("max_new_tokens", 1024)),
            "cfg_scale": float(overrides.get("cfg", 3.0)),   # hi-level key
            "guidance_scale": float(overrides.get("cfg", 3.0)),  # tf key
            "temperature": float(overrides.get("temperature", 1.3)),
            "top_p": float(overrides.get("top_p", 0.95)),
            "top_k": int(overrides.get("top_k", 35)),
        }

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=AUDIO_DIR) as tmp:
            out_path = tmp.name

        if self.hi_model is not None:
            # Use high-level API (preferred)
            def _run_hi():
                audio = self.hi_model.generate(
                    text,
                    use_torch_compile=False,
                    verbose=False,
                    cfg_scale=gen["cfg_scale"],
                    temperature=gen["temperature"],
                    top_p=gen["top_p"],
                    cfg_filter_top_k=gen["top_k"],
                    # max_new_tokens param name may differ in high level; Dia trims automatically.
                )
                self.hi_model.save_audio(out_path, audio)
            await anyio.to_thread.run_sync(_run_hi)
            return out_path

        # Transformers wrapper
        proc, model = self.tf_proc, self.tf_model
        inputs = proc(text=[text], padding=True, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        def _run_tf():
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=gen["max_new_tokens"],
                    guidance_scale=gen["guidance_scale"],
                    temperature=gen["temperature"],
                    top_p=gen["top_p"],
                    top_k=gen["top_k"],
                )
            decoded = proc.batch_decode(outputs)
            proc.save_audio(decoded, out_path)

        await anyio.to_thread.run_sync(_run_tf)
        return out_path

dia = DiaManager()
dia.load()

# --------------------------- Auth -------------------------------
def require_bearer(authorization: Optional[str] = Header(default=None)) -> None:
    if not REQUIRE_AUTH:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = authorization.split(" ", 1)[1].strip()
    if token != AUTH_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")

# --------------------------- Schemas ----------------------------
class DirectIn(BaseModel):
    text: str = Field(..., description="Plain text; may include (laughs) and {#tts ... #}")
    language: Optional[str] = Field(None, description="Language hint, e.g., 'en', 'de'")
    voice: Optional[str] = Field(None, description="Voice hint to steer Dia style")
    format: Optional[str] = Field("wav", description="wav | mp3")
    # Optional TTS overrides
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    cfg: Optional[float] = None
    max_new_tokens: Optional[int] = None

class UltravoxTTSIn(DirectIn):
    pass

# --------------------------- FastAPI ----------------------------
app = FastAPI(title="Dia-1.6B TTS Bridge (Ultravox / Vapi)", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "dtype": str(DTYPE).split(".")[-1],
        "model": MODEL_ID,
        "loaded": dia.loaded,
        "sample_rate": DEFAULT_SR,
    }

@app.get("/effects")
def effects():
    return {"supported": list(SFX.keys())}

# ------------------- Helpers for request handling ----------------
def build_final_text(text: str, language: Optional[str], voice: Optional[str]) -> Tuple[str, Optional[str], Dict[str, Any]]:
    # Parse effect + overrides
    text_no_tags, effect = strip_effects(text)
    text_clean, overrides = strip_overrides(text_no_tags)
    # Prepend hints to guide Dia styling (simple, non-invasive)
    pre = []
    if language:
        pre.append(f"[lang={language}]")
    if voice:
        pre.append(f"[voice={voice}]")
    final_text = (" ".join(pre) + " " + text_clean).strip()
    return final_text, effect, overrides

async def synth_and_pack(
    text: str,
    language: Optional[str],
    voice: Optional[str],
    fmt: str,
    body_overrides: Dict[str, Any],
) -> Response:
    final_text, effect, overrides = build_final_text(text, language, voice)
    # body fields override inline {#tts ... #}
    overrides.update({k: v for k, v in body_overrides.items() if v is not None})

    log(f"[TTS] text='{final_text[:80]}...' effect={effect} ov={overrides}")
    wav_path = await dia.tts(final_text, overrides)
    wav_path = mix_effect(wav_path, effect)
    out_path = to_mp3_if_needed(wav_path, (fmt or "wav").lower() == "mp3")

    media = "audio/mpeg" if out_path.endswith(".mp3") else "audio/wav"
    with open(out_path, "rb") as f:
        audio = f.read()
    return Response(
        content=audio,
        media_type=media,
        headers={"X-Provider": "Dia-1.6B", "X-Effect": effect or "", "X-Model": MODEL_ID},
    )

# -------------------- Ultravox Custom TTS -----------------------
@app.post("/ultravox/tts")
async def ultravox_tts(body: UltravoxTTSIn, _auth: None = Depends(require_bearer)):
    if not body.text or not body.text.strip():
        raise HTTPException(400, "text is required")
    return await synth_and_pack(
        text=body.text,
        language=body.language,
        voice=body.voice,
        fmt=body.format or "wav",
        body_overrides=dict(
            temperature=body.temperature,
            top_p=body.top_p,
            top_k=body.top_k,
            cfg=body.cfg,
            max_new_tokens=body.max_new_tokens,
        ),
    )

# -------------------- Direct test (Postman) ---------------------
@app.post("/api/generate-direct")
async def generate_direct(body: DirectIn, _auth: None = Depends(require_bearer)):
    if not body.text or not body.text.strip():
        raise HTTPException(400, "text is required")
    return await synth_and_pack(
        text=body.text,
        language=body.language,
        voice=body.voice,
        fmt=body.format or "wav",
        body_overrides=dict(
            temperature=body.temperature,
            top_p=body.top_p,
            top_k=body.top_k,
            cfg=body.cfg,
            max_new_tokens=body.max_new_tokens,
        ),
    )

# -------------------- Vapi Custom Voice (speech-update) ---------
def _extract_text_from_vapi(payload: Dict[str, Any]) -> str:
    msg = (payload or {}).get("message", {})
    art = msg.get("artifact") or {}
    # Try OpenAI-formatted
    mo = art.get("messagesOpenAIFormatted") or []
    for m in reversed(mo):
        role = (m.get("role") or "").lower()
        if role in ("assistant", "bot"):
            c = m.get("content")
            if isinstance(c, str):
                return c.strip()
            if isinstance(c, list):
                for part in c:
                    if isinstance(part, dict) and part.get("type") == "text":
                        t = part.get("text", "")
                        if t:
                            return t.strip()
    # Fallback raw messages
    mr = art.get("messages") or []
    for m in reversed(mr):
        role = (m.get("role") or "").lower()
        if role in ("assistant", "bot"):
            t = str(m.get("message", "")).strip()
            if t:
                return t
    # Fallback: firstMessage when turn=0
    if (msg.get("turn") == 0):
        a = msg.get("assistant") or {}
        fm = a.get("firstMessage")
        if isinstance(fm, str) and fm.strip():
            return fm.strip()
    return ""

@app.post("/api/generate")
async def vapi_generate(request: Request, _auth: None = Depends(require_bearer)):
    """
    Vapi 'custom-voice' webhook. We return raw audio (audio/wav).
    - For 'started' we just ack.
    - For 'stopped' we synthesize from the assistant's last message.
    """
    payload = await request.json()
    msg = (payload or {}).get("message", {})
    status = (msg.get("status") or "").lower()
    if status == "started":
        return {"status": "acknowledged"}

    if status not in ("stopped", "completed", ""):
        return {"status": "ignored", "reason": f"status '{status}' not handled"}

    text = _extract_text_from_vapi(payload)
    if not text:
        return {"status": "ignored", "reason": "no assistant text found"}

    # No external language/voice provided here; rely on inline tags if any
    resp = await synth_and_pack(
        text=text, language=None, voice=None, fmt="wav",
        body_overrides={},  # Vapi controls can be embedded via {#tts ... #}
    )
    return resp

# -------------------------- Run tip -----------------------------
# uvicorn main:app --host 0.0.0.0 --port 8080
