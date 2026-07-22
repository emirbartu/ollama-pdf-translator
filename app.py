import base64
import ipaddress
import json
import os
import queue
import shutil
import socket
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Optional
from urllib.parse import unquote, urlparse

import fitz
import uvicorn
from fastapi import FastAPI, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from main import translate_pdf

MAX_BYTES = 1 * 1024 * 1024
MAX_PAGES = 3
MAX_OUTPUT_BYTES = 3_200_000
MODEL = os.environ.get("LLM_MODEL") or ("gpt-oss:120b" if os.environ.get("VERCEL") else "llama3.2")

LANGUAGES = (
    "Arabic", "Chinese", "Dutch", "English", "French", "German", "Greek",
    "Hebrew", "Hindi", "Italian", "Japanese", "Korean", "Norwegian",
    "Polish", "Portuguese", "Russian", "Spanish", "Swedish", "Turkish",
    "Ukrainian",
)
ALLOWED_LANGUAGES = frozenset(LANGUAGES)

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Ollama PDF Translator")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

if not os.environ.get("VERCEL"):
    app.mount("/static", StaticFiles(directory="public/static"), name="static")

templates = Jinja2Templates(directory="templates")
job_lock = threading.Lock()


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    server_fallback = bool(os.environ.get("OPENAI_API_KEY")) or not os.environ.get("VERCEL")
    return templates.TemplateResponse(request, "index.html", {"languages": LANGUAGES, "server_fallback": server_fallback})


@app.get("/self-host", response_class=HTMLResponse)
def self_host(request: Request):
    return templates.TemplateResponse(request, "selfhost.html", {"languages": LANGUAGES})


@app.post("/translate")
@limiter.limit("5/hour")
@limiter.limit("20/day")
async def translate(
    request: Request,
    file: UploadFile,
    source_lang: str = Form("English"),
    target_lang: str = Form("Turkish"),
    preserve_layout: bool = Form(True),
    api_key: str = Form(""),
    base_url: str = Form(""),
    model: str = Form(""),
):
    if source_lang not in ALLOWED_LANGUAGES or target_lang not in ALLOWED_LANGUAGES:
        return HTMLResponse(_ndjson_error("Unsupported language selection."), media_type="application/x-ndjson")

    if api_key.strip():
        eff_base = base_url.strip() or "https://api.openai.com/v1"
        eff_model = model.strip() or "gpt-4o-mini"
        eff_key = api_key.strip()
        if os.environ.get("VERCEL"):
            validation_error = _validate_visitor_url(eff_base)
            if validation_error:
                return HTMLResponse(_ndjson_error(validation_error), media_type="application/x-ndjson")
    elif os.environ.get("OPENAI_API_KEY"):
        eff_base = os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        eff_key = os.environ.get("OPENAI_API_KEY")
        eff_model = MODEL
    elif not os.environ.get("VERCEL"):
        eff_base = os.environ.get("OPENAI_BASE_URL")  # None lets main.py auto-detect local Ollama
        eff_key = None
        eff_model = MODEL
    else:
        return HTMLResponse(_ndjson_error("Enter your API key to use the cloud demo."), media_type="application/x-ndjson")

    content_length = request.headers.get("content-length", "")
    if content_length.isdigit() and int(content_length) > MAX_BYTES + 65_536:  # multipart overhead
        return HTMLResponse(_ndjson_error("File too large — 1 MB max."), media_type="application/x-ndjson")
    data = await file.read()
    error = _validate(data)
    if error:
        return HTMLResponse(_ndjson_error(error), media_type="application/x-ndjson")
    if not job_lock.acquire(blocking=False):
        return HTMLResponse(
            _ndjson_error("Another translation is in progress — please try again shortly."),
            media_type="application/x-ndjson",
        )
    stem = Path(file.filename or "document.pdf").stem
    return StreamingResponse(
        _stream_translation(data, stem, source_lang, target_lang, preserve_layout, eff_base, eff_key, eff_model),
        media_type="application/x-ndjson",
    )


@app.get("/api/ollama/models")
@limiter.limit("30/minute")
def ollama_models(request: Request):
    """List locally installed Ollama models (only meaningful when self-hosted locally)."""
    if not shutil.which("ollama"):
        return {"available": False, "models": [], "error": "Ollama is not installed on this machine."}

    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
    except subprocess.TimeoutExpired:
        return {"available": False, "models": [], "error": "Ollama did not respond in time."}

    if result.returncode != 0:
        return {"available": False, "models": [], "error": "Ollama is installed but not responding — start it with `ollama serve`."}

    models = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("NAME"):
            continue
        models.append(line.split()[0])

    return {"available": True, "models": models, "error": None}


def _stream_translation(
    data: bytes,
    stem: str,
    source_lang: str,
    target_lang: str,
    preserve_layout: bool,
    eff_base: Optional[str],
    eff_key: Optional[str],
    eff_model: str,
):
    q = queue.Queue()
    tmpdir = Path(tempfile.mkdtemp(prefix="pdfjob_"))

    def work():
        try:
            input_path = tmpdir / "input.pdf"
            output_path = tmpdir / "output.pdf"
            input_path.write_bytes(data)
            result = translate_pdf(
                input_path,
                output_path,
                source_lang=source_lang,
                target_lang=target_lang,
                model=eff_model,
                preserve_layout=preserve_layout,
                base_url=eff_base,
                api_key=eff_key,
                progress_callback=lambda k, n: q.put(("progress", k, n)),
            )
            q.put(("done", output_path) if result else ("error", None))
        except Exception:
            q.put(("error", None))

    threading.Thread(target=work, daemon=True).start()

    try:
        while True:
            kind, *rest = q.get()
            if kind == "progress":
                page, total = rest
                percent = min(round(page / total * 100), 100) if total else 0
                yield json.dumps({"type": "progress", "page": page, "total": total, "percent": percent}) + "\n"
            elif kind == "done":
                output_bytes = rest[0].read_bytes()
                if len(output_bytes) > MAX_OUTPUT_BYTES:
                    yield _ndjson_error("Translated PDF is too large to return.")
                else:
                    yield json.dumps({
                        "type": "done",
                        "filename": f"{stem}_translated.pdf",
                        "pdf_b64": base64.b64encode(output_bytes).decode(),
                    }) + "\n"
                return
            else:
                yield _ndjson_error("Translation failed. Please try again.")
                return
    finally:
        job_lock.release()
        shutil.rmtree(tmpdir, ignore_errors=True)


def _ndjson_error(message: str) -> str:
    return json.dumps({"type": "error", "message": message}) + "\n"


def _validate(data: bytes):
    if not data:
        return "No file received."
    if not data.startswith(b"%PDF-"):
        return "Only PDF files are supported."
    if len(data) > MAX_BYTES:
        return "File too large — 1 MB max."
    try:
        with fitz.open(stream=data, filetype="pdf") as doc:
            if len(doc) > MAX_PAGES:
                return f"PDF too long — {MAX_PAGES} pages max for this demo."
    except Exception:
        return "Could not read this PDF."
    return None


def _validate_visitor_url(url: str) -> Optional[str]:
    """Hardened SSRF guard for visitor-provided OpenAI-compatible endpoints.

    Returns a user-friendly error message if the URL is not allowed, otherwise None.
    """
    raw = unquote(url)
    parsed = urlparse(raw)

    if parsed.scheme != "https":
        return "Only https:// endpoints are allowed."

    host = (parsed.hostname or "").lower().strip()
    if not host:
        return "That endpoint is not allowed."

    # Strip IPv6 brackets
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]

    if host in ("localhost", "localhost.localdomain"):
        return "That endpoint is not allowed."
    if host.endswith(".local") or host.endswith(".internal") or host.endswith(".localhost"):
        return "That endpoint is not allowed."

    ip: Optional[ipaddress.IPv4Address | ipaddress.IPv6Address] = None
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        try:
            ip = ipaddress.IPv4Address(socket.inet_aton(host))
        except (OSError, ValueError):
            pass

    if ip is not None:
        if isinstance(ip, ipaddress.IPv6Address):
            mapped = ip.ipv4_mapped
            if mapped is not None:
                ip = mapped
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
            or not ip.is_global
        ):
            return "That endpoint is not allowed."
        if str(ip) in ("169.254.169.254", "100.100.100.200"):
            return "That endpoint is not allowed."
        if isinstance(ip, ipaddress.IPv6Address) and str(ip) == "fd00:ec2::254":
            return "That endpoint is not allowed."

    return None


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000)
