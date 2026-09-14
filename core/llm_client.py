"""
Local LLM client for MARK XL.

Supports two backends — selected via  "llm_provider"  in config/api_keys.json:

  "llm_provider": "ollama"   (default)
        Uses Ollama's native /api/chat endpoint.
        Download: https://ollama.com
        Default port: 11434

  "llm_provider": "openai"
        Uses any OpenAI-compatible server: LM Studio, Jan, LocalAI,
        llama.cpp server, vLLM, etc.
        LM Studio download: https://lmstudio.ai   (default port: 1234)
        Set  "llm_url": "http://localhost:1234"  in config.
        Note: tool-calling support depends on the model; use a model that
        supports function/tool calls (e.g. Qwen2.5, Llama-3.1, Mistral).
"""
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Generator

import requests

# Matches a sentence boundary: [.!?] followed by whitespace, or a blank line.
# Avoids splitting on decimals (3.5) because those have no space after the dot.
_SENT_END = re.compile(r'(?<=[.!?])\s+|(?<=\n)\s*\n')


def _clean_reasoning(text: str) -> str:
    """Strip <think>...</think> internal reasoning blocks from reasoning/thinking models (e.g. Nemotron/DeepSeek)."""
    if not text:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "</think>" in cleaned:
        cleaned = re.sub(r"^.*?</think>", "", cleaned, flags=re.DOTALL)
    if "<think>" in cleaned:
        cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()

def get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


BASE_DIR    = get_base_dir()
CONFIG_PATH = BASE_DIR / "config" / "api_keys.json"

_DEFAULTS = {
    "llm_url":      "https://integrate.api.nvidia.com/v1",
    "llm_model":    "nvidia/nemotron-3-super-120b-a12b",
    "llm_provider": "nvidia",   # "nvidia" | "ollama" | "openai"
}


def get_llm_provider() -> str:
    """Returns 'nvidia', 'openai', or 'ollama'."""
    cfg = _load_config()
    raw = cfg.get("llm_provider", "").strip().lower()
    if raw in ("nvidia", "nim"):
        return "nvidia"
    if raw in ("openai", "lmstudio", "localai", "jan", "llamacpp"):
        return "openai"
    if raw == "ollama":
        return "ollama"
    if cfg.get("nvidia_api_key"):
        return "nvidia"
    return "nvidia"


def _load_config() -> dict:
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def get_auth_headers() -> dict:
    """Returns authorization headers for NVIDIA NIM or OpenAI-compatible endpoints."""
    cfg = _load_config()
    provider = get_llm_provider()
    headers = {"Content-Type": "application/json"}
    key = ""
    if provider == "nvidia":
        key = cfg.get("nvidia_api_key", "").strip()
    elif provider == "openai":
        key = cfg.get("openai_api_key", "").strip() or cfg.get("nvidia_api_key", "").strip()

    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


_SESSION: requests.Session | None = None

def _get_session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        _SESSION = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=8, pool_maxsize=16, max_retries=1)
        _SESSION.mount("https://", adapter)
        _SESSION.mount("http://", adapter)
    return _SESSION


def ensure_ollama_running(timeout: int = 15) -> bool:
    """
    For NVIDIA: verifies API key is configured.
    For OpenAI-compatible providers: pings /v1/models.
    For Ollama: pings /api/tags; auto-launches 'ollama serve' if not running.
    """
    url, _   = get_llm_settings()
    provider = get_llm_provider()

    if provider == "nvidia":
        cfg = _load_config()
        key = cfg.get("nvidia_api_key", "").strip()
        if not key or not key.startswith("nvapi-"):
            print("[LLM] WARNING: nvidia_api_key is missing or invalid in config/api_keys.json.")
            return False
        return True

    if provider == "openai":
        health = f"{url}/v1/models"
        headers = get_auth_headers()
        try:
            ok = requests.get(health, headers=headers, timeout=5).status_code == 200
            if ok:
                print(f"[LLM] OpenAI-compatible server reachable at {url}")
            return ok
        except Exception:
            return False

    # ── Ollama ──────────────────────────────────────────────────────────────
    health = f"{url}/api/tags"

    def _is_up() -> bool:
        try:
            return requests.get(health, timeout=3).status_code == 200
        except Exception:
            return False

    if _is_up():
        return True

    print("[LLM] Ollama not running — launching 'ollama serve'…")
    try:
        kwargs: dict = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        subprocess.Popen(["ollama", "serve"], **kwargs)
    except FileNotFoundError:
        print("[LLM] 'ollama' command not found. Install Ollama from https://ollama.com")
        return False
    except Exception as e:
        print(f"[LLM] Could not launch Ollama: {e}")
        return False

    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(1.0)
        if _is_up():
            print("[LLM] Ollama started successfully.")
            return True

    print("[LLM] Ollama did not respond within the timeout.")
    return False


def warmup_model(system_prompt: str | None = None) -> bool:
    """
    Pre-load the model AND prime Ollama's KV prefix cache.

    Why the system_prompt matters
    ─────────────────────────────
    Ollama caches the KV attention state of the prompt prefix across requests.
    If warmup includes the same system prompt that real requests will use, Ollama
    evaluates those tokens ONCE at startup.  Every subsequent request only needs
    to evaluate the small delta (user message ± time context) instead of the full
    300-500 token system prompt → drops first-token latency from ~17 s to <1 s.

    Pass the *static* part of the system prompt (the JARVIS protocol text, without
    timestamps or per-minute context) so the prefix stays valid across calls.
    """
    url, model = get_llm_settings()
    provider   = get_llm_provider()
    print(f"[LLM] Warming up '{model}' ({provider})…")

    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": "hi"})

    if provider == "openai":
        # OpenAI-compatible: just fire a minimal request to ensure the model is loaded.
        # No keep_alive or KV-cache priming available — server manages this internally.
        payload = {
            "model":      model,
            "messages":   messages,
            "stream":     False,
            "max_tokens": 1,
        }
        try:
            resp = requests.post(f"{url}/v1/chat/completions", json=payload, timeout=180)
            resp.raise_for_status()
            print(f"[LLM] '{model}' ready (OpenAI-compatible server).")
            return True
        except Exception as e:
            print(f"[LLM] Warmup failed (non-fatal): {e}")
            return False

    # ── Ollama ──────────────────────────────────────────────────────────────
    payload = {
        "model":      model,
        "messages":   messages,
        "stream":     False,
        "keep_alive": -1,
        # num_gpu:99 → push ALL transformer layers to GPU (Ollama caps at available)
        # This is safe even without a GPU — Ollama silently ignores if n_gpu_layers=0
        "options":    {"num_predict": 1, "num_gpu": 99},
    }
    try:
        resp = requests.post(f"{url}/api/chat", json=payload, timeout=180)
        resp.raise_for_status()
        print(f"[LLM] '{model}' loaded and KV cache primed.")
        return True
    except Exception as e:
        print(f"[LLM] Warmup failed (non-fatal): {e}")
        return False


def check_model_available(log: Callable | None = None) -> bool:
    """
    Returns True if the configured model is already pulled in Ollama.
    Logs an actionable warning (to console + optional UI callback) if not.
    Always returns True for non-Ollama providers (cannot inspect their model list).
    """
    if get_llm_provider() != "ollama":
        return True

    url, model = get_llm_settings()
    try:
        resp = requests.get(f"{url}/api/tags", timeout=5)
        resp.raise_for_status()
        pulled = [m.get("name", "") for m in resp.json().get("models", [])]
        model_base = model.split(":")[0]
        found = any(
            m == model or m == f"{model}:latest" or m == model_base or m.startswith(model_base + ":")
            for m in pulled
        )
        if not found:
            available = ", ".join(pulled) if pulled else "none"
            warn = (
                f"WRN: Model '{model}' is not pulled in Ollama.\n"
                f"     Available: {available}\n"
                f"     Fix: ollama pull {model}"
            )
            print(warn)
            if log:
                log(f"WRN: '{model}' not found — run: ollama pull {model}")
        return found
    except Exception:
        return True   # Ollama might still be starting up; non-blocking


def check_llm_readiness(auto_pull: bool = True) -> tuple[bool, str]:
    """
    Diagnostic & readiness check for LLM backend.
    Logs:
      [LLM] Ollama URL: ...
      [LLM] Ollama server: READY/NOT READY
      [LLM] Available models: ...
      [LLM] Selected model: ...
    Verifies that the configured model is available locally.
    """
    url, model = get_llm_settings()
    provider = get_llm_provider()

    print(f"[LLM] Ollama URL: {url}")

    if not ensure_ollama_running(timeout=15):
        print("[LLM] Ollama server: NOT READY")
        return False, f"Ollama server is not reachable at {url}"

    print("[LLM] Ollama server: READY")

    if provider == "ollama":
        try:
            resp = requests.get(f"{url}/api/tags", timeout=5)
            if resp.status_code == 200:
                raw_models = resp.json().get("models", [])
                available = [m.get("name", "") for m in raw_models]
                avail_str = ", ".join(available) if available else "None"
                print(f"[LLM] Available models: {avail_str}")
                print(f"[LLM] Selected model: {model}")

                model_base = model.split(":")[0]
                model_found = any(
                    m == model or m == f"{model}:latest" or m == model_base or m.startswith(f"{model_base}:")
                    for m in available
                )
                if not model_found:
                    print(f"[LLM] Model '{model}' not found in local Ollama repository.")
                    if auto_pull:
                        print(f"[LLM] Attempting to pull '{model}'...")
                        res = subprocess.run(["ollama", "pull", model], capture_output=True, text=True)
                        if res.returncode == 0:
                            print(f"[LLM] Model '{model}' pulled successfully.")
                            return True, f"Model '{model}' ready"
                        else:
                            print(f"[LLM] Failed to pull '{model}': {res.stderr}")
                    return False, f"Model '{model}' is not pulled. Run: ollama pull {model}"
                return True, f"Model '{model}' ready"
        except Exception as e:
            print(f"[LLM] Error querying /api/tags: {e}")
            return False, str(e)

    return True, "LLM server ready"


def get_llm_settings() -> tuple[str, str]:
    """Returns (base_url, model_name)."""
    cfg   = _load_config()
    url   = cfg.get("llm_url",   _DEFAULTS["llm_url"]).rstrip("/")
    model = cfg.get("llm_model", _DEFAULTS["llm_model"])
    return url, model


def call_llm(
    messages: list,
    tools:    list | None = None,
    timeout:  int = 120,
) -> dict:
    """
    Non-streaming chat request.  Routes to Ollama, NVIDIA NIM, or OpenAI-compatible backend.

    Returns:
        {"content": str, "tool_calls": list}
    """
    url, model = get_llm_settings()
    provider   = get_llm_provider()

    if provider in ("openai", "nvidia"):
        endpoint = f"{url}/v1/chat/completions" if not url.endswith("/v1") else f"{url}/chat/completions"
        headers  = get_auth_headers()
        cfg      = _load_config()
        max_tok  = int(cfg.get("max_tokens", 2048))
        # Sanitize messages: ensure content is always a valid string, never None
        cleaned_messages = []
        for msg in messages:
            mc = dict(msg)
            if mc.get("content") is None:
                mc["content"] = ""
            cleaned_messages.append(mc)

        models_to_try = [model]
        if provider == "nvidia" and "120b" in model:
            models_to_try.append("nvidia/nemotron-3-nano-omni-30b-a3b-reasoning")

        session = _get_session()
        last_error = None

        for cur_model in models_to_try:
            payload: dict = {
                "model":      cur_model,
                "messages":   cleaned_messages,
                "stream":     False,
                "max_tokens": max_tok,
            }
            if tools:
                payload["tools"] = tools

            enable_think = cfg.get("enable_thinking", False)
            if "nemotron" in cur_model.lower():
                payload["chat_template_kwargs"] = {"enable_thinking": enable_think, "force_nonempty_content": True}
            else:
                payload["chat_template_kwargs"] = {"enable_thinking": False}

            for attempt in range(3):
                try:
                    # On retry attempts, drop non-essential template kwargs to recover from backend template errors
                    req_payload = dict(payload)
                    if attempt > 0:
                        req_payload.pop("chat_template_kwargs", None)

                    resp = session.post(endpoint, json=req_payload, headers=headers, timeout=timeout)
                    if resp.status_code in (500, 502, 503, 504, 429) and attempt < 2:
                        time.sleep(1.0 * (attempt + 1))
                        continue

                    if resp.status_code >= 400:
                        err_text = ""
                        try:
                            err_text = resp.json().get("error", {}).get("message") or resp.text
                        except Exception:
                            err_text = resp.text
                        print(f"[NVIDIA] HTTP {resp.status_code} ({cur_model}): {err_text[:200]}")
                        if resp.status_code in (500, 502, 503, 504) and cur_model != models_to_try[-1]:
                            last_error = f"HTTP {resp.status_code}: {err_text[:200]}"
                            break

                    resp.raise_for_status()
                    choice = resp.json().get("choices", [{}])[0]
                    msg    = choice.get("message", {})
                    raw_tc  = msg.get("tool_calls") or []
                    tc_list = [
                        {
                            "id":       t.get("id", ""),
                            "function": {
                                "name":      t["function"]["name"],
                                "arguments": (
                                    json.loads(t["function"]["arguments"])
                                    if isinstance(t["function"].get("arguments"), str)
                                    else t["function"].get("arguments", {})
                                ),
                            },
                        }
                        for t in raw_tc
                    ]
                    content = _clean_reasoning(msg.get("content") or "")
                    # If tool calls were generated and content has thinking leftovers, clear it
                    if tc_list and ("I will" in content or "Let's" in content or "I need to" in content):
                        content = ""
                    return {
                        "content":    content,
                        "tool_calls": tc_list,
                    }
                except requests.exceptions.HTTPError as e:
                    if attempt < 2 and e.response is not None and e.response.status_code in (500, 502, 503, 504, 429):
                        time.sleep(1.0 * (attempt + 1))
                        continue
                    last_error = e
                except Exception as e:
                    if attempt < 2:
                        time.sleep(1.0)
                        continue
                    last_error = e

        raise RuntimeError(f"{provider.upper()} LLM call failed: {last_error}")

    # ── Ollama ──────────────────────────────────────────────────────────────
    endpoint = f"{url}/api/chat"
    payload = {
        "model":      model,
        "messages":   messages,
        "stream":     False,
        "keep_alive": -1,
        "options":    {"num_predict": 150, "num_gpu": 99},
    }
    if tools:
        payload["tools"] = tools

    try:
        resp = requests.post(endpoint, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        msg  = data.get("message", {})
        return {
            "content":    (msg.get("content") or "").strip(),
            "tool_calls": msg.get("tool_calls") or [],
        }
    except requests.exceptions.ConnectionError as e:
        print(f"[LLM] ConnectionError — trying to restart Ollama… ({e})")
        if ensure_ollama_running():
            try:
                resp = requests.post(endpoint, json=payload, timeout=timeout)
                resp.raise_for_status()
                data = resp.json()
                msg  = data.get("message", {})
                return {
                    "content":    (msg.get("content") or "").strip(),
                    "tool_calls": msg.get("tool_calls") or [],
                }
            except Exception:
                pass
        raise RuntimeError(
            f"Cannot connect to Ollama at {url}. "
            "Make sure Ollama is installed and run: ollama serve"
        )
    except requests.exceptions.Timeout:
        raise RuntimeError("Ollama request timed out after 120 s.")
    except requests.exceptions.HTTPError as e:
        err_detail = ""
        try:
            err_detail = e.response.json().get("error", "")
        except Exception:
            err_detail = e.response.text if e.response is not None else ""
        if e.response is not None and e.response.status_code == 404:
            print(f"[LLM] HTTP 404 Not Found from {endpoint}: {err_detail or 'Model or endpoint not found'}")
            raise RuntimeError(f"Ollama HTTP 404 Not Found: {err_detail or f'Model {model} not found'}")
        print(f"[LLM] HTTPError: {e.response.status_code if e.response else 'unknown'} — {err_detail[:200]}")
        raise RuntimeError(f"Ollama HTTP error {e.response.status_code if e.response else 'unknown'}: {err_detail or e}")
    except Exception as e:
        print(f"[LLM] Unexpected error: {type(e).__name__}: {e}")
        raise RuntimeError(f"LLM call failed: {e}")


def call_llm_text(
    prompt:  str,
    system:  str | None = None,
    model:   str | None = None,
    timeout: int = 120,
) -> str:
    """
    Simple text-only generation (no tools).
    Used by planner, executor, error_handler, code_helper, dev_agent.
    Routes to NVIDIA NIM, OpenAI, or Ollama.
    """
    url, default_model = get_llm_settings()
    provider           = get_llm_provider()
    m                  = model or default_model

    if provider == "nvidia":
        # Automatically map legacy Gemini, Anthropic, or OpenAI model strings to the configured NVIDIA model
        if not m or any(k in m.lower() for k in ("gemini", "flash", "pro", "claude", "gpt")):
            m = default_model

    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    if provider in ("openai", "nvidia"):
        endpoint = f"{url}/v1/chat/completions" if not url.endswith("/v1") else f"{url}/chat/completions"
        headers  = get_auth_headers()
        cfg      = _load_config()
        max_tok  = int(cfg.get("max_tokens", 2048))
        # Sanitize messages: ensure content is always a valid string, never None
        cleaned_messages = []
        for msg in messages:
            mc = dict(msg)
            if mc.get("content") is None:
                mc["content"] = ""
            cleaned_messages.append(mc)

        models_to_try = [m]
        if provider == "nvidia" and "120b" in m:
            models_to_try.append("nvidia/nemotron-3-nano-omni-30b-a3b-reasoning")

        session = _get_session()
        last_error = None

        for cur_model in models_to_try:
            payload = {
                "model":      cur_model,
                "messages":   cleaned_messages,
                "stream":     False,
                "max_tokens": max_tok,
            }
            if "nemotron" in cur_model.lower():
                payload["chat_template_kwargs"] = {"enable_thinking": False}

            for attempt in range(3):
                try:
                    req_payload = dict(payload)
                    if attempt > 0:
                        req_payload.pop("chat_template_kwargs", None)

                    resp = session.post(endpoint, json=req_payload, headers=headers, timeout=timeout)
                    if resp.status_code in (500, 502, 503, 504, 429) and attempt < 2:
                        time.sleep(1.0 * (attempt + 1))
                        continue

                    if resp.status_code >= 400:
                        err_text = ""
                        try:
                            err_text = resp.json().get("error", {}).get("message") or resp.text
                        except Exception:
                            err_text = resp.text
                        print(f"[NVIDIA] HTTP {resp.status_code} ({cur_model}): {err_text[:200]}")
                        if resp.status_code in (500, 502, 503, 504) and cur_model != models_to_try[-1]:
                            last_error = f"HTTP {resp.status_code}: {err_text[:200]}"
                            break

                    resp.raise_for_status()
                    choice = resp.json().get("choices", [{}])[0]
                    return _clean_reasoning(choice.get("message", {}).get("content") or "")
                except requests.exceptions.HTTPError as e:
                    if attempt < 2 and e.response is not None and e.response.status_code in (500, 502, 503, 504, 429):
                        time.sleep(1.0 * (attempt + 1))
                        continue
                    last_error = e
                except Exception as e:
                    if attempt < 2:
                        time.sleep(1.0)
                        continue
                    last_error = e

        raise RuntimeError(f"{provider.upper()} text call failed: {last_error}")

    # Ollama
    endpoint = f"{url}/api/chat"
    payload = {"model": m, "messages": messages, "stream": False, "keep_alive": -1, "options": {"num_predict": 1024}}

    try:
        resp = requests.post(endpoint, json=payload, timeout=timeout)
        resp.raise_for_status()
        return (resp.json().get("message", {}).get("content") or "").strip()
    except requests.exceptions.ConnectionError:
        if ensure_ollama_running():
            try:
                resp = requests.post(endpoint, json=payload, timeout=timeout)
                resp.raise_for_status()
                return (resp.json().get("message", {}).get("content") or "").strip()
            except Exception:
                pass
        raise RuntimeError(
            f"Cannot connect to Ollama at {url}. "
            "Make sure Ollama is installed and run: ollama serve"
        )
    except Exception as e:
        raise RuntimeError(f"LLM text call failed: {e}")

def _stream_openai(
    messages: list,
    tools:    list | None,
    timeout:  int,
) -> Generator[dict, None, None]:
    """
    Streaming backend for NVIDIA NIM and OpenAI-compatible servers (LM Studio, LocalAI, Jan…).

    Parses Server-Sent Events (SSE) and accumulates streaming tool-call fragments
    so the output format is identical to the Ollama backend.
    """
    url, model = get_llm_settings()
    endpoint   = f"{url}/v1/chat/completions" if not url.endswith("/v1") else f"{url}/chat/completions"
    headers    = get_auth_headers()

    # Sanitize messages: ensure content is always a valid string, never None
    cleaned_messages = []
    for msg in messages:
        mc = dict(msg)
        if mc.get("content") is None:
            mc["content"] = ""
        cleaned_messages.append(mc)

    cfg        = _load_config()
    max_tok    = int(cfg.get("max_tokens", 2048))
    payload: dict = {
        "model":      model,
        "messages":   cleaned_messages,
        "stream":     True,
        "max_tokens": max_tok,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    enable_think = cfg.get("enable_thinking", False)
    if "nemotron" in model.lower():
        payload["chat_template_kwargs"] = {"enable_thinking": enable_think, "force_nonempty_content": True}
    else:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    session = _get_session()
    try:
        resp = None
        for attempt in range(3):
            try:
                req_payload = dict(payload)
                if attempt > 0:
                    req_payload.pop("chat_template_kwargs", None)
                resp = session.post(endpoint, json=req_payload, headers=headers, timeout=timeout, stream=True)
                if resp.status_code in (500, 502, 503, 504, 429) and attempt < 2:
                    time.sleep(1.0 * (attempt + 1))
                    continue
                resp.raise_for_status()
                break
            except requests.exceptions.HTTPError as e:
                if attempt < 2 and e.response is not None and e.response.status_code in (500, 502, 503, 504, 429):
                    time.sleep(1.0 * (attempt + 1))
                    continue
                raise

        with resp:
            full_content = ""
            buf          = ""
            inside_think = False
            # tool_call fragments: index → {"id", "function": {"name", "arguments"}}
            tc_fragments: dict[int, dict] = {}

            for raw in resp.iter_lines():
                if not raw:
                    continue
                # SSE lines look like: b"data: {...}" or b"data: [DONE]"
                line = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                choice = chunk.get("choices", [{}])[0]
                delta  = choice.get("delta", {})
                text   = delta.get("content") or ""

                full_content += text

                if "<think>" in text:
                    inside_think = True
                if "</think>" in text:
                    inside_think = False
                    post = text.split("</think>")[-1]
                    buf += post
                elif not inside_think:
                    buf += text

                # Accumulate sentence boundaries for streaming TTS
                while True:
                    m = _SENT_END.search(buf)
                    if not m:
                        break
                    sentence = buf[: m.start() + 1].strip()
                    buf      = buf[m.end():]
                    sentence = _clean_reasoning(sentence)
                    if sentence:
                        yield {"type": "sentence", "text": sentence}

                # Accumulate streaming tool-call fragments
                for tc in (delta.get("tool_calls") or []):
                    idx = tc.get("index", 0)
                    if idx not in tc_fragments:
                        tc_fragments[idx] = {"id": "", "function": {"name": "", "arguments": ""}}
                    frag = tc_fragments[idx]
                    frag["id"] = frag["id"] or tc.get("id", "")
                    fn = tc.get("function", {})
                    frag["function"]["name"]      += fn.get("name") or ""
                    frag["function"]["arguments"] += fn.get("arguments") or ""

                finish = choice.get("finish_reason")
                if finish in ("stop", "tool_calls", "length"):
                    break

            # Flush any trailing content
            trailing = _clean_reasoning(buf.strip())
            if trailing:
                yield {"type": "sentence", "text": trailing}

            # Parse accumulated tool-call argument strings → dicts
            tool_calls: list = []
            for idx in sorted(tc_fragments):
                frag = tc_fragments[idx]
                args = frag["function"]["arguments"]
                try:
                    args = json.loads(args)
                except Exception:
                    pass   # leave as raw string; _execute_tool handles it
                tool_calls.append({
                    "id":       frag["id"],
                    "function": {"name": frag["function"]["name"], "arguments": args},
                })

            yield {
                "type":       "done",
                "content":    _clean_reasoning(full_content),
                "tool_calls": tool_calls,
            }

    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Cannot reach OpenAI-compatible server at {url}.\n"
            "Make sure LM Studio / LocalAI / Jan is running and the server is started."
        )
    except requests.exceptions.Timeout:
        raise RuntimeError("OpenAI-compatible stream timed out.")
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"OpenAI-compatible HTTP error: {e.response.status_code}")
    except Exception as e:
        raise RuntimeError(f"OpenAI-compatible stream failed: {e}")


def call_llm_stream(
    messages: list,
    tools:    list | None = None,
    timeout:  int = 120,
) -> Generator[dict, None, None]:
    """
    Streaming chat request.  Routes to Ollama or OpenAI-compatible backend.

    Yields:
        {"type": "sentence", "text": str}   — each complete sentence as it arrives
        {"type": "done", "content": str, "tool_calls": list}  — when stream ends

    Sentences are split on [.!?] + whitespace so TTS can start immediately.
    Tool calls always appear in the final "done" event.
    """
    provider = get_llm_provider()
    if provider in ("openai", "nvidia"):
        yield from _stream_openai(messages, tools, timeout)
        return

    url, model = get_llm_settings()
    endpoint   = f"{url}/api/chat"

    payload: dict = {
        "model":      model,
        "messages":   messages,
        "stream":     True,
        "keep_alive": -1,
        # 150 tokens ≈ 100 words ≈ 3-4 sentences — enough for any voice reply.
        # num_gpu:99 pushes all layers to GPU; num_thread removed (Ollama auto-tunes).
        "options":    {"num_predict": 150, "num_gpu": 99},
    }
    if tools:
        payload["tools"] = tools

    def _do_stream() -> Generator[dict, None, None]:
        with requests.post(endpoint, json=payload, timeout=timeout, stream=True) as resp:
            resp.raise_for_status()
            full_content = ""
            tool_calls:  list = []
            buf          = ""

            for raw in resp.iter_lines():
                if not raw:
                    continue
                try:
                    chunk = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                msg   = chunk.get("message", {})
                delta = msg.get("content") or ""

                full_content += delta
                buf          += delta

                # Yield complete sentences as they accumulate
                while True:
                    m = _SENT_END.search(buf)
                    if not m:
                        break
                    sentence = buf[: m.start() + 1].strip()
                    buf      = buf[m.end() :]
                    if sentence:
                        yield {"type": "sentence", "text": sentence}

                tc = msg.get("tool_calls")
                if tc:
                    tool_calls.extend(tc)

                if chunk.get("done"):
                    if buf.strip():
                        yield {"type": "sentence", "text": buf.strip()}

                    yield {
                        "type":       "done",
                        "content":    full_content.strip(),
                        "tool_calls": tool_calls,
                    }
                    return

    try:
        yield from _do_stream()
    except requests.exceptions.ConnectionError as e:
        print(f"[LLM] Stream ConnectionError — trying to restart Ollama… ({e})")
        if ensure_ollama_running():
            yield from _do_stream()
            return
        raise RuntimeError(
            f"Cannot connect to Ollama at {url}. "
            "Make sure Ollama is installed and run: ollama serve"
        )
    except requests.exceptions.Timeout:
        raise RuntimeError("Ollama stream timed out.")
    except requests.exceptions.HTTPError as e:
        err_detail = ""
        try:
            err_detail = e.response.json().get("error", "")
        except Exception:
            err_detail = e.response.text if e.response is not None else ""
        if e.response is not None and e.response.status_code == 404:
            print(f"[LLM] HTTP 404 Not Found from {endpoint}: {err_detail or 'Model or endpoint not found'}")
            raise RuntimeError(f"Ollama HTTP 404 Not Found: {err_detail or f'Model {model} not found'}")
        print(f"[LLM] HTTPError: {e.response.status_code if e.response else 'unknown'} — {err_detail[:200]}")
        raise RuntimeError(f"Ollama HTTP error {e.response.status_code if e.response else 'unknown'}: {err_detail or e}")
    except Exception as e:
        print(f"[LLM] Stream error: {type(e).__name__}: {e}")
        raise RuntimeError(f"LLM stream failed: {e}")


def call_llm_vision(
    image_bytes: bytes,
    prompt: str,
    mime_type: str = "image/jpeg",
    model: str | None = None,
    timeout: int = 90,
) -> str:
    """
    Vision analysis using NVIDIA NIM (e.g. meta/llama-3.2-11b-vision-instruct)
    or OpenAI-compatible vision endpoints.
    """
    import base64
    url, default_model = get_llm_settings()
    provider           = get_llm_provider()

    b64_img  = base64.b64encode(image_bytes).decode("ascii")
    data_url = f"data:{mime_type};base64,{b64_img}"

    vision_model = model
    if not vision_model or any(k in str(vision_model).lower() for k in ("gemini", "flash", "pro", "gpt", "claude")):
        if provider == "nvidia":
            vision_model = "meta/llama-3.2-11b-vision-instruct"
        else:
            vision_model = default_model

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": data_url},
                },
            ],
        }
    ]

    endpoint = f"{url}/v1/chat/completions" if not url.endswith("/v1") else f"{url}/chat/completions"
    headers  = get_auth_headers()
    payload  = {
        "model": vision_model,
        "messages": messages,
        "stream": False,
        "max_tokens": 512,
    }

    try:
        resp = requests.post(endpoint, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
        choice = resp.json().get("choices", [{}])[0]
        return (choice.get("message", {}).get("content") or "").strip()
    except Exception as e:
        print(f"[Vision] Vision API call error: {e}")
        return f"I encountered an error analyzing the visual content: {e}"


class _GenaiResponseShim:
    def __init__(self, text: str):
        self.text = text
        class _Part:
            def __init__(self, t):
                self.text = t
        class _Content:
            def __init__(self, t):
                self.parts = [_Part(t)]
        class _Candidate:
            def __init__(self, t):
                self.content = _Content(t)
        self.candidates = [_Candidate(text)]


class UnifiedModel:
    """
    Universal model adapter providing a .generate_content() method matching
    the Google genai.Client interface, compatible with NVIDIA NIM, OpenAI, and Ollama.
    """
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name

    def generate_content(self, contents, **kwargs):
        if isinstance(contents, str):
            res = call_llm_text(contents, model=self.model_name)
            return _GenaiResponseShim(res)
        if isinstance(contents, list):
            img_item = None
            text_parts = []
            for item in contents:
                if hasattr(item, "save") or isinstance(item, bytes):
                    img_item = item
                elif isinstance(item, str):
                    text_parts.append(item)
                elif hasattr(item, "text"):
                    text_parts.append(str(item.text))
                else:
                    text_parts.append(str(item))
            prompt_str = " ".join(text_parts).strip() or "Analyze this."
            if img_item is not None:
                import io
                if hasattr(img_item, "save"):
                    buf = io.BytesIO()
                    img_item.save(buf, format="JPEG")
                    img_b = buf.getvalue()
                else:
                    img_b = img_item
                res = call_llm_vision(img_b, prompt_str)
                return _GenaiResponseShim(res)
            res = call_llm_text(prompt_str, model=self.model_name)
            return _GenaiResponseShim(res)
        res = call_llm_text(str(contents), model=self.model_name)
        return _GenaiResponseShim(res)


class UnifiedGenaiClient:
    """Drop-in replacement for google.genai.Client when running in NVIDIA / local mode."""
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key
        self.models = self
        self.aio = self  # Support async client.aio.models.generate_content calls

    def generate_content(self, model: str = "", contents=None, **kwargs):
        m = UnifiedModel(model_name=model)
        return m.generate_content(contents, **kwargs)


def get_ai_client():
    """
    Returns a client for text and vision generation.
    Always returns UnifiedGenaiClient powered by NVIDIA NIM (or local LLM).
    Zero dependency on external Google Gemini API.
    """
    return UnifiedGenaiClient()

