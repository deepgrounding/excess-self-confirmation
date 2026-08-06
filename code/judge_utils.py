"""Shared judge helpers with no MLX dependency (safe for CPU-only `uv run`).

Used by both run_stage2_esc.py (which additionally needs mlx-lm for local
generation) and calibrate_strong_noise.py (API + numpy/scipy only).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

import numpy as np

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def api_chat_openrouter(model: str, content: str, max_tokens: int, timeout: float = 120.0) -> str:
    """Single-turn chat completion via OpenRouter. Used for J_strong / J_strong~."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("OPENROUTER_API_KEY not set in environment.")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.2,
        "max_tokens": max_tokens,
        # Reasoning-tuned models (e.g. deepseek/deepseek-v4-flash-0731) emit a
        # separate thinking trace that otherwise shares the max_tokens budget
        # with the final JSON, truncating it mid-output on harder items and
        # silently falling back to extract_judge's 0.5 default (found via a
        # 51.8%-of-336 pileup at exactly 0.5 in the first math calibration
        # attempt against this model, 2026-08-03). Ignored harmlessly by
        # OpenRouter for models without a reasoning mode (e.g. Haiku).
        "reasoning": {"enabled": False},
    }
    req = urllib.request.Request(
        OPENROUTER_URL,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
            "HTTP-Referer": "https://github.com/deepgrounding",
            "X-Title": "esc-manuscript-strong-judge",
        },
    )
    last_err = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                d = json.loads(resp.read())
                return d["choices"][0]["message"]["content"] or ""
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:300]
            last_err = f"HTTP {e.code}: {body}"
            if e.code == 429:
                time.sleep(15 * (attempt + 1))
                continue
            if e.code >= 500:
                time.sleep(5 * (attempt + 1))
                continue
            break
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
            time.sleep(5 * (attempt + 1))
    print(f"  [strong-judge API failed: {last_err}]", flush=True)
    return ""


OLLAMA_URL = "http://localhost:11434/api/chat"


def api_chat_ollama(model: str, content: str, max_tokens: int, timeout: float = 180.0) -> str:
    """Single-turn chat completion via a local Ollama server. This is now the
    default J_strong backend (main.md 4.2 amendment, pre-Stage-1-submission):
    gemma4:12b, local and free. Claude Haiku 4.5 via OpenRouter is retained as
    an independent cross-check -- use --strong-backend openrouter to invoke it.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "stream": False,
        "think": False,
        "options": {"temperature": 0.2, "num_predict": max_tokens},
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    last_err = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                d = json.loads(resp.read())
                return d.get("message", {}).get("content", "") or ""
        except Exception as e:  # noqa: BLE001
            last_err = str(e)
            time.sleep(3 * (attempt + 1))
    print(f"  [ollama strong-judge failed: {last_err}]", flush=True)
    return ""


def api_chat_strong(
    backend: str, model: str, content: str, max_tokens: int
) -> str:
    """Dispatch to the configured strong-judge backend."""
    if backend == "ollama":
        return api_chat_ollama(model, content, max_tokens)
    return api_chat_openrouter(model, content, max_tokens)


def strong_noise_seed(task_id: str, rnd: int, cand_idx: int) -> int:
    """Frozen noise seed per Appendix B step 4: seed = hash(item, round)."""
    h = hashlib.sha1(f"{task_id}|{rnd}|{cand_idx}".encode()).hexdigest()
    return int(h, 16) % (2**32)


def apply_noise(scores, tau: float, lam: float, rng: np.random.Generator, space: str):
    """Appendix B step 2 noise injection, shared by calibrate_strong_noise.py
    (offline calibration) and run_stage2_esc.py (online J_strong_tilde arm) so
    the two never drift apart. 'linear' is the frozen-spec form
    (score + N(0,tau^2)), clipped to [0,1] -- additive noise near a boundary
    (many exact 0/1 scores) gets clipped away, silently shrinking effective
    spread. 'logit' perturbs in logit space instead so noise doesn't saturate
    near 0/1; this is an amendment to try when 'linear' can't hit the SEL(0)
    gate, not the pre-registered default."""
    scores = np.asarray(scores, dtype=float)
    if space == "logit":
        eps = 1e-4
        p = np.clip(lam * scores, eps, 1.0 - eps)
        logit = np.log(p / (1.0 - p))
        noisy_logit = logit + rng.normal(0.0, tau, size=scores.shape)
        return 1.0 / (1.0 + np.exp(-noisy_logit))
    return np.clip(lam * scores + rng.normal(0.0, tau, size=scores.shape), 0.0, 1.0)


def extract_judge(text: str) -> tuple[float, str]:
    score, feedback = 0.5, "Please improve overall correctness."
    m = re.search(r"\{[^{}]*\}", text, flags=re.DOTALL)
    blob = m.group(0) if m else text
    sm = re.search(r'"score"\s*:\s*([0-9]*\.?[0-9]+)', blob)
    if sm:
        score = float(np.clip(float(sm.group(1)), 0.0, 1.0))
    fm = re.search(r'"feedback"\s*:\s*"([^"]*)"', blob)
    if fm:
        feedback = fm.group(1).strip() or feedback
    return score, feedback
