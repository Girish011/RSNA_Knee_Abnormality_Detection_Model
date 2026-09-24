"""DeepSeek chat client (OpenAI-compatible) with JSON output and spend accounting.

Prices are conservative estimates used only for the budget cap; the real bill is on the
DeepSeek console. Every call's token usage is returned so callers can log cost.
"""

from __future__ import annotations

import json
import time

import requests

from rsna_knee.agent.runpod import secret

URL = "https://api.deepseek.com/chat/completions"
FLASH = "deepseek-flash"
PRO = "deepseek-v4-pro"
# USD per million tokens (input, output incl. reasoning); deliberately on the high side.
PRICE = {FLASH: (0.30, 1.20), PRO: (1.00, 4.00)}


def usage_cost(model: str, usage: dict) -> float:
    pin, pout = PRICE.get(model, PRICE[PRO])
    return (usage.get("prompt_tokens", 0) * pin + usage.get("completion_tokens", 0) * pout) / 1e6


def chat_json(system: str, user: str, *, model: str = FLASH, retries: int = 3,
              timeout: int = 120) -> tuple[dict, float]:
    """Return (parsed JSON reply, estimated USD). Raises after ``retries`` failures."""
    body = {"model": model, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    err = None
    for i in range(retries):
        try:
            r = requests.post(URL, json=body, timeout=timeout,
                              headers={"Authorization": f"Bearer {secret('DEEPSEEK_API_KEY')}"})
            r.raise_for_status()
            d = r.json()
            return json.loads(d["choices"][0]["message"]["content"]), usage_cost(model, d.get("usage", {}))
        except (requests.RequestException, KeyError, json.JSONDecodeError) as e:
            err = e
            time.sleep(2 ** i)
    raise RuntimeError(f"deepseek {model} failed after {retries} tries: {type(err).__name__}: {str(err)[:200]}")
