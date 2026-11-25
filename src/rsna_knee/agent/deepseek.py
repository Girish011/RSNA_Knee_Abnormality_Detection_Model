"""DeepSeek chat client (OpenAI-compatible) with JSON, tool calls, and spend accounting.

Prices are conservative estimates used only for the budget cap; the real bill is on the
DeepSeek console. Every call's token usage is returned so callers can log cost.

Tool calling: pass OpenAI-style ``tools`` to ``chat`` / ``run_tools``. Multi-turn tool
loops must echo ``reasoning_content`` on assistant messages (DeepSeek V4 requirement).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

import requests

from rsna_knee.agent.runpod import secret

URL = "https://api.deepseek.com/chat/completions"
FLASH = "deepseek-flash"  # DeepSeek-V4.1-Flash per /models (2026-09-29)
PRO = "deepseek-v4-pro"
# USD per million tokens (input, output incl. reasoning); deliberately on the high side.
PRICE = {FLASH: (0.30, 1.20), PRO: (1.00, 4.00)}

ToolHandler = Callable[..., Any]


def usage_cost(model: str, usage: dict) -> float:
    pin, pout = PRICE.get(model, PRICE[PRO])
    return (usage.get("prompt_tokens", 0) * pin + usage.get("completion_tokens", 0) * pout) / 1e6


def _headers() -> dict:
    return {"Authorization": f"Bearer {secret('DEEPSEEK_API_KEY')}"}


def chat(messages: list[dict], *, model: str = FLASH, tools: list[dict] | None = None,
         tool_choice: str | dict | None = None, retries: int = 3,
         timeout: int = 120) -> tuple[dict, float]:
    """One chat completion. Returns (assistant message dict, estimated USD)."""
    body: dict[str, Any] = {"model": model, "messages": messages}
    if tools:
        body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
    else:
        body["response_format"] = {"type": "json_object"}
    err = None
    for i in range(retries):
        try:
            r = requests.post(URL, json=body, timeout=timeout, headers=_headers())
            r.raise_for_status()
            d = r.json()
            return d["choices"][0]["message"], usage_cost(model, d.get("usage", {}))
        except (requests.RequestException, KeyError, IndexError) as e:
            err = e
            time.sleep(2 ** i)
    raise RuntimeError(f"deepseek {model} failed after {retries} tries: {type(err).__name__}: {str(err)[:200]}")


def chat_json(system: str, user: str, *, model: str = FLASH, retries: int = 3,
              timeout: int = 120) -> tuple[dict, float]:
    """Return (parsed JSON reply, estimated USD). Raises after ``retries`` failures."""
    msg, cost = chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
                     model=model, retries=retries, timeout=timeout)
    content = msg.get("content") or "{}"
    try:
        return json.loads(content), cost
    except json.JSONDecodeError as e:
        raise RuntimeError(f"deepseek {model} returned non-JSON: {content[:200]}") from e


def _assistant_echo(msg: dict) -> dict:
    """Echo assistant fields DeepSeek requires on subsequent turns (incl. reasoning)."""
    out: dict[str, Any] = {"role": "assistant", "content": msg.get("content") or ""}
    if msg.get("reasoning_content") is not None:
        out["reasoning_content"] = msg["reasoning_content"]
    if msg.get("tool_calls"):
        out["tool_calls"] = msg["tool_calls"]
    return out


def run_tools(system: str, user: str, tools: list[dict], handlers: dict[str, ToolHandler],
              *, model: str = FLASH, max_rounds: int = 6, tool_choice: str | dict = "auto",
              retries: int = 3, timeout: int = 120) -> tuple[dict | None, float, list[dict]]:
    """Run a tool loop until the model stops calling tools or ``max_rounds`` is hit.

    Returns (final parsed JSON content or None, total USD, trace of tool results).
    Handlers receive kwargs from the model's JSON arguments and may return any JSON-serializable
    value (dict preferred). Handler exceptions become ``{"ok": false, "error": "..."}``.
    """
    messages: list[dict] = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    cost, trace = 0.0, []
    final: dict | None = None
    for _ in range(max_rounds):
        msg, c = chat(messages, model=model, tools=tools, tool_choice=tool_choice,
                      retries=retries, timeout=timeout)
        cost += c
        messages.append(_assistant_echo(msg))
        calls = msg.get("tool_calls") or []
        if not calls:
            raw = (msg.get("content") or "").strip()
            if raw:
                try:
                    final = json.loads(raw)
                except json.JSONDecodeError:
                    final = {"type": "stop", "why": f"non-JSON final content: {raw[:200]}"}
            break
        for tc in calls:
            fn = (tc.get("function") or {})
            name = fn.get("name") or ""
            try:
                args = json.loads(fn.get("arguments") or "{}")
                if not isinstance(args, dict):
                    args = {}
            except json.JSONDecodeError:
                args = {}
            handler = handlers.get(name)
            if handler is None:
                result: Any = {"ok": False, "error": f"unknown tool {name!r}"}
            else:
                try:
                    result = handler(**args)
                except TypeError as e:
                    result = {"ok": False, "error": f"bad args for {name}: {e}"}
                except Exception as e:  # noqa: BLE001 — surface to the model
                    result = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            payload = result if isinstance(result, str) else json.dumps(result, default=str)
            trace.append({"tool": name, "args": args, "result": result})
            messages.append({"role": "tool", "tool_call_id": tc.get("id", ""), "content": payload})
    return final, cost, trace
