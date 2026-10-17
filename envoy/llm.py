"""LLM backends for structured agent outputs.

ENVOY_BACKEND = ollama (default; local open-weight models, nothing leaves the laptop)
              | claude (Anthropic API, needs ANTHROPIC_API_KEY)
              | offline (deterministic agents, no LLM)

Each agent role can run on a different model family (config.ROLE_MODELS), so the voting panel is made of
genuinely differently trained models.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import httpx

from .config import BACKEND, FORCE_OFFLINE, MODEL, OLLAMA_URL, ROLE_MODELS, USE_SERVER_FALLBACK, OLLAMA_CTX

PRICE_PER_MTOK = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50)}


class AgentFailed(Exception):
    pass


def ollama_up() -> bool:
    try:
        return httpx.get(f"{OLLAMA_URL}/api/version", timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


def ollama_models() -> set[str]:
    try:
        return {m["name"] for m in httpx.get(f"{OLLAMA_URL}/api/tags", timeout=3).json().get("models", [])}
    except httpx.HTTPError:
        return set()


def has_claude_credentials() -> bool:
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_PROFILE"):
        return True
    return (Path.home() / ".config" / "anthropic").exists()


def _flatten(content) -> str:
    """Claude-style content blocks -> one prompt string (documents inlined with clear delimiters)."""
    if isinstance(content, str):
        return content
    parts = []
    for b in content:
        if b.get("type") == "document":
            parts.append(f"<document title=\"{b.get('title', '')}\">\n{b['source'].get('data', '')}\n</document>")
        elif b.get("type") == "text":
            parts.append(b["text"])
    return "\n\n".join(parts)


class LLM:
    def __init__(self, offline: bool | None = None, backend: str | None = None):
        b = backend or BACKEND
        if offline or FORCE_OFFLINE:
            b = "offline"
        elif b == "ollama" and not ollama_up():
            b = "claude" if has_claude_credentials() else "offline"
        elif b == "claude" and not has_claude_credentials():
            b = "ollama" if ollama_up() else "offline"
        self.backend = b
        self.offline = b == "offline"
        self.available = ollama_models() if b == "ollama" else set()
        self.client = None
        if b == "claude":
            import anthropic
            self.client = anthropic.Anthropic()
        self.calls: list[dict] = []
        self.notes: list[str] = []
        self.on_call = None
        self._loaded: str | None = None

    def model_for(self, role: str) -> str:
        if self.backend == "claude":
            return MODEL
        want = ROLE_MODELS.get(role)
        if want is None:
            from .catalog import LABS  # each lab's agent and steward run on that lab's own model family
            for prefix in ("lab_agent_", "steward_"):
                if role.startswith(prefix) and role[len(prefix):] in LABS:
                    want = LABS[role[len(prefix):]]["policy"]["agent_model"]
            want = want or ROLE_MODELS["default"]
        if want in self.available:
            return want
        fallback = next((m for m in ROLE_MODELS["fallbacks"] if m in self.available), None)
        if fallback is None:
            raise AgentFailed(f"no local model available for {role} (wanted {want})")
        self.notes.append(f"{role}: {want} not pulled, used {fallback}")
        return fallback

    def structured(self, role: str, system: str, content, schema_cls, effort: str = "medium"):
        if self.offline:
            raise AgentFailed("offline")
        if self.backend == "ollama":
            return self._ollama(role, system, content, schema_cls)
        return self._claude(role, system, content, schema_cls, effort)

    # ------------------------------------------------------------ local open-weight models
    def _ollama(self, role, system, content, schema_cls):
        model = self.model_for(role)
        if self._loaded and self._loaded != model:
            # 16 GB laptop: unload the previous family before loading the next, so only one model is ever resident
            try:
                httpx.post(f"{OLLAMA_URL}/api/generate", json={"model": self._loaded, "keep_alive": 0}, timeout=30)
            except httpx.HTTPError:
                pass
        self._loaded = model
        schema = schema_cls.model_json_schema()
        prompt = (_flatten(content) + "\n\nRespond ONLY with JSON matching this schema:\n" + json.dumps(schema))
        t0 = time.time()
        try:
            r = httpx.post(f"{OLLAMA_URL}/api/chat", timeout=600, json={
                "model": model, "stream": False, "format": schema, "keep_alive": "10m",
                "options": {"temperature": 0, "num_ctx": OLLAMA_CTX},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]})
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise AgentFailed(f"ollama error ({model}): {e}") from e
        data = r.json()
        try:
            out = schema_cls.model_validate_json(data["message"]["content"])
        except Exception as e:
            raise AgentFailed(f"{model} returned invalid JSON for {role}: {str(e)[:160]}") from e
        dur = data.get("total_duration", 0) / 1e9 or (time.time() - t0)
        toks = data.get("eval_count", 0)
        self.calls.append({"role": role, "model": model, "input_tokens": data.get("prompt_eval_count", 0),
                           "output_tokens": toks, "seconds": round(dur, 2),
                           "tokens_per_s": round(toks / max(data.get("eval_duration", 1) / 1e9, 1e-6), 1), "cost_usd": 0.0})
        if self.on_call:
            self.on_call(self.calls[-1])
        return out

    # ------------------------------------------------------------ Anthropic API (optional)
    def _claude(self, role, system, content, schema_cls, effort):
        import anthropic
        kwargs = dict(model=MODEL, max_tokens=16000, system=system, messages=[{"role": "user", "content": content}],
                      output_format=schema_cls, output_config={"effort": effort})
        t0 = time.time()
        try:
            if USE_SERVER_FALLBACK:
                try:
                    r = self.client.beta.messages.parse(betas=["server-side-fallback-2026-07-01"],
                                                        extra_body={"fallbacks": "default"}, **kwargs)
                except anthropic.BadRequestError as e:
                    self.notes.append(f"{role}: server-side fallback not accepted ({str(e)[:100]}); retried without it")
                    r = self.client.messages.parse(**kwargs)
            else:
                r = self.client.messages.parse(**kwargs)
        except anthropic.AuthenticationError as e:
            self.offline = True
            raise AgentFailed(f"authentication failed: {e}") from e
        except anthropic.RateLimitError as e:
            raise AgentFailed(f"rate limited: {e}") from e
        except anthropic.APIStatusError as e:
            raise AgentFailed(f"API error {e.status_code}: {str(e)[:200]}") from e
        except anthropic.APIConnectionError as e:
            raise AgentFailed(f"network error: {e}") from e
        if r.stop_reason == "refusal":
            raise AgentFailed("model declined (refusal)")
        if r.stop_reason == "max_tokens":
            raise AgentFailed("output truncated (max_tokens)")
        if r.parsed_output is None:
            raise AgentFailed("no structured output")
        pin, pout = PRICE_PER_MTOK.get(MODEL, (4.0, 20.0))
        u = r.usage
        self.calls.append({"role": role, "model": r.model, "input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
                           "seconds": round(time.time() - t0, 2), "tokens_per_s": None,
                           "cost_usd": round(u.input_tokens / 1e6 * pin + u.output_tokens / 1e6 * pout, 4)})
        if self.on_call:
            self.on_call(self.calls[-1])
        return r.parsed_output

    def cost_summary(self) -> dict:
        mode = {"ollama": "local open-weight models (Ollama)", "claude": f"Claude API ({MODEL})",
                "offline": "offline (deterministic agents, no LLM)"}[self.backend]
        return {"mode": mode, "backend": self.backend, "calls": len(self.calls),
                "models": sorted({c["model"] for c in self.calls}),
                "input_tokens": sum(c["input_tokens"] for c in self.calls),
                "output_tokens": sum(c["output_tokens"] for c in self.calls),
                "llm_seconds": round(sum(c["seconds"] for c in self.calls), 1),
                "cost_usd": round(sum(c["cost_usd"] for c in self.calls), 4), "per_call": self.calls, "notes": self.notes}
