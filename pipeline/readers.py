"""Three readers from three different families. Each returns (vector, raw_json_dict).

  claude : Claude Haiku 5.5 via the Anthropic SDK, structured JSON output
  ollama : a local open model (Qwen 2.5 3B) via Ollama, structured JSON output
  regex  : non-generative heuristic reader (dates, keywords) - the weak but free vote
"""
import json, re
import numpy as np
import requests
from features import FEATURES, IDS, SYSTEM_PROMPT, json_schema, to_vector, D

SCHEMA = json_schema()

# ----------------------------------------------------------------------------- redaction
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"(\+?\d[\d\s().-]{7,}\d)")
_URL   = re.compile(r"https?://\S+|www\.\S+")

def redact(text: str, max_chars: int = 3500) -> str:
    t = _EMAIL.sub("[email]", text)
    t = _URL.sub("[url]", t)
    t = _PHONE.sub("[phone]", t)
    t = re.sub(r"[ \t]+", " ", t)
    return t[:max_chars]

def user_prompt(cv: str) -> str:
    fields = "\n".join(f"- {f['id']}: {f['desc']}" for f in FEATURES)
    return f"Fields to transcribe:\n{fields}\n\nCV:\n<<<\n{cv}\n>>>"

# ----------------------------------------------------------------------------- claude
async def read_claude(client, cv: str, model: str = "claude-haiku-5-5"):
    resp = await client.messages.create(
        model=model,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt(cv)}],
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
    )
    text = next(b.text for b in resp.content if b.type == "text")
    obj = json.loads(text)
    usage = (resp.usage.input_tokens, resp.usage.output_tokens)
    return to_vector(obj), obj, usage

# ----------------------------------------------------------------------------- ollama
def read_ollama(cv: str, model: str = "qwen2.5:3b", host: str = "http://localhost:11434"):
    r = requests.post(f"{host}/api/chat", json={
        "model": model, "stream": False, "format": SCHEMA,
        "options": {"temperature": 0, "num_ctx": 8192},
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": user_prompt(cv)}],
    }, timeout=300)
    r.raise_for_status()
    content = r.json()["message"]["content"]
    try:
        obj = json.loads(content)
    except json.JSONDecodeError:
        obj = {}
    return to_vector(obj), obj, (0, 0)

# ----------------------------------------------------------------------------- regex
_LANGS = ["python", "java", "sql", "c#", "c++", "javascript", "typescript", "bash", "powershell",
          "php", "ruby", "perl", "scala", "matlab", "html", "css", "swift", "kotlin"]
_INFRA = ["active directory", "server", "network", "vmware", "linux", "windows server", "cisco", "lan",
          "wan", "firewall", "vpn", "aws", "azure", "cloud", "dns", "dhcp", "tcp/ip", "sysadmin", "helpdesk", "help desk"]
_CERTS = ["comptia", "a+", "network+", "security+", "ccna", "ccnp", "mcse", "mcsa", "mcp", "itil", "aws certified",
          "azure", "pmp", "cissp", "cisa", "ceh"]
_CSDEG = re.compile(r"(computer science|information technology|information systems|software engineering|computer engineering|computer information)", re.I)
_DEG   = re.compile(r"(bachelor|master|b\.?s\.?|m\.?s\.?|b\.?a\.?|associate|degree)", re.I)
_IT_TITLE = re.compile(r"(it |information technology|software|developer|engineer|programmer|network|system|sysadmin|administrator|help ?desk|technician|support analyst|database|data analyst|qa |devops|web)", re.I)
_MGMT_TITLE = re.compile(r"(manager|director|supervisor|team lead|lead |head of)", re.I)
_RANGE = re.compile(r"(\d{1,2})/(\d{4})\s+to\s+(?:(\d{1,2})/(\d{4})|current|present)\s+(.{0,90})", re.I)

def _years(m, now=(10, 2026)):
    m1, y1 = int(m.group(1)), int(m.group(2))
    if m.group(3):
        m2, y2 = int(m.group(3)), int(m.group(4))
    else:
        m2, y2 = now
    return max(0.0, ((y2 - y1) * 12 + (m2 - m1)) / 12.0)

def read_regex(cv: str):
    low = cv.lower()
    it = mgmt = total = 0.0
    quotes = {}
    for m in _RANGE.finditer(cv):
        yrs = _years(m); title = m.group(5)[:40]   # job title sits right after the dates
        total += yrs
        if _IT_TITLE.search(title):
            it += yrs; quotes.setdefault("it_years", m.group(0)[:80])
        if _MGMT_TITLE.search(title):
            mgmt += yrs; quotes.setdefault("mgmt_years", m.group(0)[:80])
    langs = [l for l in _LANGS if l in low]
    infra = [k for k in _INFRA if k in low]
    certs = [c for c in _CERTS if c in low]
    deg = 1.0 if (_CSDEG.search(cv) and _DEG.search(cv)) else 0.0
    vals = {
        "it_years": it if total > 0 else -1, "prog_langs": float(len(langs)),
        "infra_skills": 1.0 if infra else 0.0, "cs_degree": deg, "it_certs": float(len(certs)),
        "mgmt_years": mgmt if total > 0 else -1, "total_years": total if total > 0 else -1,
        "non_it_years": max(total - it, 0.0) if total > 0 else -1,
    }
    obj = {k: {"value": v, "quote": quotes.get(k, ", ".join((langs if k == "prog_langs" else infra if k == "infra_skills" else certs)[:5]))}
           for k, v in vals.items()}
    return to_vector(obj), obj, (0, 0)

def judge_ollama(cv: str, model: str = "qwen2.5:3b", host: str = "http://localhost:11434", temperature: float = 0.8):
    """Baseline: the LLM decides directly. Sampled (not temperature 0) so reruns can disagree, as real deployments do."""
    schema = {"type": "object", "properties": {"shortlist": {"type": "boolean"}}, "required": ["shortlist"], "additionalProperties": False}
    r = requests.post(f"{host}/api/chat", json={
        "model": model, "stream": False, "format": schema, "options": {"temperature": temperature, "num_ctx": 8192},
        "messages": [{"role": "system", "content": "You are screening CVs for an Information Technology role. Decide whether to shortlist."},
                     {"role": "user", "content": f"CV:\n<<<\n{cv}\n>>>\nShortlist for the IT role?"}]}, timeout=300)
    r.raise_for_status()
    try:
        return int(bool(json.loads(r.json()["message"]["content"])["shortlist"]))
    except Exception:
        return -1

READERS = {"claude": "claude-haiku-5-5", "ollama": "qwen2.5:3b", "ollama2": "orca-mini:latest", "regex": "heuristic-v1"}
