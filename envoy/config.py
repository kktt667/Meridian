"""Central settings. Gate thresholds live here so the demo and the tests use the same numbers."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
try:  # secrets (API keys, wallet keys) live in .env, which is git-ignored
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass
LAB_DATA_DIR = Path(os.environ.get("ENVOY_LAB_DATA", ROOT / "data" / "labs"))
RUNS_DIR = Path(os.environ.get("ENVOY_RUNS", ROOT / "runs"))

# ---- Gate policy (applied inside every lab, never by the coordinator) ----
MIN_CELL = 5            # every released count must be >= this, otherwise that stratum is suppressed
MIN_CASES_MODEL = 20    # a model-based estimate needs >= this many outcome cases among the exposed
MIN_POPULATION = 50     # the filtered population must have at least this many people
QUERY_BUDGET = 40       # queries per asker per lab per session
OVERLAP_WINDOW = 4      # two queries whose age windows differ by 1..4 years in total look like a differencing attack

# ---- AI layer ----
MODEL = os.environ.get("ENVOY_MODEL", "claude-opus-5-5")
EFFORT_PLANNER = os.environ.get("ENVOY_EFFORT_PLANNER", "high")
EFFORT_DEBATE = os.environ.get("ENVOY_EFFORT_DEBATE", "medium")
EFFORT_VOTER = os.environ.get("ENVOY_EFFORT_VOTER", "medium")
N_VOTERS = 3
BACKEND = os.environ.get("ENVOY_BACKEND", "ollama")          # ollama | claude | offline
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_CTX = int(os.environ.get("ENVOY_OLLAMA_CTX", "8192"))
# Different model families per role -> a genuinely diverse panel. Override any with ENVOY_MODEL_<ROLE>.
ROLE_MODELS = {
    "default": "qwen2.5:7b",
    "planner": "qwen2.5:7b",
    "for": "llama3.1:8b",
    "against": "mistral:7b",
    "edge_cases": "qwen2.5:7b",
    "voter_methodologist": "qwen2.5:7b",
    "voter_clinician": "llama3.1:8b",
    "voter_ethics_regulatory": "mistral:7b",
    "fallbacks": ["qwen2.5:7b", "llama3.1:8b", "mistral:7b", "qwen2.5:3b"],
}
for _k in list(ROLE_MODELS):
    if _k != "fallbacks" and os.environ.get(f"ENVOY_MODEL_{_k.upper()}"):
        ROLE_MODELS[_k] = os.environ[f"ENVOY_MODEL_{_k.upper()}"]
USE_SERVER_FALLBACK = os.environ.get("ENVOY_SERVER_FALLBACK", "1") == "1"
FORCE_OFFLINE = os.environ.get("ENVOY_OFFLINE", "0") == "1"

# ---- Payments (simulated x402 until the real integration lands) ----
PRICES_USD = {"verdict": 0.10, "evidence": 0.50, "audit": 2.00}
DEFAULT_BUDGET_USD = 10.0

# ---- x402 (real payments on Base Sepolia testnet) ----
TRANSPORT = os.environ.get("ENVOY_TRANSPORT", "inprocess")       # inprocess | x402
X402_NETWORK = os.environ.get("X402_NETWORK", "eip155:84532")      # Base Sepolia
X402_FACILITATOR = os.environ.get("X402_FACILITATOR", "https://x402.org/facilitator")
X402_MAX_PER_PAYMENT = os.environ.get("X402_MAX_PER_PAYMENT", "$2")
LAB_PORTS = {"US_BRFSS_2023": 4101, "BRAZIL_PNS_2019": 4102, "KOREA_KNHANES_2024": 4103, "US_NHANES_2017": 4104}
LAB_URLS = {k: os.environ.get(f"LAB_{k}_URL", f"http://127.0.0.1:{v}") for k, v in LAB_PORTS.items()}

RUN_MODE = os.environ.get("ENVOY_MODE", "fast")   # fast = vector panel + 1 narrator call | deep = full LLM debate
