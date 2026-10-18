"""Coordinator-side handle to a lab running as an x402-protected HTTP server.

Has the same interface as LabNode (handle / preflight / size_bucket), so recipes don't change. The researcher's
wallet pays automatically when the lab answers 402; settlement only happens if the lab returns success.
"""
from __future__ import annotations

import os

import requests

from .config import LAB_URLS, X402_MAX_PER_PAYMENT
from .schemas import LabResult

_SESSION = None
_HTTP = None


def _x402_session():
    global _SESSION, _HTTP
    if _SESSION is None:
        from eth_account import Account
        from x402 import x402ClientSync
        from x402.http import x402HTTPClientSync
        from x402.http.clients import x402_requests
        from x402.mechanisms.evm import EthAccountSigner
        from x402.mechanisms.evm.exact.register import register_exact_evm_client

        key = os.environ.get("RESEARCHER_EVM_PRIVATE_KEY")
        if not key:
            raise RuntimeError("RESEARCHER_EVM_PRIVATE_KEY not set (run scripts/make_wallets.py)")
        client = x402ClientSync().set_spend_controls({"max_amount_per_payment": X402_MAX_PER_PAYMENT})
        register_exact_evm_client(client, EthAccountSigner(Account.from_key(key)))
        _HTTP = x402HTTPClientSync(client)
        _SESSION = x402_requests(client).__enter__()
    return _SESSION, _HTTP


class RemoteLab:
    pays_itself = True

    def __init__(self, lab_id: str, tier: str = "evidence"):
        self.lab_id, self.tier = lab_id, tier
        self.url = LAB_URLS[lab_id]
        self.last_payment: dict | None = None
        self.history: dict = {}
        self._card = None

    @property
    def card(self) -> dict:
        if self._card is None:
            self._card = requests.get(f"{self.url}/card", timeout=10).json()
        return self._card

    @property
    def size_bucket(self) -> str:
        return self.card.get("size", "unknown")

    def preflight(self, request_json: str) -> dict:
        return requests.post(f"{self.url}/preflight", data=request_json, timeout=30).json()

    def handle(self, request_json: str) -> str:
        session, http = _x402_session()
        import time
        for attempt in range(4):  # transient facilitator settlement failures: retry with a fresh authorization (stays inside the 30 s SLA)
            r = session.post(f"{self.url}/query/{self.tier}", data=request_json, timeout=120,
                             headers={"content-type": "application/json"})
            if r.status_code != 402:
                break
            time.sleep(2.0 * (attempt + 1))
        self.attempts = attempt + 1
        settle = None
        try:
            settle = http.get_payment_settle_response(lambda n: r.headers.get(n))
        except ValueError:
            pass
        if r.status_code == 402:
            reason = r.text[:200]
            import base64
            import json
            for h in ("payment-required", "payment-response"):
                try:
                    d = json.loads(base64.b64decode(r.headers[h] + "=="))
                    reason = d.get("error") or d.get("errorReason") or reason
                    break
                except Exception:
                    pass
            self.last_payment = {"status": "payment_failed", "detail": reason}
            raise RuntimeError(f"x402 payment failed at {self.lab_id}: {reason}")
        self.last_payment = {
            "status": "settled" if (r.status_code == 200 and settle and getattr(settle, "success", False)) else
                      ("not_charged (gate refused)" if r.status_code == 409 else f"http_{r.status_code}"),
            "transaction": getattr(settle, "transaction", None) if settle else None,
            "network": getattr(settle, "network", None) if settle else None,
            "payer": getattr(settle, "payer", None) if settle else None,
        }
        return LabResult.model_validate(r.json()).model_dump_json()
