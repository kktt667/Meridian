"""Payment layer. SIMULATED x402 flow with escrow.

Payment and authorisation are separate decisions: money goes into escrow first, is released only if the
lab's gate approves the query, and is refunded on refusal. Swap `SimulatedX402` for a real x402 client
(same four methods) without touching the pipeline.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import time
import uuid
from dataclasses import asdict, dataclass

from .config import PRICES_USD

_SECRET = os.environ.get("ENVOY_PAYMENT_SECRET", "demo-secret-not-for-production").encode()


@dataclass
class PaymentRequired:          # what an x402 "402 Payment Required" response carries
    scheme: str
    network: str
    asset: str
    amount_usd: float
    pay_to: str
    resource: str
    simulated: bool = True


@dataclass
class Receipt:
    receipt_id: str
    lab_id: str
    query_id: str
    amount_usd: float
    status: str                  # held -> released | refunded
    signature: str
    created: float
    simulated: bool = True


class BudgetExceeded(Exception):
    pass


class SimulatedX402:
    def __init__(self, budget_usd: float):
        self.budget = budget_usd
        self.spent = 0.0
        self.receipts: list[Receipt] = []
        self.penalties: list[dict] = []

    def quote(self, lab_id: str, query_id: str, tier: str) -> PaymentRequired:
        return PaymentRequired(scheme="exact", network="base-sepolia (simulated)", asset="USDC (test)",
                               amount_usd=PRICES_USD[tier], pay_to=f"0xLAB_{lab_id[:10]}_PLACEHOLDER",
                               resource=f"/labs/{lab_id}/query/{query_id}")

    def escrow(self, req: PaymentRequired, lab_id: str, query_id: str) -> Receipt:
        # Human-set budget cap: the agent can never spend beyond it.
        if self.spent + req.amount_usd > self.budget + 1e-9:
            raise BudgetExceeded(f"Budget ${self.budget:.2f} would be exceeded")
        self.spent += req.amount_usd
        rid = uuid.uuid4().hex[:12]
        sig = hmac.new(_SECRET, f"{rid}|{lab_id}|{query_id}|{req.amount_usd}".encode(), hashlib.sha256).hexdigest()[:24]
        r = Receipt(rid, lab_id, query_id, req.amount_usd, "held", sig, time.time())
        self.receipts.append(r)
        return r

    def release(self, r: Receipt) -> Receipt:
        r.status = "released"
        return r

    def refund(self, r: Receipt) -> Receipt:
        r.status = "refunded"
        self.spent -= r.amount_usd
        return r

    def penalise(self, r: Receipt, lab_stake_usd: float, fraction: float = 0.2) -> dict:
        """Verifiable failure (SLA breach, invalid output): full refund plus compensation from the lab's stake."""
        if r.status == "held":
            self.refund(r)
        comp = round(min(lab_stake_usd, r.amount_usd * fraction), 4)
        self.penalties.append({"receipt": r.receipt_id, "lab_id": r.lab_id, "compensation_usd": comp, "simulated": True})
        return self.penalties[-1]

    def record_x402(self, lab_id: str, query_id: str, amount_usd: float, payment: dict) -> Receipt:
        """A real x402 payment made by the lab's HTTP endpoint (settled on success, never settled on refusal)."""
        status = payment.get("status", "unknown")
        if status == "settled":
            self.spent += amount_usd
        r = Receipt(payment.get("transaction") or uuid.uuid4().hex[:12], lab_id, query_id, amount_usd, status,
                    payment.get("transaction") or "", time.time(), simulated=False)
        self.receipts.append(r)
        return r

    def summary(self) -> dict:
        return {"budget_usd": self.budget, "spent_usd": round(self.spent, 2),
                "simulated": all(r.simulated for r in self.receipts) if self.receipts else True,
                "real_x402_payments": sum(1 for r in self.receipts if not r.simulated),
                "receipts": [asdict(r) for r in self.receipts], "penalties": self.penalties,
                "rule": "Labs are paid per query whatever the answer; refunds on refusal; penalties only for verifiable failures."}
