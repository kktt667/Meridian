"""One lab as its own HTTP server, with x402 payment on the query route.

  venv/bin/python -m envoy.lab_server KOREA_KNHANES_2024 --port 4103

GET  /card          free: published card + computed reputation
POST /preflight     free: dry-run gate (yes/no + refusal code only)
POST /query/{tier}  PAID via x402 (Base Sepolia test USDC). A gate refusal returns 409, and the x402
                    middleware never settles payment on an error response, so refusals cost nothing.
If no lab wallet address is configured, the server runs without payment and says so in /card.
"""
from __future__ import annotations

import argparse
import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .catalog import LABS, public_catalog
from .config import PRICES_USD, X402_FACILITATOR, X402_NETWORK
from .lab import LabNode
from .market import reputation
from .schemas import LabResult


def lab_address(lab_id: str) -> str | None:
    return os.environ.get(f"LAB_{lab_id}_ADDRESS")


def make_app(lab_id: str) -> FastAPI:
    node = LabNode.load(lab_id)
    app = FastAPI(title=f"Lab node {lab_id}")
    pay_to = lab_address(lab_id)
    if pay_to:
        from x402.http import FacilitatorConfig, HTTPFacilitatorClient, PaymentOption
        from x402.http.middleware.fastapi import PaymentMiddlewareASGI
        from x402.http.types import RouteConfig
        from x402.mechanisms.evm.exact import ExactEvmServerScheme
        from x402.server import x402ResourceServer

        server = x402ResourceServer(HTTPFacilitatorClient(FacilitatorConfig(url=X402_FACILITATOR)))
        server.register(X402_NETWORK, ExactEvmServerScheme())
        routes = {f"POST /query/{tier}": RouteConfig(
            accepts=[PaymentOption(scheme="exact", pay_to=pay_to, price=f"${price:.2f}", network=X402_NETWORK)],
            mime_type="application/json", description=f"{lab_id}: gated aggregate query ({tier} tier)")
            for tier, price in PRICES_USD.items()}
        app.add_middleware(PaymentMiddlewareASGI, routes=routes, server=server)

    @app.get("/card")
    def card():
        c = next(x for x in public_catalog({lab_id: node.size_bucket}) if x["lab_id"] == lab_id)
        return {**c, "wallet": pay_to or "not configured (free mode)", "x402": bool(pay_to),
                "network": X402_NETWORK, "reputation": reputation(lab_id)}

    @app.post("/preflight")
    async def preflight(request: Request):
        return node.preflight((await request.body()).decode())

    @app.post("/query/{tier}")
    async def query(tier: str, request: Request):
        out = node.handle((await request.body()).decode())
        res = LabResult.model_validate_json(out)
        # 409 on refusal: x402 cancels settlement for any response >= 400, so the asker is never charged.
        return JSONResponse(content=res.model_dump(), status_code=200 if res.status == "ok" else 409)

    return app


if __name__ == "__main__":
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("lab_id", choices=list(LABS))
    ap.add_argument("--port", type=int, required=True)
    a = ap.parse_args()
    uvicorn.run(make_app(a.lab_id), host="127.0.0.1", port=a.port, log_level="warning")
