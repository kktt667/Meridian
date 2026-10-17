"""Create TESTNET wallets for the demo and append them to .env (git-ignored). Never use these on mainnet.

  venv/bin/python scripts/make_wallets.py

Then fund the RESEARCHER address with Base Sepolia test USDC from https://faucet.circle.com
(the x402 facilitator pays gas, so no test ETH is needed for the payer).
"""
from pathlib import Path

from eth_account import Account

ENV = Path(__file__).resolve().parent.parent / ".env"
LABS = ["US_BRFSS_2023", "BRAZIL_PNS_2019", "KOREA_KNHANES_2024", "US_NHANES_2017"]

existing = ENV.read_text() if ENV.exists() else ""
lines = []
if "RESEARCHER_EVM_PRIVATE_KEY" not in existing:
    a = Account.create()
    lines += [f"RESEARCHER_EVM_PRIVATE_KEY={a.key.hex()}", f"RESEARCHER_EVM_ADDRESS={a.address}"]
for lab in LABS:
    if f"LAB_{lab}_ADDRESS" not in existing:
        a = Account.create()
        lines += [f"LAB_{lab}_PRIVATE_KEY={a.key.hex()}", f"LAB_{lab}_ADDRESS={a.address}"]
if lines:
    with ENV.open("a") as f:
        f.write("\n# --- x402 testnet wallets (Base Sepolia) created by scripts/make_wallets.py ---\n" + "\n".join(lines) + "\n")
env = dict(l.split("=", 1) for l in ENV.read_text().splitlines() if "=" in l and not l.startswith("#"))
print("Researcher (payer) address:", env["RESEARCHER_EVM_ADDRESS"])
print("Fund it with Base Sepolia USDC: https://faucet.circle.com  (network: Base Sepolia)")
for lab in LABS:
    print(f"  {lab:20s} receives at {env[f'LAB_{lab}_ADDRESS']}")
