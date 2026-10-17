"""Hash-chained audit log: each entry includes the hash of the previous one, so any edit breaks the chain."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path


class AuditLog:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.prev = "GENESIS"
        self.entries: list[dict] = []

    def add(self, kind: str, payload: dict) -> dict:
        body = {"ts": round(time.time(), 3), "kind": kind, "payload": payload, "prev": self.prev}
        h = hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()
        entry = {**body, "hash": h}
        self.prev = h
        self.entries.append(entry)
        with self.path.open("a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        return entry

    @staticmethod
    def verify(path: Path) -> tuple[bool, int]:
        prev, n = "GENESIS", 0
        for line in path.read_text().splitlines():
            e = json.loads(line)
            body = {k: e[k] for k in ("ts", "kind", "payload", "prev")}
            if e["prev"] != prev or hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest() != e["hash"]:
                return False, n
            prev, n = e["hash"], n + 1
        return True, n
