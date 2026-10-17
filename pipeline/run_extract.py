"""Sample real CVs, redact, run the three readers, save the T tensor.

usage: python run_extract.py --csv <path/to/Resume.csv> --out <dir> [--n 120] [--readers claude,ollama,regex] [--judge]
"""
import argparse, asyncio, json, os, time, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from features import D, IDS
from readers import redact, read_claude, read_ollama, read_regex, READERS

TARGET = "INFORMATION-TECHNOLOGY"

def load_env(path):
    if os.path.exists(path):
        for line in open(path):
            if "=" in line and not line.startswith("#"):
                k, v = line.strip().split("=", 1); os.environ.setdefault(k, v.strip().strip('"'))

def sample(csv, n, seed=7):
    df = pd.read_csv(csv)
    pos = df[df.Category == TARGET].sample(n=min(n, (df.Category == TARGET).sum()), random_state=seed)
    neg = df[df.Category != TARGET].sample(n=n, random_state=seed)
    out = pd.concat([pos, neg]).sample(frac=1, random_state=seed).reset_index(drop=True)
    out["label"] = (out.Category == TARGET).astype(int)
    out["text"] = out.Resume_str.map(redact)
    return out[["ID", "Category", "label", "text"]]

async def run_claude_all(texts, concurrency=8):
    import anthropic
    client = anthropic.AsyncAnthropic()
    sem = asyncio.Semaphore(concurrency)
    async def one(i, t):
        async with sem:
            for attempt in range(3):
                try:
                    return i, await read_claude(client, t)
                except Exception as e:  # noqa
                    err = e; await asyncio.sleep(2 * (attempt + 1))
            print(f"  claude failed on {i}: {err}", file=sys.stderr)
            return i, (np.full(D, np.nan), {}, (0, 0))
    return await asyncio.gather(*(one(i, t) for i, t in enumerate(texts)))

async def judge_all(texts, reruns=3, concurrency=8):
    """Baseline: the LLM decides directly (what the industry does). Rerun to measure inconsistency."""
    import anthropic
    client = anthropic.AsyncAnthropic()
    sem = asyncio.Semaphore(concurrency)
    schema = {"type": "object", "properties": {"shortlist": {"type": "boolean"}}, "required": ["shortlist"], "additionalProperties": False}
    async def one(i, t, k):
        async with sem:
            try:
                r = await client.messages.create(
                    model="claude-haiku-5-5", max_tokens=256,
                    system="You are screening CVs for an Information Technology role. Decide whether to shortlist.",
                    messages=[{"role": "user", "content": f"CV:\n<<<\n{t}\n>>>\nShortlist for the IT role?"}],
                    output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}})
                return i, k, int(json.loads(next(b.text for b in r.content if b.type == "text"))["shortlist"])
            except Exception as e:  # noqa
                print(f"  judge failed on {i}/{k}: {e}", file=sys.stderr); return i, k, -1
    res = await asyncio.gather(*(one(i, t, k) for i, t in enumerate(texts) for k in range(reruns)))
    J = np.full((len(texts), reruns), -1)
    for i, k, v in res: J[i, k] = v
    return J

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=120); ap.add_argument("--readers", default="claude,ollama,regex")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--append", action="store_true", help="add these readers to an existing run dir (same sample)")
    a = ap.parse_args()
    load_env(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
    os.makedirs(a.out, exist_ok=True)
    df = sample(a.csv, a.n)
    texts = df.text.tolist(); n = len(texts)
    readers = a.readers.split(",")
    T = np.full((n, len(readers), D), np.nan); raw = {}
    usage = {}; old_readers = []
    if a.append and os.path.exists(os.path.join(a.out, "T.npy")):
        old_meta = pd.read_csv(os.path.join(a.out, "meta.csv"))
        assert (old_meta.ID.values == df.ID.values).all(), "sample differs from existing run; use the same --n and csv"
        T_old = np.load(os.path.join(a.out, "T.npy")); old = json.load(open(os.path.join(a.out, "run.json")))
        old_readers = old["readers"]; usage = old.get("usage_tokens", {})
        raw = json.load(open(os.path.join(a.out, "raw.json")))
        T = np.concatenate([T_old, T], axis=1)
    else:
        df.drop(columns=["text"]).to_csv(os.path.join(a.out, "meta.csv"), index=False)
    off = len(old_readers)
    print(f"{n} CVs ({df.label.sum()} IT / {n - df.label.sum()} other), readers={readers}")
    for r_idx, name in enumerate(readers, start=off):
        t0 = time.time()
        if name == "claude":
            res = asyncio.run(run_claude_all(texts))
            toks = np.array([u for _, (_, _, u) in res]); usage["claude"] = toks.sum(0).tolist()
            for i, (v, obj, _) in res: T[i, r_idx] = v; raw.setdefault(str(i), {})["claude"] = obj
        elif name.startswith("ollama"):
            for i, t in enumerate(texts):
                try: v, obj, _ = read_ollama(t, model=READERS[name])
                except Exception as e: print(f"  ollama failed on {i}: {e}", file=sys.stderr); v, obj = np.full(D, np.nan), {}
                T[i, r_idx] = v; raw.setdefault(str(i), {})[name] = obj
                if i % 5 == 0: print(f"  {name} {i}/{n}  {time.time()-t0:.0f}s", flush=True)
        elif name == "regex":
            for i, t in enumerate(texts):
                v, obj, _ = read_regex(t); T[i, r_idx] = v; raw.setdefault(str(i), {})["regex"] = obj
        print(f"reader {name:7s} done in {time.time()-t0:.0f}s; found {100*np.mean(~np.isnan(T[:, r_idx])):.0f}% of fields")
    np.save(os.path.join(a.out, "T.npy"), T)
    readers = old_readers + readers
    json.dump({"readers": readers, "models": {k: READERS[k] for k in readers}, "ids": IDS, "usage_tokens": usage,
               "dataset": "opensporks/resumes (HF, CC0-1.0), Resume.csv, downloaded 2026-10-17", "n": n, "target": TARGET},
              open(os.path.join(a.out, "run.json"), "w"), indent=1)
    json.dump(raw, open(os.path.join(a.out, "raw.json"), "w"))
    if a.judge:
        t0 = time.time()
        if os.environ.get("ANTHROPIC_API_KEY"):
            J = asyncio.run(judge_all(texts)); jm = "claude-haiku-5-5"
        else:
            from readers import judge_ollama
            J = np.array([[judge_ollama(t) for _ in range(3)] for t in texts]); jm = READERS["ollama"]
        np.save(os.path.join(a.out, "judge.npy"), J)
        rj = json.load(open(os.path.join(a.out, "run.json"))); rj["judge_model"] = jm
        json.dump(rj, open(os.path.join(a.out, "run.json"), "w"), indent=1)
        print(f"LLM-judge baseline ({jm}, 3 reruns) done in {time.time()-t0:.0f}s")
    print("saved to", a.out)

if __name__ == "__main__":
    main()
