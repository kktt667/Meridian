"""Feature registry for the 'shortlist for an IT role' decision.

Every reader must return exactly these fields. Proxies stay in the vector (audit-only)
but carry weight 0. Values: numbers, booleans as 0/1, and -1 meaning 'not stated'.
"""
import numpy as np

NOT_STATED = -1.0

FEATURES = [
    dict(id="it_years",     kind="num",  cap=10, tol=1.0, w=3.0, proxy=False,
         desc="Years of professional experience in IT roles (software, sysadmin, networking, IT support, data, QA)."),
    dict(id="prog_langs",   kind="num",  cap=6,  tol=1.0, w=2.0, proxy=False,
         desc="Number of distinct programming or scripting languages mentioned (e.g. Python, Java, SQL, C#, JavaScript, Bash, PowerShell)."),
    dict(id="infra_skills", kind="bool", cap=1,  tol=0.0, w=2.0, proxy=False,
         desc="Mentions infrastructure, networking or system-administration skills (servers, networks, Active Directory, cloud, Linux/Windows administration, VMware)."),
    dict(id="cs_degree",    kind="bool", cap=1,  tol=0.0, w=1.0, proxy=False,
         desc="Has a degree in computer science, information technology, information systems, software engineering or a closely related field."),
    dict(id="it_certs",     kind="num",  cap=4,  tol=1.0, w=1.0, proxy=False,
         desc="Number of IT certifications held (CompTIA, Cisco, Microsoft, AWS, ITIL, etc.)."),
    dict(id="mgmt_years",   kind="num",  cap=10, tol=1.0, w=0.5, proxy=False,
         desc="Years spent in people-management roles (manager, director, supervisor, team lead)."),
    dict(id="total_years",  kind="num",  cap=25, tol=2.0, w=0.0, proxy=True,
         desc="Total years of professional experience across all roles."),
    dict(id="non_it_years", kind="num",  cap=25, tol=2.0, w=0.0, proxy=False,
         desc="Years of professional experience in roles that are NOT IT."),
]

IDS   = [f["id"] for f in FEATURES]
D     = len(FEATURES)
CAP   = np.array([f["cap"] for f in FEATURES], float)
TOL   = np.array([f["tol"] for f in FEATURES], float)
W     = np.array([f["w"]   for f in FEATURES], float)
IS_NUM = np.array([f["kind"] == "num" for f in FEATURES])
PROXY  = np.array([f["proxy"] for f in FEATURES])

SYSTEM_PROMPT = (
    "You are a transcription tool. You read a CV and copy out specific facts as numbers. "
    "You never evaluate, rank, recommend or judge the candidate. "
    "If a fact is not stated in the CV, return -1 for its value and an empty quote. "
    "For each field also return a short verbatim quote (at most 80 characters, copied exactly "
    "from the CV) that is the evidence for the value. Booleans are 1 (true) or 0 (false)."
)

def json_schema():
    props = {}
    for f in FEATURES:
        props[f["id"]] = {
            "type": "object",
            "description": f["desc"],
            "properties": {
                "value": {"type": "number"},
                "quote": {"type": "string"},
            },
            "required": ["value", "quote"],
            "additionalProperties": False,
        }
    return {"type": "object", "properties": props, "required": IDS, "additionalProperties": False}

def to_vector(obj: dict) -> np.ndarray:
    """JSON reader output -> float vector of length D (NaN for not stated)."""
    v = np.full(D, np.nan)
    for j, f in enumerate(FEATURES):
        try:
            x = float(obj[f["id"]]["value"])
        except (KeyError, TypeError, ValueError):
            continue
        if x == NOT_STATED or x < 0:
            continue
        v[j] = min(x, 1.0) if f["kind"] == "bool" else x
    return v
