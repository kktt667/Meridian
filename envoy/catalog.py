"""Marketplace catalog. Metadata each lab chooses to publish: never data.

Wallet addresses are placeholders for the (simulated) x402 flow.
"""
from .config import MIN_CASES_MODEL, MIN_CELL, MIN_POPULATION, PRICES_USD

PURPOSES = ["public_health_research", "academic_research", "clinical_guideline_review",
            "re_identification", "insurance_pricing", "marketing", "law_enforcement"]


def _policy(model: str, min_population: int = MIN_POPULATION, **kw) -> dict:
    return {"min_population": min_population, "min_cell": MIN_CELL, "min_cases": MIN_CASES_MODEL,
            "allowed_purposes": ["public_health_research", "academic_research", "clinical_guideline_review"],
            "refused_purposes": ["re_identification", "insurance_pricing", "marketing", "law_enforcement"],
            "max_queries_per_job": 8, "sla_seconds": 30, "stake_usd": 5.0, "agent_model": model, **kw}

LABS = {
    "US_BRFSS_2023": {
        "name": "US Behavioral Risk Factor Surveillance System 2023",
        "country": "United States", "region": "North America",
        "source": "CDC BRFSS 2023 annual survey (telephone)", "url": "https://www.cdc.gov/brfss/annual_data/annual_2023.html",
        "licence_note": "US government survey data; check CDC terms.",
        "groups": ["White", "Black", "Asian", "Hispanic", "AIAN", "Other"],
        "measured_bp": False, "policy": _policy("llama3.1:8b"),
        "notes": ["Telephone survey (landline + cell).", "Age is top-coded at 80.", "Race is an imputed broad category (_IMPRACE)."],
        "definitions": {
            "hypertension": "Ever told by a health professional you have high blood pressure (BPHIGH6; pregnancy-only and borderline excluded)",
            "stroke": "Ever told you had a stroke (CVDSTRK3)",
            "heart_attack": "Ever told you had a heart attack / myocardial infarction (CVDINFR4)",
        },
    },
    "BRAZIL_PNS_2019": {
        "name": "Brazil National Health Survey (PNS) 2019",
        "country": "Brazil", "region": "South America",
        "source": "IBGE Pesquisa Nacional de Saude 2019 (household interview)",
        "url": "https://www.ibge.gov.br/estatisticas/sociais/saude/9160-pesquisa-nacional-de-saude.html",
        "licence_note": "IBGE public microdata; check IBGE terms.",
        "groups": ["White", "Black", "Asian", "Brown", "Indigenous"],
        "measured_bp": False, "policy": _policy("mistral:7b"),
        "notes": ["Household interview of one selected adult per home.",
                  "The heart-attack item (Q06307) is only asked if the respondent reports heart disease (Q06306); "
                  "'no heart disease' must be coded as 'no heart attack'.", "Self-reported colour/race (C009)."],
        "definitions": {
            "hypertension": "Doctor ever diagnosed hypertension (Q00201)",
            "stroke": "Doctor ever diagnosed stroke (Q068)",
            "heart_attack": "Doctor ever diagnosed heart disease AND it was an infarction (Q06306/Q06307)",
        },
    },
    "KOREA_KNHANES_2024": {
        "name": "Korea National Health and Nutrition Examination Survey 2024",
        "country": "South Korea", "region": "East Asia",
        "source": "KDCA KNHANES 2024 (interview + examination)", "url": "https://knhanes.kdca.go.kr",
        "licence_note": "Registration-based download; no redistribution.",
        "groups": [],
        "measured_bp": True, "policy": _policy("qwen2.5:7b", min_population=30),
        "notes": ["Diagnosis items coded 0/1; 8 = not applicable and 9 = unknown are treated as missing.",
                  "Single survey year (2024); small numbers of young strokes.", "Blood pressure is the mean of the 2nd and 3rd readings."],
        "definitions": {
            "hypertension": "Doctor ever diagnosed hypertension (DI1_dg)",
            "stroke": "Doctor ever diagnosed stroke (DI3_dg)",
            "heart_attack": "Doctor ever diagnosed myocardial infarction (DI5_dg)",
            "measured_high_bp": "Examination SBP >= 140 or DBP >= 90 (mean of 2nd/3rd reading)",
        },
    },
    "US_NHANES_2017": {
        "name": "US National Health and Nutrition Examination Survey 2017-18",
        "country": "United States", "region": "North America",
        "source": "CDC NHANES 2017-2018 (interview + examination)", "url": "https://wwwn.cdc.gov/nchs/nhanes/",
        "licence_note": "US government survey data; check CDC terms.",
        "groups": ["White", "Black", "Asian", "Hispanic", "Other"],
        "measured_bp": True, "policy": _policy("llama3.1:8b"),
        "notes": ["2017-18 cycle; medical-condition items asked of adults 20+.", "Blood pressure: mean of up to three readings."],
        "definitions": {
            "hypertension": "Ever told by a doctor you had high blood pressure (BPQ020)",
            "stroke": "Ever told you had a stroke (MCQ160F)",
            "heart_attack": "Ever told you had a heart attack (MCQ160E)",
            "measured_high_bp": "Examination SBP >= 140 or DBP >= 90 (mean of up to 3 readings)",
        },
    },
}

GATE_POLICY = {
    "min_cell": MIN_CELL, "min_cases_for_estimates": MIN_CASES_MODEL, "min_population": MIN_POPULATION,
    "outputs": "aggregates only (counts >= min_cell, model coefficients); no row-level data, no free text",
}


def public_catalog(n_buckets: dict | None = None) -> list[dict]:
    out = []
    for lab_id, m in LABS.items():
        out.append({
            "lab_id": lab_id, **{k: m[k] for k in ("name", "country", "region", "source", "url", "licence_note", "groups", "measured_bp")},
            "variables": sorted(m["definitions"].keys()),
            "definitions": m["definitions"],
            "operations": ["logistic_or", "event_split_strata"] + (["undiagnosed_strata"] if m["measured_bp"] else []),
            "prices_usd": PRICES_USD, "gate": GATE_POLICY, "policy": m["policy"],
            "size": (n_buckets or {}).get(lab_id, "unknown"),
            "wallet": f"0xLAB_{lab_id[:10]}_PLACEHOLDER",
        })
    return out
