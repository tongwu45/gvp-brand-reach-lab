"""Brand-safety classification of video placements.

Three classifiers share one interface, ``predict(placements) -> np.ndarray`` of
predicted risk tiers (0 = safe ... 4 = floor):

* ``KeywordClassifier`` - the classic blocklist baseline.
* ``LLMClassifier``     - reads the title in context via an LLM (OpenAI-compatible,
                           Anthropic or Gemini endpoints), returning a validated tier.
* ``HybridClassifier``  - routes only the risky slice of supply to the LLM to
                           control cost, and trusts the keyword pass elsewhere.

``OracleClassifier`` returns the ground truth and is used as an upper bound.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError

from .config import FLOOR, HIGH, LOW, MEDIUM, SAFE, TOLERANCE


class Classifier(Protocol):
    name: str

    def predict(self, placements: pd.DataFrame) -> np.ndarray: ...


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------
KEYWORDS: dict[int, list[str]] = {
    FLOOR: ["terrorist", "execution", "propaganda", "recruitment"],
    HIGH: ["nude", "blood", "gore", "weed", "pills", "racist", "slur", "hate", "uncensored",
           "breast", "leaked", "graphic"],
    MEDIUM: ["violent", "murder", "overdose", "war", "crime", "criminal", "police", "opioid",
             "drug", "bomb", "shooting", "killer", "dead", "deaths"],
    LOW: ["crude", "swearing", "horror", "boxing", "fight", "beer", "explicit", "shot", "destroyed"],
}
_PATTERNS = {tier: re.compile(r"\b(" + "|".join(words) + r")\b", re.I) for tier, words in KEYWORDS.items()}


class KeywordClassifier:
    """Predicts the highest tier whose keyword list matches the title."""

    name = "keyword_blocklist"

    def predict_title(self, title: str) -> int:
        for tier in (FLOOR, HIGH, MEDIUM, LOW):
            if _PATTERNS[tier].search(title):
                return tier
        return SAFE

    def hits(self, title: str) -> bool:
        return self.predict_title(title) != SAFE

    def predict(self, placements: pd.DataFrame) -> np.ndarray:
        return np.array([self.predict_title(t) for t in placements.title])


class OracleClassifier:
    """Perfect labels: the ceiling any classifier could reach."""

    name = "oracle"

    def predict(self, placements: pd.DataFrame) -> np.ndarray:
        return placements.true_tier.values.copy()


class NoFilter:
    name = "no_filter"

    def predict(self, placements: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(placements), dtype=int)


# --------------------------------------------------------------------------
# LLM classifier
# --------------------------------------------------------------------------
class Verdict(BaseModel):
    tier: int = Field(ge=0, le=4)
    category: str
    rationale: str


PROMPT = """You are a brand-safety reviewer for video advertising.
Classify the video below into one risk tier, judging the actual meaning in context,
not isolated words (e.g. "shooting" in photography or a "bath bomb" are safe).

Tiers:
0 safe   - suitable for any brand
1 low    - mild profanity, mild or stylised violence, alcohol, horror
2 medium - news or documentary about violence, crime, drugs, disasters; health or
           political misinformation; dangerous stunts
3 high   - graphic violence, sexual content, illegal drug purchase, hate speech
4 floor  - terrorism, extreme gore; never monetisable

Video title: "{title}"

Answer with JSON only: {{"tier": <0-4>, "category": "<short label>", "rationale": "<one sentence>"}}"""

Transport = Callable[[str], str]   # prompt -> raw model text


def _post(url: str, headers: dict, body: dict) -> dict:
    import requests  # imported lazily so the simulator runs without it

    r = requests.post(url, headers=headers, json=body, timeout=60)
    r.raise_for_status()
    return r.json()


def openai_transport(model: str = "gpt-4o-mini", base_url: str | None = None) -> Transport:
    base = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    key = os.environ["OPENAI_API_KEY"]

    def call(prompt: str) -> str:
        out = _post(f"{base}/chat/completions", {"Authorization": f"Bearer {key}"}, {
            "model": model, "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": prompt}],
        })
        return out["choices"][0]["message"]["content"]

    return call


def anthropic_transport(model: str = "claude-haiku-4-5") -> Transport:
    key = os.environ["ANTHROPIC_API_KEY"]

    def call(prompt: str) -> str:
        out = _post("https://api.anthropic.com/v1/messages",
                    {"x-api-key": key, "anthropic-version": "2023-06-01"},
                    {"model": model, "max_tokens": 200, "temperature": 0,
                     "messages": [{"role": "user", "content": prompt}]})
        return out["content"][0]["text"]

    return call


def gemini_transport(model: str = "gemini-2.5-flash") -> Transport:
    key = os.environ["GEMINI_API_KEY"]

    def call(prompt: str) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
        out = _post(url, {}, {"contents": [{"parts": [{"text": prompt}]}],
                              "generationConfig": {"temperature": 0, "responseMimeType": "application/json"}})
        return out["candidates"][0]["content"]["parts"][0]["text"]

    return call


TRANSPORTS = {"openai": openai_transport, "anthropic": anthropic_transport, "gemini": gemini_transport}


def parse_verdict(raw: str) -> Verdict:
    """Extract and validate the JSON verdict from a model reply."""
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        raise ValueError(f"no JSON object in model reply: {raw[:80]!r}")
    return Verdict.model_validate(json.loads(match.group(0)))


class LLMClassifier:
    """Classifies each *unique* title once, with an on-disk cache.

    If the model reply cannot be parsed after ``retries`` attempts, the placement
    fails closed to ``fallback_tier`` (blocked for most advertisers) rather than
    silently passing.
    """

    name = "llm"

    def __init__(self, transport: Transport, cache_path: str | Path | None = None,
                 retries: int = 1, fallback_tier: int = HIGH):
        self.transport = transport
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache: dict[str, dict] = {}
        if self.cache_path and self.cache_path.exists():
            self.cache = json.loads(self.cache_path.read_text())
        self.retries = retries
        self.fallback_tier = fallback_tier
        self.calls = 0
        self.failures = 0

    @staticmethod
    def _key(title: str) -> str:
        return hashlib.sha1(title.strip().lower().encode()).hexdigest()

    def classify(self, title: str) -> Verdict:
        k = self._key(title)
        if k in self.cache:
            return Verdict(**self.cache[k])
        verdict = None
        for _ in range(self.retries + 1):
            self.calls += 1
            try:
                verdict = parse_verdict(self.transport(PROMPT.format(title=title)))
                break
            except (ValueError, ValidationError, KeyError):
                continue
        if verdict is None:
            self.failures += 1
            return Verdict(tier=self.fallback_tier, category="unparsed", rationale="fail-closed")
        self.cache[k] = verdict.model_dump()
        return verdict

    def save(self) -> None:
        if self.cache_path:
            self.cache_path.write_text(json.dumps(self.cache, indent=1))

    def predict(self, placements: pd.DataFrame) -> np.ndarray:
        preds = np.array([self.classify(t).tier for t in placements.title])
        self.save()
        return preds


class HybridClassifier:
    """Cost-aware routing: send a placement to the LLM only if the keyword pass
    flagged it (possible false positive) or it comes from higher-risk supply
    (possible false negative). Everything else keeps the cheap keyword label."""

    name = "hybrid"

    def __init__(self, llm: LLMClassifier, risky_tiers: tuple[str, ...] = ("gvp_long_tail", "gvp_mfa")):
        self.llm = llm
        self.kw = KeywordClassifier()
        self.risky_tiers = risky_tiers
        self.routed_share = 0.0

    def predict(self, placements: pd.DataFrame) -> np.ndarray:
        kw = self.kw.predict(placements)
        route = (kw != SAFE) | placements.pub_tier.isin(self.risky_tiers).values
        self.routed_share = float(route.mean())
        preds = kw.copy()
        idx = np.flatnonzero(route)
        if len(idx):
            preds[idx] = self.llm.predict(placements.iloc[idx])
        return preds


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def evaluate(pred: np.ndarray, placements: pd.DataFrame, suitability: str = "standard") -> dict:
    """Decision-level metrics for one advertiser suitability setting.

    over_block: share of genuinely suitable inventory (traffic-weighted) that gets blocked
    leak:       share of genuinely unsuitable inventory (traffic-weighted) that gets through
    """
    tol = TOLERANCE[suitability]
    w = placements.traffic.values
    truth_ok = placements.true_tier.values <= tol
    pred_ok = pred <= tol
    tricky = placements.tricky.values
    return {
        "tier_accuracy": float((pred == placements.true_tier.values).mean()),
        "over_block": float(w[truth_ok & ~pred_ok].sum() / w[truth_ok].sum()),
        "leak": float(w[~truth_ok & pred_ok].sum() / w[~truth_ok].sum()),
        "tricky_decision_accuracy": float((pred_ok[tricky] == truth_ok[tricky]).mean()) if tricky.any() else float("nan"),
    }
