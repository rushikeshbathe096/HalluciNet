# server/debate_coordinator.py
"""
Rule-based debate adjudication: defense vs claim vs ground-truth phrases.
No hardcoded outcomes — values follow from string overlap and structure checks.
"""

from __future__ import annotations

import re
import os
import sys
from typing import List

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from grader import _preprocess  # reuse normalisation

_STOP = frozenset(
    "the a an is in of and to it was at by for on with that this or as be are were".split()
)


def _token_set(text: str) -> set:
    t = _preprocess(text or "")
    return {w for w in t.split() if w and w not in _STOP}


def _defense_references_response(defense: str, generated: str) -> bool:
    """>20% of defense content words also appear in generated response."""
    d = _token_set(defense)
    g = _token_set(generated)
    if not d:
        return False
    inter = len(d & g)
    return (inter / len(d)) > 0.20


def _defense_contradicts_ground_truth(defense: str, ground_truth_phrases: List[str]) -> bool:
    low = (defense or "").lower()
    for phrase in ground_truth_phrases or []:
        if not phrase:
            continue
        p = phrase.strip()
        if len(p) < 3:
            continue
        if p.lower() in low:
            return True
    return False


class DebateCoordinator:
    def __init__(self) -> None:
        self.debate_history: List[dict] = []

    def run_debate(
        self,
        reference: str,
        generated_response: str,
        detector_claim: str,
        generator_defense: str,
        ground_truth_phrases: list,
    ) -> dict:
        gtp = [str(p) for p in (ground_truth_phrases or []) if p]
        defense = (generator_defense or "").strip()
        n_words = len(defense.split())
        defense_valid = bool(
            defense
            and n_words > 10
            and _defense_references_response(defense, generated_response)
        )
        contradicts = _defense_contradicts_ground_truth(defense, gtp)
        det_maint = True
        if contradicts:
            outcome = "detector_wins"
            delta = -0.30
            reason = (
                "Defense quoted or included a ground-truth hallucination phrase, "
                "treating the generator as having admitted the error."
            )
        elif defense_valid and n_words > 20:
            outcome = "inconclusive"
            delta = 0.10
            reason = (
                "Defense is long, on-topic, and plausibly engaged with the response — "
                "inconclusive without further model-based adjudication."
            )
        else:
            outcome = "detector_wins"
            delta = -0.15
            reason = (
                "Defense failed length/grounding checks (empty, too short, or weak overlap "
                "with the generated text)."
            )

        gen_score = max(0.0, min(1.0, 0.5 + delta)) if not contradicts else max(0.0, 0.5 + delta)
        if outcome == "inconclusive":
            det_maint = True
        else:
            det_maint = True

        result = {
            "outcome": outcome,
            "defense_valid": defense_valid,
            "defense_contradicts_truth": contradicts,
            "generator_defense_score": round(gen_score, 4),
            "detector_maintained": det_maint,
            "adjudication_reason": reason,
            "generator_final_reward_delta": round(delta, 4),
        }
        self.debate_history.append(result)
        return result

    def get_stats(self) -> dict:
        if not self.debate_history:
            return {
                "total_debates": 0,
                "detector_wins": 0,
                "inconclusive": 0,
            }
        tot = len(self.debate_history)
        dw = sum(1 for r in self.debate_history if r.get("outcome") == "detector_wins")
        inc = sum(1 for r in self.debate_history if r.get("outcome") == "inconclusive")
        return {
            "total_debates": tot,
            "detector_wins": dw,
            "inconclusive": inc,
            "detector_win_rate": round(dw / max(tot, 1), 4),
        }
