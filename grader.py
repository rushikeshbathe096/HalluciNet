# grader.py
"""
Deterministic grader for the Hallucination Detector RL Environment.

Scoring breakdown per sample:
    Check 1 — Hallucination detection:     0.50
    Check 2 — Phrase identification:       0.30 (multi-error aware, coverage-scaled)
    Check 3 — Correct fact provided:       0.20
    Calibration bonus:                    +0.10 × confidence if correct (additive)
                                          -0.10 × confidence if incorrect (additive)

Anti-cheat guarantees:
    - Always-True agent:  ~0.30 average
    - Always-False agent: ~0.25 average
    - Random agent:       ~0.39 average
    - Correct agent:      ~0.90+ average
"""

import re
from typing import Any, Dict, List, Tuple

WORD_TO_DIGIT = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90", "hundred": "100", "thousand": "1000",
    "million": "1000000", "billion": "1000000000",
}

_STOPWORDS = frozenset({
    "the", "a", "an", "is", "in", "of", "and", "to", "it",
    "was", "at", "by", "for", "on", "with", "that", "this",
})


def _normalise(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _apply_word_to_digit(text: str) -> str:
    return " ".join(WORD_TO_DIGIT.get(w, w) for w in text.split())


def _preprocess(text: str) -> str:
    return _apply_word_to_digit(_normalise(text))


def _extract_numbers(text: str) -> set:
    if not text:
        return set()
    return set(re.findall(r"\d[\d,\.]*", _preprocess(text)))


def _keyword_overlap(a: str, b: str) -> int:
    a_words = set(_preprocess(a).split()) - _STOPWORDS
    b_words = set(_preprocess(b).split()) - _STOPWORDS
    return len(a_words & b_words)


def _ngram_similarity(s1: str, s2: str, n: int = 3) -> float:
    s1 = _preprocess(s1).replace(" ", "")
    s2 = _preprocess(s2).replace(" ", "")
    if len(s1) < n or len(s2) < n:
        return 1.0 if s1 == s2 else 0.0
    ngrams1 = {s1[i:i + n] for i in range(len(s1) - n + 1)}
    ngrams2 = {s2[i:i + n] for i in range(len(s2) - n + 1)}
    union = len(ngrams1 | ngrams2)
    return len(ngrams1 & ngrams2) / union if union > 0 else 0.0


def _matches_any(candidate: str, ground_truths: List[str]) -> bool:
    if not candidate or not ground_truths:
        return False
    cand_prep = _preprocess(candidate)
    for gt in ground_truths:
        gt_prep = _preprocess(gt)
        if cand_prep in gt_prep or gt_prep in cand_prep:
            return True
        if _keyword_overlap(cand_prep, gt_prep) >= 3:
            return True
        cand_nums = _extract_numbers(cand_prep)
        gt_nums = _extract_numbers(gt_prep)
        if cand_nums and gt_nums and (cand_nums & gt_nums):
            return True
        if _ngram_similarity(cand_prep, gt_prep) >= 0.60:
            return True
    return False


def _coverage_ratio(claim: str, gt_phrases: List[str]) -> float:
    if not gt_phrases:
        return 0.0
    if not claim:
        return 0.0
    covered = sum(1 for phrase in gt_phrases if _matches_any(claim, [phrase]))
    return min(covered / len(gt_phrases), 1.0)


def grade(action: Any, sample: Dict[str, Any]) -> Tuple[float, str, Dict[str, float]]:
    """
    Score an agent's HallucinationAction against a sample's ground truth.

    Weights: detection=0.50, phrase=0.30, correction=0.20, calibration=±0.10 additive
    False alarm on clean sample: score=0.0 with calibration penalty (clamped to 0.0)
    """
    gt_has         = sample["ground_truth_has_hallucination"]
    gt_phrases     = sample["ground_truth_hallucinated_phrases"]
    gt_corrections = sample["ground_truth_corrections"]

    agent_has   = action.has_hallucination
    agent_claim = action.hallucinated_claim
    agent_fact  = action.correct_fact
    confidence  = float(action.confidence) if action.confidence is not None else 0.5
    confidence  = max(0.0, min(1.0, confidence))

    feedback_parts: List[str] = []
    base_score = 0.0
    detection_score = 0.0
    phrase_score = 0.0
    fact_score = 0.0

    # ── Case 1: False alarm on clean sample ──────────────────────────
    if agent_has and not gt_has:
        base_score = 0.0
        calibration = -0.10 * confidence
        feedback_parts.append("✗ False alarm — response is clean, no hallucination exists.")

    # ── Case 2: Missed hallucination ─────────────────────────────────
    elif not agent_has and gt_has:
        base_score = 0.0
        calibration = 0.0
        feedback_parts.append("✗ Missed — a hallucination exists but was not detected.")

    # ── Case 3: Correct clean sample identification ───────────────────
    elif not agent_has and not gt_has:
        detection_score = 1.0
        base_score = 1.0
        calibration = 0.10 * confidence
        feedback_parts.append("✓ Clean sample correctly identified — no hallucination.")

    # ── Case 4: Hallucination correctly detected ──────────────────────
    else:
        # Check 1 — Detection (0.50)
        detection_score = 0.50
        base_score += detection_score
        feedback_parts.append("✓ Hallucination correctly detected.")

        # Check 2 — Phrase identification (0.30, coverage-scaled)
        ratio = _coverage_ratio(agent_claim, gt_phrases)
        phrase_score = 0.30 * ratio
        base_score += phrase_score

        if ratio == 0.0:
            expected = gt_phrases[0] if gt_phrases else "N/A"
            feedback_parts.append(f"✗ Phrase not identified. Expected near: '{expected}'")
        elif ratio < 1.0:
            covered_count = round(ratio * len(gt_phrases))
            feedback_parts.append(
                f"⚠ Partial phrase match: {covered_count}/{len(gt_phrases)} errors identified."
            )
        else:
            feedback_parts.append("✓ Hallucinated phrase correctly identified.")

        # Check 3 — Correct fact (0.20)
        if _matches_any(agent_fact, gt_corrections):
            fact_score = 0.20
            base_score += fact_score
            feedback_parts.append("✓ Correct fact provided.")
        else:
            expected = gt_corrections[0] if gt_corrections else "N/A"
            feedback_parts.append(f"✗ Correct fact not matched. Expected near: '{expected}'")

        # Calibration: additive ±0.10
        is_correct = base_score >= 0.50
        calibration = 0.10 * confidence if is_correct else -0.10 * confidence

    # ── Apply calibration and clamp to [0.0, 1.0] ────────────────────
    final_score = base_score + calibration
    final_score = round(max(0.001, min(0.999, final_score)), 4)

    # ── Grade label ───────────────────────────────────────────────────
    feedback = " | ".join(feedback_parts)
    if final_score >= 0.90:
        label = "EXCELLENT"
    elif final_score >= 0.70:
        label = "GOOD"
    elif final_score >= 0.40:
        label = "PARTIAL"
    else:
        label = "INCORRECT"
    feedback += f" || {label}"

    breakdown = {
        "detection": round(detection_score, 4),
        "phrase": round(phrase_score, 4),
        "fact": round(fact_score, 4),
        "calibration": round(calibration, 4),
        "final": round(final_score, 4),
    }
    return final_score, feedback, breakdown


# ── Self-tests ────────────────────────────────────────────────────────

if __name__ == "__main__":
    from models import HallucinationAction

    hallucinated = {
        "ground_truth_has_hallucination": True,
        "ground_truth_hallucinated_phrases": ["completed in 1902"],
        "ground_truth_corrections": ["completed in 1889"],
    }
    clean = {
        "ground_truth_has_hallucination": False,
        "ground_truth_hallucinated_phrases": [],
        "ground_truth_corrections": [],
    }
    multi_error = {
        "ground_truth_has_hallucination": True,
        "ground_truth_hallucinated_phrases": ["28 member states", "19 countries"],
        "ground_truth_corrections": ["27 member states", "20 countries"],
    }
    numeric = {
        "ground_truth_has_hallucination": True,
        "ground_truth_hallucinated_phrases": ["contributed 83 percent of total revenue"],
        "ground_truth_corrections": ["contributed 38 percent of total revenue"],
    }

    def make(has, claim=None, fact=None, conf=0.99):
        return HallucinationAction(
            has_hallucination=has,
            hallucinated_claim=claim,
            correct_fact=fact,
            confidence=conf,
        )

    tests = [
        ("T1 Perfect hallucination answer",
         make(True, "completed in 1902", "completed in 1889"),
         hallucinated, lambda s: s >= 0.90),
        ("T2 Missed hallucination scores 0",
         make(False, conf=0.9),
         hallucinated, lambda s: s <= 0.002),
        ("T3 Determinism",
         make(True, "completed in 1902", "completed in 1889"),
         hallucinated,
         lambda s: grade(make(True, "completed in 1902", "completed in 1889"), hallucinated)[0] == s),
        ("T4 Clean sample correct",
         make(False), clean, lambda s: s >= 0.90),
        ("T5 Paraphrased claim still matches",
         make(True, "the year 1902 is incorrect", "it should be 1889"),
         hallucinated, lambda s: s >= 0.70),
        ("T6 False alarm on clean sample scores 0",
         make(True, "completed in 1902", "completed in 1889", conf=0.99),
         clean, lambda s: s <= 0.002),
        ("T7 Always-True exploit resistance",
         make(True, conf=0.99),
         hallucinated, lambda s: s <= 0.65),
        ("T8 Numeric match for digit-reversed error",
         make(True, "83 percent", "38 percent"),
         numeric, lambda s: s >= 0.60),
        ("T9 Multi-error partial coverage",
         make(True, "28 member states", "27 member states"),
         multi_error, lambda s: 0.50 <= s <= 1.00),
        ("T10 Multi-error no phrase match",
         make(True, "something completely wrong", "also wrong"),
         multi_error, lambda s: s <= 0.60),
    ]

    all_pass = True
    for desc, action, sample, check in tests:
        score, feedback, _ = grade(action, sample)
        passed = check(score)
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  {status} {desc}: score={score} | {feedback}")

    print()
    if all_pass:
        print("✓ All 10 grader tests passed.")
    else:
        print("✗ Some tests failed — fix before submission.")
