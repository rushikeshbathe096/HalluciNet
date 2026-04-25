"""
Grader validation script.
Run with: python -m tests.evaluate_scores
All sections must pass before submission.
"""

import re
from tasks import get_task
from grader import grade
from models import HallucinationAction


# ── Simulated agents ──────────────────────────────────────────────────

def agent_easy(sample) -> HallucinationAction:
    """Strong agent — correctly detects, partially quotes phrase."""
    has = sample["ground_truth_has_hallucination"]
    if has:
        raw_phrase = sample["ground_truth_hallucinated_phrases"][0]
        words = raw_phrase.split()
        simulated_claim = " ".join(words[:3])
        raw_correction = sample["ground_truth_corrections"][0]
        corr_words = raw_correction.split()
        simulated_fact = " ".join(corr_words[:3])
    else:
        simulated_claim = None
        simulated_fact = None
    return HallucinationAction(
        has_hallucination=has,
        hallucinated_claim=simulated_claim,
        correct_fact=simulated_fact,
        confidence=0.90
    )


def agent_medium(sample) -> HallucinationAction:
    """Decent agent — detects correctly but gives vague descriptions."""
    has = sample["ground_truth_has_hallucination"]
    if has:
        raw_phrase = sample["ground_truth_hallucinated_phrases"][0]
        if re.search(r"\d{4}", raw_phrase):
            simulated_claim = "the year mentioned appears incorrect"
            simulated_fact = "the reference states a different year"
        elif re.search(r"\d+[\d,\.]*\s*(metres?|kilometers?|km|m|billion|million|percent|%)",
                       raw_phrase, re.IGNORECASE):
            simulated_claim = "the figure given seems too high or too low"
            simulated_fact = "the reference document states a different value"
        elif re.search(r"[A-Z][a-z]+ [A-Z][a-z]+", raw_phrase):
            words = raw_phrase.split()
            simulated_claim = words[-1] if words else "wrong name"
            simulated_fact = None
        else:
            words = [w for w in raw_phrase.split() if len(w) > 3]
            simulated_claim = words[0] if words else raw_phrase.split()[0]
            simulated_fact = None
    else:
        simulated_claim = None
        simulated_fact = None
    return HallucinationAction(
        has_hallucination=has,
        hallucinated_claim=simulated_claim,
        correct_fact=simulated_fact,
        confidence=0.60
    )


def agent_hard(sample) -> HallucinationAction:
    """Weak agent on hard samples — confused by distractors."""
    has = sample["ground_truth_has_hallucination"]
    if not has:
        simulated_has = True  # false alarm on clean samples
    else:
        simulated_has = True
    return HallucinationAction(
        has_hallucination=simulated_has,
        hallucinated_claim=None,
        correct_fact=None,
        confidence=0.45
    )


def agent_expert(sample) -> HallucinationAction:
    """Very weak agent on expert samples — multi-hop reasoning required."""
    has = sample["ground_truth_has_hallucination"]
    if not has:
        simulated_has = True  # false alarm on adversarial clean samples
    else:
        simulated_has = True
    return HallucinationAction(
        has_hallucination=simulated_has,
        hallucinated_claim=None,
        correct_fact=None,
        confidence=0.35
    )


def agent_always_true(sample) -> HallucinationAction:
    return HallucinationAction(
        has_hallucination=True,
        hallucinated_claim=None,
        correct_fact=None,
        confidence=1.0
    )


def agent_always_false(sample) -> HallucinationAction:
    return HallucinationAction(
        has_hallucination=False,
        hallucinated_claim=None,
        correct_fact=None,
        confidence=1.0
    )


def agent_random_claim(sample) -> HallucinationAction:
    return HallucinationAction(
        has_hallucination=sample["ground_truth_has_hallucination"],
        hallucinated_claim="something appears wrong here",
        correct_fact="the reference says something different",
        confidence=0.7
    )


# ── Evaluation runner ─────────────────────────────────────────────────

def evaluate_agent(agent_fn, task_id: str) -> dict:
    samples = get_task(task_id)
    scores = []
    for sample in samples:
        action = agent_fn(sample)
        score, _, _ = grade(action, sample)
        scores.append(score)
    return {
        "scores": scores,
        "average": round(sum(scores) / len(scores), 4),
        "min":     round(min(scores), 4),
        "max":     round(max(scores), 4),
    }


def check_determinism() -> bool:
    sample = get_task("easy")[0]
    action = HallucinationAction(
        has_hallucination=True,
        hallucinated_claim="completed in 1902",
        correct_fact="completed in 1889",
        confidence=0.9
    )
    s1, _, _ = grade(action, sample)
    s2, _, _ = grade(action, sample)
    return s1 == s2


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 55)
    print("GRADER VALIDATION — Run before Day 2 integration")
    print("=" * 55)

    agents = {
        "easy":   agent_easy,
        "medium": agent_medium,
        "hard":   agent_hard,
        "expert": agent_expert,
    }
    targets = {
        "easy":   (0.75, 1.00),
        "medium": (0.40, 0.75),
        "hard":   (0.20, 0.55),
        "expert": (0.10, 0.40),
    }

    print("\n── SECTION 1: Score distribution (realistic agents) ──")
    print(f"{'Task':<10} {'Average':>8} {'Min':>7} {'Max':>7}  Target")
    print("-" * 50)

    distribution_pass = True
    results = {}

    for task_id, agent_fn in agents.items():
        r = evaluate_agent(agent_fn, task_id)
        results[task_id] = r
        lo, hi = targets[task_id]
        in_range = lo <= r["average"] <= hi
        status = "✓" if in_range else "✗"
        if not in_range:
            distribution_pass = False
        print(f"{task_id:<10} {r['average']:>8.3f} {r['min']:>7.3f} {r['max']:>7.3f}  {status} [{lo:.2f}–{hi:.2f}]")

    if not distribution_pass:
        print("\n⚠  Distribution out of target range.")
    else:
        print("\n✓ Score distribution within target range.")

    print("\n── SECTION 2: Score separation (gap ≥ 0.15 required) ──")
    easy_avg   = results["easy"]["average"]
    medium_avg = results["medium"]["average"]
    hard_avg   = results["hard"]["average"]
    expert_avg = results["expert"]["average"]

    gap_em = round(easy_avg - medium_avg, 4)
    gap_mh = round(medium_avg - hard_avg, 4)
    order_ok = easy_avg > medium_avg > hard_avg

    print(f"Easy:   {easy_avg:.3f}")
    print(f"Medium: {medium_avg:.3f}  (gap from easy:   {gap_em:+.3f})")
    print(f"Hard:   {hard_avg:.3f}  (gap from medium: {gap_mh:+.3f})")
    print(f"Order Easy > Medium > Hard:    {'✓' if order_ok else '✗ FAIL'}")
    print(f"Gap Easy→Medium ≥ 0.15:        {'✓' if gap_em >= 0.15 else '✗ FAIL'}")
    print(f"Gap Medium→Hard ≥ 0.15:        {'✓' if gap_mh >= 0.15 else '✗ FAIL'}")

    separation_pass = order_ok and gap_em >= 0.15 and gap_mh >= 0.15
    if not separation_pass:
        print("\n⚠  Score separation insufficient.")

    print("\n── SECTION 3: Exploit resistance ──")
    exploit_pass = True

    for task_id in ["easy", "medium", "hard"]:
        at = evaluate_agent(agent_always_true,  task_id)["average"]
        af = evaluate_agent(agent_always_false, task_id)["average"]
        rc = evaluate_agent(agent_random_claim, task_id)["average"]

        at_ok = at <= 0.45
        af_ok = af <= 0.45
        rc_ok = rc <= 0.70

        if not (at_ok and af_ok and rc_ok):
            exploit_pass = False

        print(
            f"{task_id:<8} "
            f"always-True={at:.3f} {'✓' if at_ok else '✗'}  "
            f"always-False={af:.3f} {'✓' if af_ok else '✗'}  "
            f"random-claim={rc:.3f} {'✓' if rc_ok else '✗'}"
        )

    if not exploit_pass:
        print("\n⚠  Grader can be gamed.")
    else:
        print("\n✓ No exploit strategy scores above threshold.")

    print("\n── SECTION 4: Determinism ──")
    det = check_determinism()
    print(f"Same input → same output: {'✓' if det else '✗ FAIL'}")

    print("\n── SECTION 5: Per-sample detail (easy task) ──")
    for i, sample in enumerate(get_task("easy")):
        action = agent_easy(sample)
        score, feedback, _ = grade(action, sample)
        has_gt = sample["ground_truth_has_hallucination"]
        grade_label = feedback.split("||")[1].strip() if "||" in feedback else ""
        print(f"  Sample {i+1:>2} ({'hallucinated' if has_gt else 'clean':>11}): {score:.4f}  {grade_label}")

    print("\n── SECTION 6: Per-sample detail (medium task) ──")
    for i, sample in enumerate(get_task("medium")):
        action = agent_medium(sample)
        score, feedback, _ = grade(action, sample)
        has_gt = sample["ground_truth_has_hallucination"]
        grade_label = feedback.split("||")[1].strip() if "||" in feedback else ""
        print(f"  Sample {i+1:>2} ({'hallucinated' if has_gt else 'clean':>11}): {score:.4f}  {grade_label}")

    print("\n── SECTION 7: Per-sample detail (hard task) ──")
    for i, sample in enumerate(get_task("hard")):
        action = agent_hard(sample)
        score, feedback, _ = grade(action, sample)
        has_gt = sample["ground_truth_has_hallucination"]
        grade_label = feedback.split("||")[1].strip() if "||" in feedback else ""
        print(f"  Sample {i+1:>2} ({'hallucinated' if has_gt else 'clean':>11}): {score:.4f}  {grade_label}")

    print("\n" + "=" * 55)
    all_pass = distribution_pass and separation_pass and exploit_pass and det
    if all_pass:
        print("✓ ALL CHECKS PASSED — grader ready for submission")
    else:
        print("✗ SOME CHECKS FAILED — fix before submission")
        if not distribution_pass:
            print("  → Fix: adjust sample difficulty in tasks.py")
        if not separation_pass:
            print("  → Fix: increase difficulty gap in tasks.py")
        if not exploit_pass:
            print("  → Fix: review anti-cheat logic in grader.py")
        if not det:
            print("  → Fix: remove any randomness from grader.py")
    print("=" * 55)
