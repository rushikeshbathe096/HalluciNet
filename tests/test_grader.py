import sys
sys.path.insert(0, '.')
from tasks import get_task
from models import HallucinationAction
from grader import grade


for difficulty in ["easy", "medium", "hard"]:
    scores = []

    for s in get_task(difficulty):
        action = HallucinationAction(
            has_hallucination=s["ground_truth_has_hallucination"],
            hallucinated_claim=(s["ground_truth_hallucinated_phrases"][0] if s["ground_truth_hallucinated_phrases"] else None),
            correct_fact=(s["ground_truth_corrections"][0] if s["ground_truth_corrections"] else None),
            confidence=0.9
        )

        result = grade(action, s)
        score = result[0] if isinstance(result, tuple) else result
        scores.append(score)

    print(f"{difficulty}: {sum(scores)/len(scores):.4f}")