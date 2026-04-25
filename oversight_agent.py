class OversightAgent:
    """
    Monitors generator and detector behavior across episodes.
    Detects overconfidence, systematic blind spots, reward gaming.
    Fleet AI bonus prize target.
    """
    def __init__(self):
        self.episode_history = []

    def record(self, confidence: float, was_wrong: bool, error_type: str):
        self.episode_history.append({
            "confidence": confidence,
            "was_wrong": was_wrong,
            "error_type": error_type
        })

    def evaluate(self) -> dict:
        results = self.episode_history
        if not results:
            return {"reliability_score": 1.0, "overconfidence_rate": 0.0,
                    "blind_spots": [], "system_feedback": "No data yet"}

        overconfident_wrongs = [
            r for r in results
            if r["confidence"] > 0.8 and r["was_wrong"]
        ]

        error_counts = {}
        for r in results:
            if r["was_wrong"]:
                error_counts[r["error_type"]] = error_counts.get(r["error_type"], 0) + 1

        blind_spots = [k for k, v in error_counts.items() if v >= 3]
        reliability = 1.0 - (len(overconfident_wrongs) / max(len(results), 1))
        overconfidence_rate = len(overconfident_wrongs) / max(len(results), 1)

        if reliability > 0.8:
            feedback = "System performing reliably"
        elif reliability > 0.6:
            feedback = f"Moderate reliability. Blind spots: {blind_spots}"
        else:
            feedback = f"Low reliability. High overconfidence rate: {overconfidence_rate:.2f}"

        return {
            "reliability_score": round(reliability, 3),
            "overconfidence_rate": round(overconfidence_rate, 3),
            "blind_spots": blind_spots,
            "system_feedback": feedback
        }

    def reset(self):
        self.episode_history = []
