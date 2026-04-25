"""
curriculum.py -- Adaptive Curriculum for Adversarial HalluciNet

Tracks both generator and detector performance.
Advances difficulty when both agents plateau -- recursive skill amplification.

Theme 4: Self-Improvement
Theme 1: Multi-Agent (both agents improve together)
"""

TASK_ORDER = ["easy", "medium", "hard", "expert"]
PROMOTION_THRESHOLD = 0.75  # detector must catch at this rate to advance
DEMOTION_THRESHOLD = 0.40   # detector below this rate triggers demotion
STAGNATION_ROUNDS = 5  # advance if no improvement after this many rounds


class AdversarialCurriculumManager:
    """
    Manages difficulty progression for the adversarial loop.

    Advances when:
    - Detector catch rate consistently above 0.75 (detector mastered the task)
    - Generator fooling rate consistently above 0.75 (generator mastered the task)
    - Either condition: escalate to harder task to challenge both

    This creates recursive skill amplification:
    Better detector -> generator must create harder hallucinations
    Better generator -> detector must learn to catch subtler errors
    Both improve together through competition
    """

    def __init__(self):
        self.current_level = 0
        self.detector_history = []
        self.generator_history = []
        self.window = 3
        self.promotions = 0
        self.demotions = 0
        self.session_log = []

    @property
    def current_task(self) -> str:
        return TASK_ORDER[self.current_level]

    def record_session(self, summary: dict) -> dict:
        det_rate = summary["detector_catch_rate"]
        gen_rate = summary["generator_fooling_rate"]

        self.detector_history.append(det_rate)
        self.generator_history.append(gen_rate)

        if len(self.detector_history) > self.window:
            self.detector_history.pop(0)
            self.generator_history.pop(0)

        det_avg = sum(self.detector_history) / len(self.detector_history)
        gen_avg = sum(self.generator_history) / len(self.generator_history)

        decision = "stay"
        reason = ""
        window_full = len(self.detector_history) >= self.window

        # Promote when detector masters current task
        if (
            window_full
            and det_avg >= PROMOTION_THRESHOLD
            and self.current_level < len(TASK_ORDER) - 1
        ):
            self.current_level += 1
            self.promotions += 1
            decision = "promote"
            reason = (
                f"Detector avg {det_avg:.2f} >= {PROMOTION_THRESHOLD} over "
                f"{self.window} sessions"
            )
            self.detector_history = []
            self.generator_history = []

        # Also promote when generator too dominant (needs harder task)
        elif (
            window_full
            and gen_avg >= PROMOTION_THRESHOLD
            and self.current_level < len(TASK_ORDER) - 1
        ):
            self.current_level += 1
            self.promotions += 1
            decision = "promote"
            reason = (
                f"Generator avg {gen_avg:.2f} >= {PROMOTION_THRESHOLD} -- "
                "harder task needed"
            )
            self.detector_history = []
            self.generator_history = []

        # Demote when detector consistently fails current task
        elif (
            window_full
            and det_avg < DEMOTION_THRESHOLD
            and self.current_level > 0
        ):
            self.current_level -= 1
            self.demotions += 1
            decision = "demote"
            reason = (
                f"Detector avg {det_avg:.2f} < {DEMOTION_THRESHOLD} over "
                f"{self.window} sessions"
            )
            self.detector_history = []
            self.generator_history = []

        if decision == "promote":
            played_level = self.current_level - 1
        elif decision == "demote":
            played_level = self.current_level + 1
        else:
            played_level = self.current_level
        played_level = max(0, min(len(TASK_ORDER) - 1, played_level))

        entry = {
            "task": TASK_ORDER[played_level],
            "det_rate": det_rate,
            "gen_rate": gen_rate,
            "det_avg": round(det_avg, 4),
            "gen_avg": round(gen_avg, 4),
            "decision": decision,
            "reason": reason,
            "next_task": self.current_task,
        }
        self.session_log.append(entry)
        return entry

    def print_log(self):
        print("\n" + "=" * 70)
        print("ADVERSARIAL CURRICULUM PROGRESSION")
        print("=" * 70)
        for i, e in enumerate(self.session_log):
            print(
                f"Session {i + 1:3d} | task={e['task']:<8} | "
                f"det={e['det_rate']:.2f} | gen={e['gen_rate']:.2f} | "
                f"decision={e['decision']:<8} | next={e['next_task']}"
            )
            if e["reason"]:
                print(f"          Reason: {e['reason']}")
        print(f"\nFinal level: {self.current_task} | Promotions: {self.promotions} | Demotions: {self.demotions}")
        print("=" * 70 + "\n")

    def status(self) -> dict:
        return {
            "current_task": self.current_task,
            "current_level": self.current_level,
            "promotions": self.promotions,
            "demotions": self.demotions,
            "sessions_completed": len(self.session_log),
        }
