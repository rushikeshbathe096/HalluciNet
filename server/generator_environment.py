"""
generator_environment.py -- The Generator Agent Environment

The generator receives a reference document and must produce
a hallucinated response that fools the detector.

Reward design:
- If detector misses the hallucination: +1.0 (generator wins)
- If detector catches it: -0.5 (generator loses)
- Calibration bonus: if generator was confident AND fooled: +0.1
- Complexity bonus: more subtle errors get higher reward
- Anti-exploit: always-obvious errors get penalized

This implements Theme 1 (Multi-Agent) and Theme 4 (Self-Improvement):
The generator learns to create harder challenges as the detector improves.
"""

import uuid
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import Optional, Any
from openenv.core import Environment
from models import (
    GeneratorAction,
    GeneratorObservation,
    GeneratorState,
    HallucinationAction,
)
from tasks import get_task
from grader import grade


# Error type complexity scores -- more subtle errors get higher reward
ERROR_COMPLEXITY = {
    "year_swap": 0.30,  # easiest -- obvious
    "name_swap": 0.35,
    "number_swap": 0.40,
    "location_swap": 0.40,
    "negation": 0.60,  # harder -- requires logic
    "entity_flip": 0.65,
    "unit_shift": 0.70,
    "partial_truth": 0.80,  # hardest -- subtle
    "adversarial_clean": 0.90,  # hardest -- sounds wrong, is true
}


def evaluate_generator_response(
    generated_response: str,
    reference_document: str,
    error_type: str,
    generator_confidence: float,
) -> tuple[float, str, dict]:
    """
    Evaluate the generator's hallucination quality.

    A good hallucination:
    1. Contains exactly one factual error
    2. Is fluent and plausible
    3. Is not obviously wrong
    4. Matches the declared error type

    Returns: (reward, feedback, breakdown)
    """
    breakdown = {
        "fluency": 0.0,
        "subtlety": 0.0,
        "type_match": 0.0,
        "calibration": 0.0,
        "final": 0.0,
    }

    # Check fluency -- response must be non-empty and reasonable length
    if not generated_response or len(generated_response.strip()) < 20:
        return 0.001, "Generated response too short or empty.", breakdown

    if len(generated_response) > 500:
        return (
            0.001,
            "Generated response too long -- hallucinations should be concise.",
            breakdown,
        )

    breakdown["fluency"] = 0.30

    # Check subtlety -- response should not be identical to reference
    ref_words = set(reference_document.lower().split())
    gen_words = set(generated_response.lower().split())
    overlap = len(ref_words & gen_words) / max(len(ref_words), 1)

    if overlap > 0.95:
        return 0.001, "Generated response too similar to reference -- not a hallucination.", breakdown

    if overlap < 0.20:
        return 0.001, "Generated response too different from reference -- not plausible.", breakdown

    # Subtlety score based on overlap (ideal: 60-85% similar)
    if 0.60 <= overlap <= 0.85:
        breakdown["subtlety"] = 0.40
    elif 0.40 <= overlap < 0.60:
        breakdown["subtlety"] = 0.25
    else:
        breakdown["subtlety"] = 0.10

    # Error type complexity bonus
    complexity = ERROR_COMPLEXITY.get(error_type, 0.30)
    breakdown["type_match"] = complexity * 0.20

    # Calibration -- reward honest confidence
    base_score = breakdown["fluency"] + breakdown["subtlety"] + breakdown["type_match"]

    if base_score >= 0.70 and generator_confidence >= 0.70:
        breakdown["calibration"] = 0.10
    elif base_score < 0.50 and generator_confidence >= 0.80:
        breakdown["calibration"] = -0.10

    final = min(
        max(
            breakdown["fluency"]
            + breakdown["subtlety"]
            + breakdown["type_match"]
            + breakdown["calibration"],
            0.001,
        ),
        0.999,
    )

    breakdown["final"] = round(final, 4)

    feedback = (
        f"Fluency: {breakdown['fluency']:.2f} | "
        f"Subtlety: {breakdown['subtlety']:.2f} | "
        f"Type complexity: {breakdown['type_match']:.2f} | "
        f"Calibration: {breakdown['calibration']:.2f} | "
        f"Final: {final:.3f}"
    )

    return round(final, 4), feedback, breakdown


class GeneratorEnvironment(Environment[GeneratorAction, GeneratorObservation, GeneratorState]):
    """
    Environment for the generator agent.

    The generator learns to create subtle hallucinations that fool
    a capable detector. As the detector improves, the generator
    must create harder challenges -- recursive skill amplification.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._samples = []
        self._index = 0
        self._scores = []
        self._episode_id = None
        self._task_id = ""
        self._steps = 0
        self._done = False
        self._max_steps = 10
        self._previous_responses = []
        self._detector_caught_history = []

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> GeneratorObservation:
        task_id = kwargs.get("task_id", "easy")

        self._samples = get_task(task_id)
        self._task_id = task_id
        self._index = 0
        self._scores = []
        self._steps = 0
        self._done = False
        self._episode_id = episode_id or str(uuid.uuid4())
        self._previous_responses = []
        self._detector_caught_history = []
        self._max_steps = {"easy": 8, "medium": 10, "hard": 12, "expert": 15}.get(task_id, 8)

        first = self._samples[0]
        return GeneratorObservation(
            reference_document=first["reference_document"],
            task_id=self._task_id,
            previous_responses=[],
            detector_caught=None,
            fooling_rate=0.0,
            done=False,
            reward=None,
            feedback="Generate a subtle hallucination that will fool the detector agent.",
            steps_taken=0,
            max_steps=self._max_steps,
            metadata={
                "episode_id": self._episode_id,
                "ground_truth": first["ground_truth_hallucinated_phrases"],
                "hint": "Create a plausible but factually wrong response.",
            },
        )

    def step(
        self,
        action: GeneratorAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> GeneratorObservation:

        if self._done:
            raise RuntimeError("Episode done. Call reset() first.")

        if not self._samples or self._index >= len(self._samples):
            raise RuntimeError("No active episode. Call reset() first.")

        step_start = time.time()
        timeout_limit_s = float(timeout_s) if timeout_s is not None else 30.0

        self._steps += 1
        current_sample = self._samples[self._index]

        # Evaluate generator's hallucination quality
        gen_score, feedback, breakdown = evaluate_generator_response(
            generated_response=action.generated_response,
            reference_document=current_sample["reference_document"],
            error_type=action.error_type,
            generator_confidence=action.confidence,
        )

        self._scores.append(gen_score)
        self._previous_responses.append(action.generated_response)

        # Delta reward
        if len(self._scores) == 1:
            reward = gen_score
        else:
            previous_avg = sum(self._scores[:-1]) / (len(self._scores) - 1)
            reward = gen_score - previous_avg

        episode_score = sum(self._scores) / len(self._scores)
        fooling_rate = episode_score

        self._index += 1
        done = self._index >= len(self._samples) or self._steps >= self._max_steps
        self._done = done

        step_duration = time.time() - step_start
        if step_duration > timeout_limit_s:
            self._done = True
            return GeneratorObservation(
                reference_document="",
                task_id=self._task_id,
                previous_responses=self._previous_responses,
                detector_caught=None,
                fooling_rate=round(fooling_rate, 4),
                done=True,
                reward=0.001,
                feedback="Step timeout — 30 second limit exceeded",
                steps_taken=self._steps,
                max_steps=self._max_steps,
                metadata={"episode_id": self._episode_id, "timeout": True},
            )

        if done:
            return GeneratorObservation(
                reference_document="",
                task_id=self._task_id,
                previous_responses=self._previous_responses,
                detector_caught=None,
                fooling_rate=round(fooling_rate, 4),
                done=True,
                reward=round(reward, 4),
                feedback=f"Episode complete. Fooling rate: {fooling_rate:.4f}. {feedback}",
                steps_taken=self._steps,
                max_steps=self._max_steps,
                metadata={
                    "episode_id": self._episode_id,
                    "reward_breakdown": breakdown,
                },
            )

        nxt = self._samples[self._index]
        return GeneratorObservation(
            reference_document=nxt["reference_document"],
            task_id=self._task_id,
            previous_responses=self._previous_responses[-3:],
            detector_caught=None,
            fooling_rate=round(fooling_rate, 4),
            done=False,
            reward=round(reward, 4),
            feedback=feedback,
            steps_taken=self._steps,
            max_steps=self._max_steps,
            metadata={
                "episode_id": self._episode_id,
                "reward_breakdown": breakdown,
                "hint": nxt.get("hint", ""),
            },
        )

    def state(self) -> GeneratorState:
        return GeneratorState(
            episode_id=self._episode_id,
            task_id=self._task_id,
            steps_taken=self._steps,
            fooling_rate=round(sum(self._scores) / len(self._scores), 4) if self._scores else 0.0,
            is_done=self._done,
        )
