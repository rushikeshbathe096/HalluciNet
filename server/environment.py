# server/environment.py
import random
import uuid
import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import Optional, Any
from openenv.core import Environment
from models import HallucinationAction, HallucinationObservation, HallucinationState
from tasks import get_task
from grader import grade


class HallucinationEnvironment(Environment[HallucinationAction, HallucinationObservation, HallucinationState]):

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._samples     = []
        self._index       = 0
        self._scores      = []
        self._episode_id  = None
        self._task_id     = ""
        self._steps       = 0
        self._done        = False
        self._max_steps   = 10
        self._step_log: list = []
        self._last_submitted: Optional[dict] = None

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> HallucinationObservation:
        # task_id comes through kwargs from the HTTP request body
        task_id = kwargs.get("task_id", "easy")

        self._samples    = get_task(task_id)
        self._task_id    = task_id
        self._index      = 0
        random.shuffle(self._samples)
        self._scores     = []
        self._steps      = 0
        self._done       = False
        self._step_log   = []
        self._last_submitted = None
        self._episode_id = episode_id or str(uuid.uuid4())
        self._max_steps  = {
            "easy": 10,
            "medium": 12,
            "hard": 20,
            "expert": 22,
            "adversarial": 12,
        }.get(task_id, 10)

        first = self._samples[0]
        return HallucinationObservation(
            done=False,
            reward=None,
            task_id=self._task_id,
            sample_index=0,
            total_samples=len(self._samples),
            reference_document=first["reference_document"],
            llm_response=first["llm_response"],
            feedback="Episode started. Analyse both texts and submit your findings.",
            score=0.0,
            steps_taken=0,
            max_steps=self._max_steps,
            metadata={
                "episode_id": self._episode_id,
                "hint": first.get("hint", ""),
                "is_clean": first.get("is_clean"),
            }
        )

    def step(
        self,
        action: HallucinationAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> HallucinationObservation:

        if self._done:
            raise RuntimeError("Episode done. Call reset() first.")

        if not self._samples or self._index >= len(self._samples):
            raise RuntimeError(
            "No active episode. Call reset() before calling step()."
        )
        step_start = time.time()
        timeout_limit_s = float(timeout_s) if timeout_s is not None else 30.0

        self._steps += 1
        current_sample = self._samples[self._index]

        sample_score, feedback_text, breakdown = grade(action, current_sample)
        self._scores.append(sample_score)

        self._last_submitted = {"sample": current_sample, "action": action}
        self._step_log.append(
            {
                "task_id": self._task_id,
                "error_type": str(current_sample.get("error_type", "unknown")),
                "agent_has": bool(action.has_hallucination),
                "gt_has": bool(current_sample.get("ground_truth_has_hallucination")),
                "detector_confidence": float(action.confidence) if action.confidence is not None else 0.5,
                "detector_correct": bool(action.has_hallucination)
                == bool(current_sample.get("ground_truth_has_hallucination")),
            }
        )

        if len(self._scores) == 1:
            reward = sample_score
        else:
            previous_avg = sum(self._scores[:-1]) / (len(self._scores) - 1)
            reward = sample_score - previous_avg

        episode_score = sum(self._scores) / len(self._scores)
        episode_score = min(max(episode_score, 0.01), 0.99)  # clamp to strict (0, 1)
        self._index += 1

        done = (
            self._index >= len(self._samples)
            or self._steps >= self._max_steps
        )
        self._done = done

        step_duration = time.time() - step_start
        if step_duration > timeout_limit_s:
            self._done = True
            return HallucinationObservation(
                done=True,
                reward=0.001,
                feedback="Step timeout — 30 second limit exceeded",
                task_id=self._task_id,
                sample_index=self._index,
                total_samples=len(self._samples),
                reference_document="",
                llm_response="",
                score=0.001,
                steps_taken=self._steps,
                max_steps=self._max_steps,
                metadata={"episode_id": self._episode_id, "timeout": True},
            )

        if done:
            return HallucinationObservation(
                done=True,
                reward=round(reward, 4),
                task_id=self._task_id,
                sample_index=self._index,
                total_samples=len(self._samples),
                reference_document="",
                llm_response="",
                feedback=f"Episode complete. Final score: {episode_score:.4f}. {feedback_text}",
                score=round(episode_score, 4),
                steps_taken=self._steps,
                max_steps=self._max_steps,
                metadata={
                    "episode_id": self._episode_id,
                    "reward_breakdown": breakdown,
                }
            )
        else:
            nxt = self._samples[self._index]
            return HallucinationObservation(
                done=False,
                reward=round(reward, 4),
                task_id=self._task_id,
                sample_index=self._index,
                total_samples=len(self._samples),
                reference_document=nxt["reference_document"],
                llm_response=nxt["llm_response"],
                feedback=feedback_text,
                score=round(episode_score, 4),
                steps_taken=self._steps,
                max_steps=self._max_steps,
                metadata={
                    "episode_id": self._episode_id,
                    "hint": nxt.get("hint", ""),
                    "last_sample_score": sample_score,
                    "reward_breakdown": breakdown,
                }
            )

    def get_episode_summary(self) -> dict:
        """Call after a completed episode (when the last return had done=True)."""
        if not self._step_log:
            return {}
        log = self._step_log
        n = len(log)
        det_rate = sum(1 for r in log if r.get("detector_correct")) / max(n, 1)
        hallu = [r for r in log if r.get("gt_has")]
        gen_fool = (
            sum(1 for r in hallu if not r.get("agent_has", True)) / max(len(hallu), 1)
            if hallu
            else 0.0
        )
        return {
            "detector_catch_rate": round(det_rate, 4),
            "generator_fooling_rate": round(gen_fool, 4),
            "task_id": self._task_id,
        }

    def get_oversight_episode_dict(self) -> dict:
        """Single aggregate for OversightAgent.record_episode (done boundary)."""
        if not self._step_log:
            return {}
        log = self._step_log
        confs = [float(r.get("detector_confidence", 0.0)) for r in log]
        ets = [r.get("error_type", "unknown") for r in log]
        err = "mixed" if len(set(ets)) > 1 else ets[0] if ets else "unknown"
        return {
            "error_type": err,
            "detector_confidence": sum(confs) / max(len(confs), 1),
            "detector_correct": all(r.get("detector_correct") for r in log),
            "generator_confidence": 0.0,
            "generator_won": any(r.get("gt_has") and not r.get("agent_has") for r in log),
            "task_id": self._task_id,
        }

    def get_last_debate_context(self) -> Optional[dict]:
        if not self._last_submitted:
            return None
        s = self._last_submitted["sample"]
        a = self._last_submitted["action"]
        return {
            "reference_document": s.get("reference_document", ""),
            "llm_response": s.get("llm_response", ""),
            "detector_claim": a.hallucinated_claim or "" if a else "",
            "ground_truth_phrases": list(s.get("ground_truth_hallucinated_phrases") or []),
        }

    def state(self) -> HallucinationState:
        return HallucinationState(
            episode_id=self._episode_id,
            task_id=self._task_id,
            sample_index=self._index,
            total_samples=len(self._samples),
            episode_score=round(
                sum(self._scores) / len(self._scores), 4
            ) if self._scores else 0.0,
            steps_taken=self._steps,
            step_count=self._steps,
            is_done=self._done
        )
