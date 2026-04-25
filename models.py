# models.py
from __future__ import annotations
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class HallucinationAction(BaseModel):
    """
    What the agent sends to the environment at each step.

    has_hallucination:  True if the agent believes the LLM response
                        contains a factual error vs the reference document.
    hallucinated_claim: The exact wrong phrase from the LLM response.
                        Should be None if has_hallucination is False.
    correct_fact:       What the reference document actually says.
                        Should be None if has_hallucination is False.
    confidence:         How confident the agent is in its answer (0.0–1.0).
                        Used by the grader's confidence calibration formula.
    """
    model_config = ConfigDict(extra="ignore")

    has_hallucination: bool
    hallucinated_claim: Optional[str] = None
    correct_fact: Optional[str] = None
    confidence: float = Field(
        default=0.5,
        gt=0.0,
        lt=1.0,
        description="Agent confidence in its answer, strictly between 0.0 and 1.0"
    )


class HallucinationObservation(BaseModel):
    """
    What the environment returns after reset() or step().

    The agent reads reference_document and llm_response,
    then submits a HallucinationAction via step().
    """
    model_config = ConfigDict(extra="ignore")

    # Episode control
    done: bool = False
    reward: Optional[float] = None

    # Task tracking
    task_id: str
    sample_index: int
    total_samples: int

    # Core data — what the agent reads and acts on
    reference_document: str
    llm_response: str

    # Feedback from grader
    feedback: Optional[str] = None
    score: float = 0.0

    # Episode progress
    steps_taken: int = 0
    max_steps: int = 10

    # Auxiliary data (episode_id, hint, sample_score, etc.)
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Auxiliary info: episode_id, hint, per-step sample score"
    )


class HallucinationState(BaseModel):
    """
    Internal episode state returned by the /state endpoint.
    Used for debugging, monitoring, and the OpenEnv web UI.
    Safe defaults allow state() to be called before reset().
    """
    model_config = ConfigDict(extra="ignore")

    episode_id: Optional[str] = None
    task_id: str = ""

    sample_index: int = 0
    total_samples: int = 0

    episode_score: float = 0.0
    steps_taken: int = 0
    step_count: int = 0

    is_done: bool = False


# Generator Agent Models
# The generator agent creates hallucinations to fool the detector.
# It receives a reference document and produces a response
# that contains a subtle factual error.


class GeneratorAction(BaseModel):
    """Action submitted by the generator agent.

    The generator must create a response that:
    1. Sounds plausible and fluent
    2. Contains exactly one factual error
    3. Is subtle enough to fool a capable detector
    """

    generated_response: str
    error_type: str
    confidence: float = Field(
        default=0.5,
        gt=0.0,
        lt=1.0,
        description="Generator confidence, strictly between 0.0 and 1.0"
    )


class GeneratorObservation(BaseModel):
    """What the generator agent sees each step."""

    reference_document: str
    task_id: str
    previous_responses: list
    detector_caught: Optional[bool]
    fooling_rate: float
    done: bool
    reward: Optional[float]
    feedback: str
    steps_taken: int
    max_steps: int
    metadata: dict


class GeneratorState(BaseModel):
    """Generator episode state."""

    episode_id: Optional[str]
    task_id: str
    steps_taken: int
    fooling_rate: float
    is_done: bool


class AdversarialResult(BaseModel):
    """Result of one generator vs detector exchange."""

    reference_document: str
    generated_response: str
    generator_action: dict
    detector_action: dict
    generator_reward: float
    detector_reward: float
    detector_caught: bool
    difficulty: str
    breakdown: dict