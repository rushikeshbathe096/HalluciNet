"""
adversarial_coordinator.py -- The Multi-Agent Adversarial Loop

Coordinates the generator and detector agents in a self-play loop.
The generator creates hallucinations. The detector tries to catch them.
Both improve through competition -- Theme 1 + Theme 4.

The coordinator tracks:
- Generator fooling rate (how often it tricks the detector)
- Detector catch rate (how often it catches the generator)
- Difficulty escalation (harder tasks when both agents improve)
- Convergence (when both agents plateau -- escalate to harder task)
"""

import asyncio
import json
import time
from typing import Optional
from openai import OpenAI
from models import HallucinationAction, GeneratorAction


def _bounded_confidence(value: object, default: float = 0.5) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = default
    return min(max(confidence, 0.001), 0.999)


# System Prompts

GENERATOR_SYSTEM_PROMPT = """You are an adversarial hallucination generator.

Your job is to create a SUBTLE factual error in a response to fool a
hallucination detector. The error must be:
1. Plausible -- the response should sound correct and fluent
2. Subtle -- not obviously wrong
3. Exactly one error -- do not introduce multiple errors
4. Based on the reference document -- change one specific fact

Error types you can create:
- year_swap: change a year by a small amount (1889 -> 1902)
- name_swap: change a person's name to a plausible wrong one
- number_swap: change a number slightly (21,196 -> 8,000)
- negation: add or remove a negation (liable -> not liable)
- entity_flip: swap who did what to whom
- unit_shift: change units (kilometres -> metres, keeping number)
- partial_truth: keep some facts correct, make one subtly wrong

Respond ONLY with valid JSON:
{
  "generated_response": "your hallucinated response here",
  "error_type": "year_swap",
  "confidence": 0.0 to 1.0
}"""

DETECTOR_SYSTEM_PROMPT = """You are a hallucination detection expert.

You will be given a reference document and an LLM response.
Determine if the response contains a factual error.

Respond ONLY with valid JSON:
{
  "has_hallucination": true or false,
  "hallucinated_claim": "exact wrong phrase or null",
  "correct_fact": "what reference says or null",
  "confidence": 0.0 to 1.0
}"""


# Agent Callers

def call_generator(
    client: OpenAI,
    model: str,
    reference: str,
    previous_caught: Optional[bool],
    fooling_rate: float,
) -> GeneratorAction:
    """Call the generator agent to create a hallucination."""

    hint = ""
    if previous_caught is True:
        hint = "\nYour last attempt was caught. Try a more subtle error type."
    elif previous_caught is False:
        hint = "\nYour last attempt succeeded! Continue with similar subtlety."

    user_content = f"""REFERENCE DOCUMENT:
{reference}

Your current fooling rate: {fooling_rate:.2f}
{hint}

Create a subtle hallucination based on this reference."""

    max_retries = 3
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.8,  # Higher temp for creative generation
                max_tokens=300,
                stream=False,
                timeout=45.0,
            )
            raw = (completion.choices[0].message.content or "").strip()
            if "```" in raw:
                raw = raw.split("```")[1].replace("json", "").strip()
            data = json.loads(raw)
            return GeneratorAction(
                generated_response=data.get("generated_response", ""),
                error_type=data.get("error_type", "year_swap"),
                confidence=_bounded_confidence(data.get("confidence", 0.5)),
            )
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"[GENERATOR ERROR] {e}")
                return GeneratorAction(
                    generated_response=reference,
                    error_type="year_swap",
                    confidence=0.5,
                )
            time.sleep(2**attempt)


def call_detector(
    client: OpenAI,
    model: str,
    reference: str,
    response: str,
) -> HallucinationAction:
    """Call the detector agent to catch the hallucination."""

    user_content = f"""REFERENCE DOCUMENT:
{reference}

LLM RESPONSE:
{response}"""

    max_retries = 3
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": DETECTOR_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0,
                max_tokens=256,
                stream=False,
                timeout=45.0,
            )
            raw = (completion.choices[0].message.content or "").strip()
            if "```" in raw:
                raw = raw.split("```")[1].replace("json", "").strip()
            data = json.loads(raw)
            return HallucinationAction(
                has_hallucination=bool(data.get("has_hallucination", False)),
                hallucinated_claim=data.get("hallucinated_claim") or None,
                correct_fact=data.get("correct_fact") or None,
                confidence=_bounded_confidence(data.get("confidence", 0.5)),
            )
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"[DETECTOR ERROR] {e}")
                return HallucinationAction(
                    has_hallucination=False,
                    hallucinated_claim=None,
                    correct_fact=None,
                    confidence=0.5,
                )
            time.sleep(2**attempt)


# Adversarial Round

def run_adversarial_round(
    client: OpenAI,
    model: str,
    reference: str,
    task_id: str,
    previous_caught: Optional[bool],
    generator_fooling_rate: float,
    round_num: int,
) -> dict:
    """
    Run one round of generator vs detector.

    1. Generator creates a hallucination
    2. Detector tries to catch it
    3. Both get rewards based on outcome
    4. Result logged for curriculum manager
    """

    print(f"\n[ROUND {round_num}] task={task_id}", flush=True)

    # Generator creates hallucination
    gen_action = call_generator(
        client, model, reference, previous_caught, generator_fooling_rate
    )

    print(
        f"[GENERATOR] type={gen_action.error_type} confidence={gen_action.confidence:.2f}",
        flush=True,
    )
    print(f"[GENERATED] {gen_action.generated_response[:100]}...", flush=True)

    # Detector tries to catch it
    det_action = call_detector(client, model, reference, gen_action.generated_response)

    print(
        f"[DETECTOR] caught={det_action.has_hallucination} confidence={det_action.confidence:.2f}",
        flush=True,
    )

    # Determine outcome
    # Generator wins if detector does NOT flag the hallucination
    detector_caught = det_action.has_hallucination
    generator_wins = not detector_caught

    # Generator reward
    if generator_wins:
        gen_reward = min(0.999, 0.70 + gen_action.confidence * 0.10)
        gen_feedback = "SUCCESS -- detector missed your hallucination"
    else:
        gen_reward = max(0.001, 0.30 - gen_action.confidence * 0.10)
        gen_feedback = "CAUGHT -- detector found your hallucination"

    # Detector reward -- use existing grader
    from grader import grade
    from tasks import get_task

    samples = get_task(task_id)
    # Use deterministic index-based matching (same ordering as session loop)
    matching_sample = None
    if samples:
        sample_index = (round_num - 1) % len(samples)
        matching_sample = samples[sample_index]

    if matching_sample:
        graded = grade(det_action, matching_sample)
        if len(graded) == 3:
            det_score, det_feedback, det_breakdown = graded
        else:
            det_score, det_feedback = graded
            det_breakdown = {"detection": det_score, "final": det_score}
    else:
        # Fallback -- evaluate detection only
        det_score = 0.999 if detector_caught else 0.001
        det_feedback = "Detection evaluated"
        det_breakdown = {"detection": det_score, "final": det_score}

    result = {
        "round": round_num,
        "task_id": task_id,
        "reference": reference[:80] + "...",
        "generated_response": gen_action.generated_response[:80] + "...",
        "error_type": gen_action.error_type,
        "generator_confidence": gen_action.confidence,
        "detector_caught": detector_caught,
        "detector_confidence": det_action.confidence,
        "generator_reward": round(gen_reward, 4),
        "detector_reward": round(det_score, 4),
        "generator_wins": generator_wins,
        "gen_feedback": gen_feedback,
        "det_feedback": det_feedback,
    }

    print(
        f"[RESULT] generator_wins={generator_wins} gen_reward={gen_reward:.3f} det_reward={det_score:.3f}",
        flush=True,
    )

    return result


# Full Adversarial Session

def run_adversarial_session(
    client: OpenAI,
    model: str,
    task_id: str = "easy",
    rounds: int = 10,
) -> dict:
    """
    Run a full adversarial session between generator and detector.

    Both agents see the same reference documents.
    Generator tries to fool detector.
    Detector tries to catch generator.
    Results feed into curriculum manager.
    """
    from tasks import get_task

    samples = get_task(task_id)
    results = []
    generator_wins = 0
    detector_wins = 0
    previous_caught = None

    print(f"\n{'=' * 60}", flush=True)
    print(f"ADVERSARIAL SESSION -- task={task_id} rounds={rounds}", flush=True)
    print(f"{'=' * 60}", flush=True)

    for round_num in range(1, min(rounds, len(samples)) + 1):
        sample = samples[(round_num - 1) % len(samples)]
        reference = sample["reference_document"]

        gen_fooling_rate = generator_wins / max(round_num - 1, 1)

        result = run_adversarial_round(
            client,
            model,
            reference,
            task_id,
            previous_caught,
            gen_fooling_rate,
            round_num,
        )

        results.append(result)
        previous_caught = result["detector_caught"]

        if result["generator_wins"]:
            generator_wins += 1
        else:
            detector_wins += 1

    total = len(results)
    gen_fooling_rate = generator_wins / max(total, 1)
    det_catch_rate = detector_wins / max(total, 1)

    avg_gen_reward = sum(r["generator_reward"] for r in results) / max(total, 1)
    avg_det_reward = sum(r["detector_reward"] for r in results) / max(total, 1)

    summary = {
        "task_id": task_id,
        "total_rounds": total,
        "generator_wins": generator_wins,
        "detector_wins": detector_wins,
        "generator_fooling_rate": round(gen_fooling_rate, 4),
        "detector_catch_rate": round(det_catch_rate, 4),
        "avg_generator_reward": round(avg_gen_reward, 4),
        "avg_detector_reward": round(avg_det_reward, 4),
        "results": results,
    }

    print(f"\n{'=' * 60}", flush=True)
    print("SESSION SUMMARY", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"Generator wins: {generator_wins}/{total} ({gen_fooling_rate:.1%})", flush=True)
    print(f"Detector wins:  {detector_wins}/{total} ({det_catch_rate:.1%})", flush=True)
    print(f"Avg gen reward: {avg_gen_reward:.4f}", flush=True)
    print(f"Avg det reward: {avg_det_reward:.4f}", flush=True)
    print(f"{'=' * 60}\n", flush=True)

    return summary
