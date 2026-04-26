# HalluciNet Adversarial: Building a Detector That Knows When It Might Be Wrong

Hallucinations are not just wrong answers; they are often **confident wrong answers**.  
HalluciNet Adversarial is built around that core failure mode.

Instead of a single static benchmark, we run a two-agent loop:

1. a **Generator** that tries to craft subtle factual errors,
2. a **Detector** that must catch them and report calibrated confidence.

That pressure creates a harder and more realistic training/evaluation regime than one-pass QA checks.

---

## Why this environment exists

Most hallucination setups optimize detection accuracy only.  
In production, confidence quality matters just as much:

- a wrong answer with **high confidence** is dangerous,
- a wrong answer with **low confidence** is recoverable via fallback/human review.

HalluciNet rewards both correctness and calibration, so the model learns not only *what* is wrong, but also *when it is uncertain*.

---

## System architecture

## 1) Detector environment

The detector receives:
- `reference_document`
- `llm_response`

It submits:
- `has_hallucination`
- `hallucinated_claim`
- `correct_fact`
- `confidence`

Scoring comes from a deterministic grader (`grader.py`):

| Component | Weight |
|---|---:|
| Detection correctness | 0.50 |
| Phrase grounding | 0.30 |
| Correct fact | 0.20 |
| Confidence calibration | ±0.10 |

## 2) Generator environment

The generator attempts subtle edits (year/name/number/negation/entity-style errors) and is rewarded when the detector misses.

## 3) Curriculum manager

Difficulty is adapted over rolling windows:
- promote when strong sustained performance appears,
- demote when detector catch rate drops below threshold.

This keeps training pressure near capability boundaries.

---

## Current task inventory

The local codebase currently uses **5 tasks**:

| Task | Samples | Max Steps |
|---|---:|---:|
| easy | 10 | 10 |
| medium | 12 | 12 |
| hard | 19 | 20 |
| expert | 20 | 22 |
| adversarial | 12 | 12 |

The OpenEnv manifest (`openenv.yaml`) is aligned with these counts and includes the adversarial task.

---

## What was added in this iteration

### 1) ELO system

`server/elo.py` introduced `ELOTracker`:
- updates after each detector step,
- tracks detector vs generator rating trajectory,
- exposed via:
  - `GET /elo/standings`
  - `GET /elo/history`

### 2) Calibration tracker + ECE-style metric

`server/calibration.py` stores `(confidence, correctness)` and reports:
- per-bin confidence/accuracy,
- calibration gap,
- aggregate calibration error,
- qualitative interpretation.

Exposed via `GET /calibration`.

### 3) Dynamic leaderboard persistence

`server/leaderboard.py` persists model scores in `leaderboard.json`:
- no static hardcoded table in API response,
- supports incremental recording via `POST /leaderboard/record`.

### 4) Oversight became actionable

`server/oversight_agent.py` now has `should_inject_adversarial()`:
- if last 3 episodes are overconfident wrong,
- next `/reset` can force `task_id="adversarial"`.

This turns oversight into a control mechanism, not only a monitor.

### 5) Debate endpoint compatibility

`/debate` now returns `debate_round: true`, and validator payloads were adjusted to send a real `generator_defense`.

---

## The real proof: GRPO training

Everything above is infrastructure. The question is: **does it actually produce a better model?**

We trained `Qwen2.5-3B-Instruct` using GRPO (the technique behind DeepSeek-R1) with HalluciNet's deterministic grader as the reward function:

- **Method:** `trl.GRPOTrainer` + LoRA (4-bit QLoRA on T4 GPU)
- **Reward:** `grader.py` — same multi-signal scorer used in the environment
- **Curriculum:** adaptive difficulty during training (easy → medium → hard)
- **Notebook:** [Open in Colab](https://colab.research.google.com/drive/1vrqo8CFXBYJi3lHaFJWVQSPgcw9B34rz?usp=sharing)

### Results

| Task | Base Qwen2.5-3B | GRPO-Trained | Improvement |
|---|---:|---:|---:|
| Easy | 0.454 | **0.647** | **+42%** |
| Medium | 0.375 | **0.774** | **+106%** |
| Hard | — | **0.729** | **New capability** |

The trained 3B model approaches the zero-shot performance of `llama-3.1-8b-instant` (8B) — a model **2.7× larger**.

This is the core argument: the environment's reward signal is rich enough to drive genuine skill acquisition through RL, not just prompt engineering.

---

## API surface (high-level)

- Detector/generator control: `/reset`, `/step`, `/generator/reset`, `/generator/step`
- Evaluation and governance: `/debate`, `/oversight`, `/curriculum/status`, `/stats`
- Model analytics: `/leaderboard`, `/calibration`, `/elo/standings`, `/elo/history`
- Metadata compatibility: `/metadata`, `/schema`, `/mcp`
- UX: `/`, `/demo`

---

## Engineering notes

Strengths:
- deterministic grader doubles as GRPO reward function,
- explicit schemas,
- richer operational signals (ELO + calibration + oversight + debate),
- persisted leaderboard state,
- **actual RL training with measurable improvement** (not just an eval loop).

Active technical debt to track:
- process-global mutable runtime state in `server/app.py`,
- task arrays are currently shuffled from shared objects in memory,
- leaderboard record endpoint should enforce tighter input validation,
- expert-tier performance still low (3B model limitation).

---

## Why this matters

The project goal is not just "detect hallucinations."  
It is to train systems that:

1. catch factual mistakes under adversarial pressure,
2. remain calibrated under uncertainty,
3. expose governance signals that can trigger safer behavior,
4. **demonstrably improve through RL training on the environment's reward signal.**

We built the environment, trained a model on it, and proved the model gets meaningfully better. That combination is what moves a hallucination detector from benchmark toy to deployable component.

---

Built by Team TLE for Meta PyTorch OpenEnv Hackathon x Scaler 2026.
