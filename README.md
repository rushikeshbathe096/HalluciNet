---
title: HalluciNet Adversarial
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# 🔍 HalluciNet Adversarial
## Multi-Agent Self-Improving Hallucination Detection

[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-2.0-success)]()
[![Theme 1: Multi-Agent](https://img.shields.io/badge/Theme-Multi--Agent-blue)]()
[![Theme 4: Self-Improvement](https://img.shields.io/badge/Theme-Self--Improvement-purple)]()
[![Theme 3: World Modeling](https://img.shields.io/badge/Theme-World--Modeling-orange)]()
[![Halluminate Bonus](https://img.shields.io/badge/Bonus-Halluminate-red)]()
[![Fleet AI Bonus](https://img.shields.io/badge/Bonus-Fleet--AI-orange)]()

> **"We didn't train a model to be right.  
> We trained it to know when it might be wrong.  
> That's a harder problem. That's what HalluciNet solves."**

---

## 🔗 Links

| Resource | Link |
|---|---|
| 🤗 **Live Environment** | https://rushikeshbathe096-hallucinet.hf.space |
| 💻 **GitHub** | https://github.com/rushikeshbathe096/HalluciNet |
| 📓 **GRPO Training Colab** | [Open in Colab](https://colab.research.google.com/drive/1vrqo8CFXBYJi3lHaFJWVQSPgcw9B34rz?usp=sharing) |
| 📝 **Training Details** | [TRAINING.md](./TRAINING.md) |
| 📝 **Blog Post** | [blog.md](./blog.md) |
| 🔁 **Round 1 Environment** | https://rushikeshbathe096-hallucination-detector.hf.space |

---

## 1. The Problem

LLMs hallucinate. Everyone knows this.

The real problem is not that they are wrong.  
The real problem is that they are **confidently wrong.**

> *"The Eiffel Tower was completed in 1902, two years after the Paris Exposition."*  
> Confidence: 0.94

That sentence sounds authoritative. It is completely wrong.  
The Eiffel Tower was completed in 1889.

A model that produces this with 0.94 confidence is not just incorrect —  
it is dangerous. In healthcare, legal, and financial AI, confident wrong  
answers cause real harm.

**Current benchmarks test correctness.**  
**Nobody trains models to know when they might be wrong.**

That is the capability gap HalluciNet closes.

---

## 2. The Environment

Three agents. One adversarial loop. Four difficulty tiers.

```
[Reference Document]
        │
        ▼
[Generator Agent] ──── creates subtle hallucination
        │
        ▼
[Hallucinated Response]
        │
        ▼
[Detector Agent] ──── flags it, quotes exact wrong phrase,
        │              states correct fact, assigns confidence
        │
   ┌────┴────┐
   │         │
 Caught     Missed
   │         │
   ▼         ▼
[Debate]  [Generator Reward+]
   │
   ▼
[Oversight Agent] ──── monitors patterns, flags blind spots
   │
   ▼
[Curriculum Manager] ── escalates difficulty when reliable
```

**What the agent sees:**
- Reference document (ground truth)
- LLM response (may contain hallucination)

**What the agent does:**
- Decides: hallucination present or clean?
- Quotes: exact wrong phrase from the response
- States: correct fact from the reference
- Assigns: calibrated confidence (0–1)

**What the agent gets rewarded for:**

| Component | Weight | What It Measures |
|---|---:|---|
| Hallucination detection | 0.50 | Did it catch the error? |
| Exact phrase identification | 0.30 | Can it quote what's wrong? |
| Correct fact stated | 0.20 | Does it know the truth? |
| Confidence calibration | ±0.10 | Does it know what it knows? |

**Hard to game:**  
Adversarial clean samples penalize detectors that flag everything.  
A detector with 100% recall but poor precision scores low.

**The Debate Round (unique to HalluciNet):**  
When the Detector flags a hallucination, the Generator gets one  
defense turn. The Detector re-evaluates. Ground truth adjudicates.  
Agents don't just compete — they argue.

**Five difficulty tiers:**

| Task | Samples | Challenge |
|---|---:|---|
| Easy | 10 | Year swaps, obvious number changes |
| Medium | 12 | Name swaps, location errors |
| Hard | 19 | Negation traps, entity flips, adversarial clean |
| Expert | 20 | Multi-hop reasoning, date arithmetic |
| Adversarial | 12 | Research-grade misleading framing |

---

## 3. Results

### Before vs After Training

**BEFORE training (untrained Qwen2.5-3B):**
```
Input:    "The Eiffel Tower was completed in 1902 in Paris, France."
Reference: "...completed in 1889..."
Agent:    has_hallucination = False
          confidence = 0.71
Result:   ❌ MISSED — confidently wrong about being right
```

**AFTER GRPO training (same model, same input):**
```
Input:    "The Eiffel Tower was completed in 1902 in Paris, France."
Reference: "...completed in 1889..."
Agent:    has_hallucination = True
          hallucinated_claim = "completed in 1902"
          correct_fact = "completed in 1889"
          confidence = 0.91
Result:   ✅ CAUGHT — quoted exact wrong phrase, stated correct fact
```

This is learned behavior. Not hardcoded logic.

### Multi-Model Benchmark

| Model | Params | Easy | Medium | Hard | Trained? |
|---|---:|---:|---:|---:|:---:|
| Qwen2.5-3B (base) | 3B | 0.454 | 0.375 | — | ❌ |
| TinyLlama-1.1B | 1.1B | 0.556 | 0.639 | — | ❌ |
| **Qwen2.5-3B-GRPO** | **3B** | **0.647** | **0.774** | **0.729** | ✅ |
| llama-3.1-8b-instant | 8B | 0.800 | 0.800 | — | ❌ |

Key insight: our GRPO-trained 3B model reaches **Hard task capability**  
that the untrained 3B model cannot attempt at all.  
It outperforms llama-3.1-8b on Medium while being **2.7× smaller.**

### Improvement Summary

| Task | Baseline | Trained | Improvement |
|---|---:|---:|---:|
| Easy | 0.454 | 0.647 | **+42.5%** |
| Medium | 0.375 | 0.774 | **+106.4%** |
| Hard | 0.000 | 0.729 | **New capability** |
| Overall (GRPO run) | 0.674 | 0.751 | **+11.4%** |

### What The Model Learned

Through GRPO training the model acquired five specific capabilities:

1. **Structured output discipline** — consistently produces valid JSON
2. **Phrase grounding** — locates exact wrong claim, not vague assertions
3. **Correction accuracy** — states what reference actually says
4. **Confidence calibration** — reports lower confidence when uncertain
5. **Clean sample resistance** — resists false positives on adversarial-clean samples

This is learned behavior. Not hardcoded logic.

---

## 4. Why It Matters

Every AI system deployed in healthcare, legal, or finance needs  
exactly this capability — a model that says *"I'm not sure"*  
when it is not sure.

Not because we told it to be honest.  
Because the environment made dishonesty unprofitable.

**Who cares:**
- AI safety researchers building evaluation benchmarks
- Teams deploying LLMs in high-stakes domains
- Anyone building systems where confident wrong answers cause harm

**Could a researcher write a paper on this?**  
Yes. Calibrated uncertainty in LLM outputs is an active research area.  
HalluciNet is the first RL training environment specifically targeting  
confidence calibration through adversarial self-play.

---

## Monitoring & Governance

| Component | File | What It Does |
|---|---|---|
| ELO Rating | `server/elo.py` | Tracks Generator vs Detector skill like chess |
| Calibration (ECE) | `server/calibration.py` | Confidence vs accuracy alignment |
| Leaderboard | `server/leaderboard.py` | Any model can be benchmarked live |
| Oversight Agent | `server/oversight_agent.py` | Detects blind spots, overconfidence |
| Debate Validator | `server/debate_coordinator.py` | Adjudicates Generator defense turns |
| Curriculum Manager | `curriculum.py` | Promotes difficulty when system reliable |

---

## Quick Start

```bash
# Test live environment
curl https://rushikeshbathe096-hallucinet.hf.space/health

# Run a detector episode
curl -X POST https://rushikeshbathe096-hallucinet.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "hard"}'

curl -X POST https://rushikeshbathe096-hallucinet.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "has_hallucination": true,
      "hallucinated_claim": "completed in 1902",
      "correct_fact": "completed in 1889",
      "confidence": 0.95
    }
  }'

# Check ELO standings
curl https://rushikeshbathe096-hallucinet.hf.space/elo/standings

# Check leaderboard
curl https://rushikeshbathe096-hallucinet.hf.space/leaderboard
```

---

## API Reference

### Core
| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/reset` | Start detector episode |
| POST | `/step` | Submit detector action |
| GET | `/state` | Current episode state |
| POST | `/generator/reset` | Start generator episode |
| POST | `/generator/step` | Submit generator action |

### Governance
| Method | Endpoint | Description |
|---|---|---|
| POST | `/debate` | Generator defense turn |
| GET | `/oversight/status` | System reliability score |
| GET | `/curriculum/status` | Current difficulty level |
| GET | `/world/model` | World model — Theme 3 |
| GET | `/training/summary` | GRPO training results |
| GET | `/elo/standings` | Agent ELO ratings |
| GET | `/calibration` | ECE calibration curve |
| GET | `/leaderboard` | Multi-model benchmark |
| POST | `/leaderboard/record` | Submit model scores |

---

## Repo Structure

```
HalluciNet/
├── models.py                   # Pydantic schemas
├── tasks.py                    # 73+ curated samples across 5 tiers
├── grader.py                   # Deterministic 4-component grader
├── curriculum.py               # Adaptive difficulty manager
├── adversarial_coordinator.py  # Generator vs Detector loop
├── inference.py                # Self-play training simulation
├── openenv.yaml                # OpenEnv 2.0 manifest
├── leaderboard.json            # Persisted benchmark scores
├── TRAINING.md                 # GRPO training documentation
└── server/
    ├── app.py                  # FastAPI — 20+ endpoints + UI
    ├── environment.py          # Detector RL environment
    ├── generator_environment.py
    ├── debate_coordinator.py
    ├── oversight_agent.py
    ├── leaderboard.py
    ├── calibration.py
    └── elo.py
```

---

## OpenEnv Compliance

```bash
openenv validate --verbose
# [OK] HalluciNet: Ready for multi-mode deployment
# [YES] docker
# [YES] openenv_serve  
# [YES] uv_run
# [YES] python_module
```

Both `HallucinationEnvironment` and `GeneratorEnvironment` inherit  
from `openenv.core.Environment` with proper `reset()`, `step()`,  
`state()` implementations.

---

## Training

Full GRPO training documentation: [TRAINING.md](./TRAINING.md)

**Base model:** `unsloth/Qwen2.5-3B-Instruct` (4-bit quantized)  
**Method:** GRPO via `trl.GRPOTrainer` + LoRA (rank 16)  
**Reward:** HalluciNet deterministic grader (same as environment)  
**Platform:** Google Colab T4 GPU  
**Notebook:** [Open in Colab](https://colab.research.google.com/drive/1vrqo8CFXBYJi3lHaFJWVQSPgcw9B34rz?usp=sharing)

The training pipeline validates the environment design end-to-end.  
The grader's deterministic reward enables stable GRPO training.  
The curriculum provides automatic difficulty scheduling during RL.  
The multi-signal reward trains all aspects simultaneously.

**The environment was designed for RL training.  
The GRPO results prove it works.**

---

Built for Meta PyTorch OpenEnv Hackathon × Scaler 2026  
**Team TLE** — Abeer Nikhil Sane · Shreyas Shringare · Rushikesh Bathe  
SPIT Mumbai
