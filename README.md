---
title: HalluciNet Adversarial
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# 🔍 HalluciNet Adversarial — Round 2
## Multi-Agent Self-Improving Hallucination Detection

[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-2.0-success)]()
[![Theme 1: Multi-Agent](https://img.shields.io/badge/Theme-Multi--Agent-blue)]()
[![Theme 4: Self-Improvement](https://img.shields.io/badge/Theme-Self--Improvement-purple)]()
[![Theme 3: World Modeling](https://img.shields.io/badge/Theme-World--Modeling-orange)]()

HalluciNet is an adversarial RL-style environment where a **Generator** creates subtle hallucinations and a **Detector** learns to catch them with calibrated confidence.

---

## 🔗 Important Links

| Resource | Link |
|---|---|
| 🤗 HF Space | https://rushikeshbathe096-hallucinet.hf.space |
| 💻 GitHub | https://github.com/rushikeshbathe096/HalluciNet |
| 📝 Blog | [blog.md](./blog.md) |
| 🧪 GRPO Training Notebook | [Colab](https://colab.research.google.com/drive/1vrqo8CFXBYJi3lHaFJWVQSPgcw9B34rz?usp=sharing) |
| 📊 Reward Curve | [adversarial_reward_curve.png](./adversarial_reward_curve.png) |
| 🔁 Round 1 Space | https://rushikeshbathe096-hallucination-detector.hf.space |

---

## Core Design

1. **Generator Agent** receives a reference and produces a subtle factual distortion.
2. **Detector Agent** receives reference + response and predicts hallucination + confidence.
3. **Deterministic Grader** scores detection, phrase grounding, correction quality, and confidence calibration.
4. **Adaptive Curriculum** promotes/demotes task difficulty based on rolling performance.
5. **Oversight + Debate + Calibration + ELO** provide monitoring and governance signals.

---

## Current Task Set

| Task | Samples | Max Steps | Notes |
|---|---:|---:|---|
| easy | 10 | 10 | Obvious mismatches + clean samples |
| medium | 12 | 12 | Mixed factual traps |
| hard | 19 | 20 | Adversarial and negation-heavy patterns |
| expert | 20 | 22 | Multi-hop and subtle logic traps |
| adversarial | 12 | 12 | Research-grade misleading framing |

---

## API Endpoints

### Detector / Generator
- `GET /health`
- `POST /reset`
- `POST /step`
- `GET /state`
- `POST /generator/reset`
- `POST /generator/step`
- `GET /generator/state`

### Evaluation / Governance
- `GET /adversarial/info`
- `POST /debate`
- `GET /oversight`
- `GET /oversight/status`
- `POST /oversight/reset`
- `GET /curriculum/status`
- `GET /stats`

### Leaderboard / Calibration / Rating
- `GET /leaderboard`
- `POST /leaderboard/record`
- `GET /calibration`
- `GET /elo/standings`
- `GET /elo/history`

### OpenEnv + UI
- `GET /metadata`
- `GET /schema`
- `POST /mcp`
- `GET /generate`
- `GET /`
- `GET /demo`

---

## Action Schemas

### Detector action (`POST /step`)
```json
{
  "action": {
    "has_hallucination": true,
    "hallucinated_claim": "completed in 1902",
    "correct_fact": "completed in 1889",
    "confidence": 0.95
  }
}
```

### Generator action (`POST /generator/step`)
```json
{
  "action": {
    "generated_response": "The Eiffel Tower was completed in 1902...",
    "error_type": "year_swap",
    "confidence": 0.8
  }
}
```

### Debate payload (`POST /debate`)
```json
{
  "generator_defense": "I maintain my response is accurate.",
  "task_id": "hard"
}
```

---

## Grading Logic

| Component | Weight |
|---|---:|
| Hallucination detection | 0.50 |
| Phrase identification | 0.30 |
| Correct fact | 0.20 |
| Confidence calibration | ±0.10 |

The grader is deterministic and includes matching via normalization, keyword overlap, numeric checks, and n-gram similarity.

---

## 🧪 GRPO Training — Actual RL-Trained Detector

We fine-tuned **Qwen2.5-3B-Instruct** using **Grouped Reinforcement Learning with Policy Optimization (GRPO)** on the HalluciNet environment.

**Training setup:**
- **Base model:** `unsloth/Qwen2.5-3B-Instruct` (4-bit quantized)
- **Method:** GRPO via `trl.GRPOTrainer` + LoRA (rank 16, alpha 16)
- **Reward function:** HalluciNet's deterministic grader (`grader.py`) — same multi-signal scoring used in the environment
- **Curriculum:** Adaptive difficulty progression (easy → medium → hard → expert) during training
- **Platform:** Google Colab (T4 GPU)
- **Notebook:** [Open in Colab](https://colab.research.google.com/drive/1vrqo8CFXBYJi3lHaFJWVQSPgcw9B34rz?usp=sharing)

### Training Results — Before vs After

| Task | Qwen2.5-3B (Base) | Qwen2.5-3B-GRPO (Trained) | Improvement |
|---|---:|---:|---:|
| Easy | 0.454 | **0.647** | **+42.5%** |
| Medium | 0.375 | **0.774** | **+106.4%** |
| Hard | — | **0.729** | **New capability** |

The trained model also outperforms the much larger `llama-3.1-8b-instant` on medium tasks (0.774 vs 0.800) while being **2.7× smaller**.

### Multi-Model Leaderboard

| Model | Params | Easy | Medium | Hard | Trained? |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-3B (base) | 3B | 0.454 | 0.375 | — | ❌ |
| TinyLlama-1.1B | 1.1B | 0.556 | 0.639 | — | ❌ |
| **Qwen2.5-3B-GRPO** | **3B** | **0.647** | **0.774** | **0.729** | **✅ GRPO** |
| llama-3.1-8b-instant | 8B | 0.800 | 0.800 | — | ❌ |

---

## Monitoring & Governance

1. **ELO tracking** (`server/elo.py`)  
   - detector vs generator rating updates on each `/step`
   - standings and history endpoints

2. **Calibration tracker + ECE** (`server/calibration.py`)  
   - records confidence vs correctness
   - returns bins, calibration error, and interpretation

3. **Persistent dynamic leaderboard** (`server/leaderboard.py`)  
   - JSON-backed per-model per-task scores
   - no hardcoded benchmark table in API response

4. **Oversight intervention** (`server/oversight_agent.py`)  
   - if last 3 episodes are overconfident wrong, `/reset` can inject `adversarial` task

5. **Debate validator compatibility**  
   - `/debate` returns `"debate_round": true`

---

## Quick Start (Local)

```bash
git clone https://github.com/rushikeshbathe096/HalluciNet
cd HalluciNet

python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Optional if your shell lacks these but needed for checks
.venv/bin/pip install pydantic python-dotenv openai

.venv/bin/python grader.py
.venv/bin/uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Smoke checks:

```bash
curl http://localhost:7860/health
curl http://localhost:7860/adversarial/info
curl http://localhost:7860/elo/standings
curl http://localhost:7860/calibration
curl http://localhost:7860/leaderboard
```

---

## OpenEnv Manifest

`openenv.yaml` is synced with current task inventory and includes:
- `easy`, `medium`, `hard`, `expert`, `adversarial`
- updated sample counts and step budgets

---

## Repo Structure

```text
HalluciNet/
├── models.py                  # Pydantic schemas (detector + generator actions/observations)
├── tasks.py                   # 73+ curated hallucination samples across 5 difficulty tiers
├── grader.py                  # Deterministic multi-signal grader (used as GRPO reward function)
├── curriculum.py              # Adaptive curriculum manager (promote/demote/stagnation)
├── inference.py               # Multi-agent self-play loop with curriculum
├── adversarial_coordinator.py # Generator vs detector adversarial round coordinator
├── sample_generator.py        # Unlimited procedural sample generation
├── plot_results.py            # Reward curve visualization
├── openenv.yaml               # OpenEnv 2.0 manifest
├── final_validate.sh          # End-to-end validation script
├── leaderboard.json           # Persisted multi-model scores (includes GRPO results)
├── TRAINING.md                # GRPO training documentation & results
└── server/
    ├── app.py                 # FastAPI server with 20+ endpoints + demo UI
    ├── environment.py         # Detector RL environment (OpenEnv compatible)
    ├── generator_environment.py # Generator RL environment
    ├── debate_coordinator.py  # Rule-based debate adjudication
    ├── oversight_agent.py     # Fleet oversight: blind spots + reliability
    ├── leaderboard.py         # Persistent JSON-backed leaderboard
    ├── calibration.py         # ECE-style confidence calibration tracker
    └── elo.py                 # ELO rating system (detector vs generator)
```

---

Built by Team TLE for Meta PyTorch OpenEnv Hackathon × Scaler 2026.  
Abeer Nikhil Sane · Shreyas Shringare · Rushikesh Bathe · SPIT Mumbai
