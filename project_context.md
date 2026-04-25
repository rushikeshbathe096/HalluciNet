# HalluciNet Adversarial — Project Context

## Project Overview

**Name:** HalluciNet Adversarial (Round 2)
**Team:** Team TLE — Abeer Nikhil Sane, Shreyas Shringare, Rushikesh Bathe | SPIT Mumbai
**Hackathon:** Meta PyTorch OpenEnv Hackathon × Scaler 2026
**Themes:** Theme 1 (Multi-Agent) + Theme 4 (Self-Improvement)

**Live Space:** https://shreyshringare-hallucinet.hf.space
**GitHub:** https://github.com/shreyshringare/hallucinet_round2
**Round 1 Space:** https://rushikeshbathe096-hallucination-detector.hf.space

---

## What the System Does

HalluciNet is an adversarial reinforcement learning environment for hallucination detection.

Two agents compete in a self-play loop:

- **Generator Agent** — receives a reference document, creates a subtle factual hallucination, is rewarded when the detector misses it
- **Detector Agent** — receives the reference + generated response, must detect the hallucination, is graded by a deterministic 4-dimension grader

An **Adaptive Curriculum** escalates difficulty as both agents improve, implementing recursive skill amplification (Theme 4).

---

## Stack

| Component | Detail |
|-----------|--------|
| Framework | FastAPI + Uvicorn |
| Inference model | llama-3.1-8b-instant via Groq |
| Training model | Qwen2.5-3B-Instruct (4-bit QLoRA, GRPO via TRL + Unsloth) |
| Deployment | Docker on Hugging Face Spaces |
| OpenEnv version | 2.0, validated 6/6 |
| Python | 3.12 |

---

## Project Structure

```
hallucinet_round2/
├── models.py                      # HallucinationAction + GeneratorAction pydantic models
├── tasks.py                       # 53 curated samples (8 easy, 10 medium, 15 hard, 20 expert)
├── grader.py                      # Deterministic 4-dimension grader
├── adversarial_coordinator.py     # Multi-agent session runner (generator vs detector)
├── curriculum.py                  # Adaptive curriculum manager (promote / demote)
├── inference.py                   # Adversarial self-play inference script
├── sample_generator.py            # Unlimited sample generation
├── plot_results.py                # Reward curve plotting
├── adversarial_results.csv        # Real run results (latest)
├── adversarial_results_groq_run.csv  # Earlier Groq run results
├── adversarial_reward_curve.png   # Training evidence image
├── blog.md                        # HF blog post
├── openenv.yaml                   # OpenEnv manifest
├── Dockerfile                     # Container definition
├── requirements.txt               # Python dependencies
├── project_context.md             # This file
└── server/
    ├── app.py                     # FastAPI — all endpoints
    ├── environment.py             # Detector environment logic
    └── generator_environment.py   # Generator environment logic
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| /health | GET | Health check |
| /reset | POST | Start detector episode (body: `{"task_id": "easy"}`) |
| /step | POST | Submit detector action |
| /state | GET | Current detector episode state |
| /generator/reset | POST | Start generator episode |
| /generator/step | POST | Submit generator action |
| /adversarial/info | GET | System info |
| /generate | GET | Generate fresh samples |
| /leaderboard | GET | Error type fool rates |

---

## Task Design

| Level | Samples | Max Steps | Design |
|-------|---------|-----------|--------|
| easy | 8 | 10 | 1 obvious error per sample, 2 clean samples |
| medium | 10 | 12 | 2–3 mixed errors per sample, 2 clean samples |
| hard | 15 | 15 | Negation traps, entity confusion, adversarial clean samples |
| expert | 20 | 22 | Multi-hop reasoning, date arithmetic, necessary vs sufficient logic |

### Adversarial Clean Samples (hard + expert)
Counter-intuitive true facts that tempt the detector into false positives:
- Venus day is longer than Venus year (true)
- Triple Witching = final month of each quarter (semantically equivalent)
- Marie Curie — only winner in two *scientific* fields (Linus Pauling's Peace Prize disqualifies him)
- UN Security Council resolution — "blocked by veto" = "not passed" (double negation trap)

---

## Grader Design

**File:** `grader.py`

| Component | Weight | Description |
|-----------|--------|-------------|
| Detection | 0.50 | Correct has_hallucination flag |
| Phrase identification | 0.30 | Coverage-scaled over all ground truth phrases |
| Correct fact | 0.20 | Agent's correction matches ground truth |
| Confidence calibration | ±0.10 | Additive bonus/penalty based on confidence |

**Matching pipeline inside `_matches_any`:**
1. Substring match (exact)
2. Keyword overlap (non-stopword tokens)
3. Numeric intersection (digit extraction)
4. Trigram similarity (character n-gram Jaccard)

**Exploit resistance (post-fix values):**

| Strategy | Score |
|----------|-------|
| Always-True | ~0.30 |
| Always-False | ~0.25 |
| Random | ~0.39 |
| Correct calibrated | ~0.999 |

---

## Curriculum Logic

**File:** `curriculum.py`

**Promotion** (advance to harder task) triggers when, over a rolling window of 3 sessions:
- Detector catch rate avg >= 0.75, OR
- Generator fooling rate avg >= 0.75

**Demotion** (drop to easier task) triggers when, over a rolling window of 3 sessions:
- Detector catch rate avg < 0.40

Both reset the history window after a transition. Tracks `promotions` and `demotions` counters separately.

---

## Inference Script

**File:** `inference.py`

- Runs N sessions (default 6), each with R rounds (default 8)
- Each session calls `run_adversarial_session` from `adversarial_coordinator.py`
- Logs in mandatory OpenEnv format: `[START]`, `[STEP]`, `[END]`
- `[END]` rewards field reflects **detector rewards** (primary learning signal)
- Results saved to `adversarial_results.csv`

**Env vars:**
```
GROQ_API_KEY=gsk_...          # required
API_BASE_URL=https://api.groq.com/openai/v1
MODEL_NAME=llama-3.1-8b-instant
SESSIONS=6
ROUNDS_PER_SESSION=8
```

---

## Real Adversarial Run Results (latest CSV)

6 sessions with llama-3.1-8b-instant:

| Session | Task | Gen Reward | Det Reward | Combined | Decision |
|---------|------|-----------|-----------|---------|----------|
| 1 | easy | 0.2906 | 0.7210 | 0.5058 | stay |
| 2 | easy | 0.2880 | 0.4965 | 0.3923 | stay |
| 3 | easy | 0.2829 | 0.4676 | 0.3753 | promote |
| 4 | medium | 0.3623 | 0.5961 | 0.4792 | stay |
| 5 | medium | 0.4986 | 0.4476 | 0.4731 | stay |
| 6 | medium | 0.2870 | 0.7945 | 0.5408 | stay |

Curriculum promotions: 1 (easy → medium)

---

## OpenEnv Validation

```
passed: True
passed_count: 6/6
✅ openapi_version_available
✅ health_endpoint
✅ metadata_endpoint
✅ schema_endpoint
✅ mcp_endpoint
✅ mode_endpoint_consistency
```

---

## Changes Made in This Session

### Round 1 — Documentation
- Created `blog.md` — full HalluciNet blog post (two-agent system, confidence calibration, adversarial curriculum, exploit resistance, training stack, real results)
- Rewrote `README.md` — links table, endpoint docs, action schemas, difficulty levels, quick start, adversarial results, project structure, OpenEnv validation output

### Round 2 — Grader Fixes (grader.py)

**Problem 1: `_matches_any` was too lenient**
- `_keyword_overlap` threshold: `>= 2` → `>= 3`
- `_ngram_similarity` threshold: `>= 0.40` → `>= 0.60`
- Effect: Reduced false phrase matches. Medium scores stop inflating to ~0.99. Hard rewards become more stable.
- All 10 grader self-tests still pass (numeric matching acts as fallback for T5, T8).

**Problem 2: Calibration threshold misclassified partial correct answers**
- `is_correct = base_score >= 0.80` → `is_correct = base_score >= 0.50`
- Effect: An agent that correctly detects + partially identifies (base_score ~0.65) is no longer penalized on confidence. Removes contradictory reward signal on hard/expert.

### Round 2 — Task Data Fix (tasks.py)

**Problem: H5 (Amazon/Whole Foods) had an unparseable ellipsis in ground truth phrase**
- Old: `"Whole Foods ... acquiring Amazon"` — ellipsis breaks substring and ngram matching, H5 always scored 0 on phrase silently
- Fixed: `"acquiring Amazon"` — exact substring of the actual llm_response

### Round 2 — Curriculum Improvements (curriculum.py)

**Added demotion logic**
- New constant: `DEMOTION_THRESHOLD = 0.40`
- When detector avg catch rate < 0.40 over 3 sessions and not already at easiest level → demote one level, reset history
- Symmetric with existing promotion logic

**Added `demotions` counter**
- `self.demotions = 0` initialized alongside `self.promotions`
- Incremented on every demotion event
- Shown in `print_log()` summary line
- Returned in `status()` dict

**Fixed demotion task name logging bug**
- Old: `entry["task"]` used `self.current_level - (1 if promote else 0)` — demote case showed the new lower level instead of the level that was being played
- Fixed: explicit `played_level` calculation handles promote (`-1`), demote (`+1`), and stay (`0`) correctly

### Round 2 — Inference Fix (inference.py)

**`log_end` was logging generator rewards instead of detector rewards**
- Old: `rewards=all_gen_rewards[-ROUNDS_PER_SESSION:]`
- Fixed: `rewards=all_det_rewards[-ROUNDS_PER_SESSION:]`
- Effect: The `[END]` line in OpenEnv-format logs now reflects the correct agent's reward history

---

## Known Remaining Limitations

1. **Generator does not truly learn** — both agents use the same static LLM (llama-3.1-8b-instant) with different prompts. The adversarial loop is simulated, not parameter-updating RL on the generator side.

2. **Expert grader mismatch** — expert tasks use semantic/logical errors (margin vs markup, adiabatic vs isothermal). The string-based grader cannot reliably score paraphrased corrections. Mitigation: add multiple correction aliases per expert sample.

3. **Static task set** — 53 samples total. `sample_generator.py` exists but is not called during inference. Risk of memorization over many training runs.

4. **Combined reward masks individual signals** — `combined_score = (gen + det) / 2` is used for curriculum decisions. A dominant detector with a weak generator looks the same as balanced performance.

---

## How to Run Locally (Ubuntu)

```bash
cd /home/abeer/Desktop/r2/hallucinet_round2

# One-time setup
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
# edit .env and set GROQ_API_KEY

# Grader self-tests (no server needed)
.venv/bin/python grader.py

# Start server
.venv/bin/uvicorn server.app:app --host 0.0.0.0 --port 7860

# Health check (new terminal)
curl http://localhost:7860/health

# Run full inference
.venv/bin/python inference.py 2>&1 | tee run_output.txt

# Verify log format
grep -E '^\[(START|STEP|END)\]' run_output.txt

# Check curriculum decisions
grep '\[CURRICULUM\]' run_output.txt

# View results
cat adversarial_results.csv
```
