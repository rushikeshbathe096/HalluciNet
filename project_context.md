# HalluciNet Adversarial — Project Context (Current)

## 1) Repository Snapshot

- **Repo root:** `/home/rushikeshbathe096/ml/HalluciNet`
- **GitHub repo:** `rushikeshbathe096/HalluciNet`
- **Primary runtime:** FastAPI app at `server/app.py`
- **Current branch:** `main`
- **Deployment target used in validation scripts:** `https://rushikeshbathe096-hallucinet.hf.space`

This document reflects the **current local codebase state**, including recent additions (ELO, calibration tracking, dynamic leaderboard, oversight intervention, debate validator compatibility).

---

## 2) System Goal

HalluciNet is an adversarial hallucination-detection environment with two interacting roles:

1. **Detector**: judges if an LLM response hallucinates against a reference document.
2. **Generator**: produces subtle factual distortions to evade detection.

It includes:

- deterministic grading (`grader.py`)
- adaptive curriculum (`curriculum.py`)
- oversight analytics and intervention (`server/oversight_agent.py`)
- debate adjudication (`server/debate_coordinator.py`)
- persistent leaderboard (`server/leaderboard.py` + `leaderboard.json`)
- calibration tracking with ECE (`server/calibration.py`)
- ELO tracking (`server/elo.py`)

---

## 3) Task Inventory (Authoritative)

Derived from `tasks.py` (`count_samples()`):

| Task ID | Samples | Runtime Max Steps (`server/environment.py`) | Notes |
|---|---:|---:|---|
| easy | 10 | 10 | Obvious mismatches, includes clean samples |
| medium | 12 | 12 | Mixed errors, moderate traps |
| hard | 19 | 20 | Stronger adversarial patterning |
| expert | 20 | 22 | Multi-hop / subtle logic traps |
| adversarial | 12 | 12 | Research-style traps and misleading framing |

`openenv.yaml` has been aligned to include all five tasks and updated counts, including `adversarial`.

---

## 4) API Surface (`server/app.py`)

### Core detector/generator

- `GET /health`
- `POST /reset`
- `POST /step`
- `GET /state`
- `POST /generator/reset`
- `POST /generator/step`
- `GET /generator/state`
- `GET /adversarial/info`
- `GET /stats`
- `GET /generate`

### Competition + analytics

- `GET /leaderboard`
- `POST /leaderboard/record`
- `GET /oversight`
- `GET /oversight/status`
- `POST /oversight/reset`
- `GET /curriculum/status`
- `POST /debate`
- `GET /calibration`
- `GET /elo/standings`
- `GET /elo/history`

### OpenEnv compatibility / metadata

- `GET /metadata`
- `GET /schema`
- `POST /mcp`

### UI

- `GET /`
- `GET /demo`

---

## 5) Important Recent Behavior Changes (Local Code)

## 5.1 ELO system

- Added `server/elo.py` with `ELOTracker`.
- Wired in `/step`:
  - if `obs.score > 0.5`: detector wins ELO round
  - else generator wins ELO round
- Added:
  - `GET /elo/standings`
  - `GET /elo/history`

## 5.2 Calibration tracker (ECE)

- Added `server/calibration.py` with binning + ECE-like calibration error computation.
- Wired into `/step`:
  - records `(confidence, was_correct)` per detector action
- Added endpoint:
  - `GET /calibration`

## 5.3 Debate endpoint validator compatibility

- `/debate` response now includes:
  - `"debate_round": True`
- `final_validate.sh` debate check payload changed from `{}` to:
  - `{"generator_defense": "test defense"}`

## 5.4 Oversight action (not just monitoring)

- Added `OversightAgent.should_inject_adversarial()`.
- `/reset` now checks this and forces `task_id="adversarial"` when the last three episodes are overconfident misses.
- Logs:
  - `"[OVERSIGHT] Injecting adversarial sample"`

## 5.5 API consistency fixes

- `/adversarial/info` task list includes `adversarial`.
- `/stats` now reports dynamic detector episode count:
  - `len(oversight_agent.episode_history)`

---

## 6) Module Responsibilities

- `tasks.py`  
  Static curated datasets, task listing/count helper functions.

- `grader.py`  
  Deterministic scoring:
  - detection (0.50)
  - phrase (0.30, coverage-scaled)
  - correction (0.20)
  - confidence calibration bonus/penalty

- `curriculum.py`  
  Promotion/demotion policy over rolling windows with thresholds and session logging.

- `server/environment.py`  
  Detector episode state machine, reward shaping, step logs, summaries for oversight/curriculum.

- `server/generator_environment.py`  
  Generator-side episode environment.

- `server/oversight_agent.py`  
  Reliability, overconfidence rate, blind spot detection, and adversarial-injection decision.

- `server/debate_coordinator.py`  
  Rule-based adjudication for generator defense vs detector claim vs ground truth.

- `server/leaderboard.py`  
  JSON-backed model performance persistence and ranking.

- `server/calibration.py`  
  Confidence calibration bins + error computation.

- `server/elo.py`  
  Elo ratings/history for detector vs generator.

- `inference.py`, `adversarial_coordinator.py`  
  Multi-round orchestration and run logging.

---

## 7) Data Files

- `leaderboard.json`  
  Current persisted per-model/task scores used by `/leaderboard`.

- `adversarial_results.csv`, `adversarial_results_groq_run.csv`  
  Session-level logged outputs for experiments/runs.

- `adversarial_reward_curve.png`  
  Visualization artifact.

---

## 8) Validator / Operational Notes

## 8.1 `final_validate.sh`

Script checks:

- health and metadata endpoints
- oversight endpoint shape
- debate endpoint return semantics (`debate_round=True`)
- leaderboard presence
- openenv validation and build checks

Recent correction: debate check now sends a real body with `generator_defense`.

## 8.2 Local environment constraints seen during triage

- `pydantic`, `python-dotenv`, `openai` were needed for local import/self-test execution.
- `openenv` CLI was missing in this shell at triage time.
- `uvicorn/fastapi` availability can differ across environments; local import checks should run in the project venv/target runtime.

---

## 9) Current Known Gaps / Risks (Technical Debt)

1. **Global mutable singletons in `server/app.py`**  
   Environments and trackers are process-global; this can cause cross-session state bleed under concurrency.

2. **Task list mutation risk**  
   `get_task()` returns shared lists from `TASKS`; environment shuffles in place, mutating canonical order globally.

3. **Leaderboard input validation is permissive**  
   `POST /leaderboard/record` does not currently constrain task IDs or score ranges.

4. **Debate coordinator has simplifications**  
   Rule-based thresholds are static and `detector_maintained` currently always resolves true.

5. **Deployment drift risk**  
   Live HF endpoint responses can lag local code until commit + push + successful rebuild.

---

## 10) Quick Local Runbook

```bash
cd /home/rushikeshbathe096/ml/HalluciNet

# Install dependencies (recommended in venv)
python3 -m pip install -r requirements.txt

# Optional extra packages if environment is bare
python3 -m pip install pydantic python-dotenv openai

# Run grader self-check
python3 grader.py

# Run server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Smoke checks
curl http://localhost:7860/health
curl http://localhost:7860/adversarial/info
curl http://localhost:7860/stats
curl http://localhost:7860/elo/standings
curl http://localhost:7860/calibration
```

---

## 11) Current Intent of This Context File

This is the canonical engineering handoff snapshot for:

- architecture and API layout
- exact task inventory
- recently implemented local changes
- validation status and known risks

Update this file whenever API contracts, task inventory, validation scripts, or evaluation logic changes.
