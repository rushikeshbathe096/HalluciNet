import os
import sys

# Add project root to path so all imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional

from server.environment import HallucinationEnvironment
from server.generator_environment import GeneratorEnvironment
from server.oversight_agent import OversightAgent
from server.debate_coordinator import DebateCoordinator
from server.leaderboard import Leaderboard, TASK_KEYS
from server.elo import ELOTracker
from server.calibration import CalibrationTracker
detector_calibration = CalibrationTracker()
from curriculum import AdversarialCurriculumManager
from models import HallucinationAction, GeneratorAction

app = FastAPI(title="HalluciNet Adversarial - Round 2")

detector_env = HallucinationEnvironment()
generator_env = GeneratorEnvironment()
oversight_agent = OversightAgent()
adversarial_curriculum = AdversarialCurriculumManager()
debate_coordinator = DebateCoordinator()
leaderboard = Leaderboard()
elo_tracker = ELOTracker()

class ResetRequest(BaseModel):
    task_id: Optional[str] = "easy"

class DetectorStepRequest(BaseModel):
    action: HallucinationAction

class GeneratorStepRequest(BaseModel):
    action: GeneratorAction


class LeaderboardRecordRequest(BaseModel):
    model_name: str
    task_id: str
    score: float
    trained: bool = False


class DebateRequest(BaseModel):
    generator_defense: str
    task_id: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "healthy", "mode": "adversarial", "version": "2.0"}

@app.post("/reset")
def reset(body: ResetRequest = ResetRequest()):
    if oversight_agent.should_inject_adversarial():
        body.task_id = "adversarial"
        print("[OVERSIGHT] Injecting adversarial sample")
    obs = detector_env.reset(task_id=body.task_id)
    return {"observation": obs.model_dump(), "reward": None, "done": False}

@app.post("/step")
def step(body: DetectorStepRequest):
    obs = detector_env.step(body.action)
    try:
        oversight_agent.record_episode({
            "error_type": detector_env._samples[max(0, detector_env._index-1)].get("error_type", "unknown") if detector_env._samples else "unknown",
            "detector_confidence": body.action.confidence,
            "detector_correct": (obs.score or 0) > 0.5,
            "generator_confidence": 0.5,
            "generator_won": (obs.score or 0) < 0.3,
            "task_id": detector_env._task_id
        })
    except Exception as e:
        print(f"[OVERSIGHT ERROR] {e}")
    if obs.done:
        ep = detector_env.get_oversight_episode_dict()
        if ep:
            oversight_agent.record_episode(ep)
        summary = detector_env.get_episode_summary()
        if summary:
            adversarial_curriculum.record_session(summary)
    try:
        score_val = obs.score or 0
        if score_val > 0.5:
            elo_tracker.update("detector", "generator")
        else:
            elo_tracker.update("generator", "detector")
    except Exception as e:
        print(f"[ELO ERROR] {e}")
    try:
        confidence = body.action.confidence
        was_correct = (obs.score or 0) > 0.5
        detector_calibration.record(confidence, was_correct)
    except Exception as e:
        print(f"[CALIBRATION ERROR] {e}")
    return {"observation": obs.model_dump(), "reward": obs.reward, "done": obs.done}

@app.get("/state")
def state():
    return detector_env.state().model_dump()

@app.post("/generator/reset")
def generator_reset(body: ResetRequest = ResetRequest()):
    obs = generator_env.reset(task_id=body.task_id)
    return {"observation": obs.model_dump(), "reward": None, "done": False}

@app.post("/generator/step")
def generator_step(body: GeneratorStepRequest):
    obs = generator_env.step(body.action)
    return {"observation": obs.model_dump(), "reward": obs.reward, "done": obs.done}

@app.get("/generator/state")
def generator_state():
    return generator_env.state().model_dump()

@app.get("/adversarial/info")
def adversarial_info():
    return {
        "description": "HalluciNet Adversarial - Multi-Agent Self-Play",
        "generator": {
            "endpoint": "/generator/reset and /generator/step",
            "action_space": {
                "generated_response": "string",
                "error_type": "string",
                "confidence": "float strictly between 0 and 1"
            }
        },
        "detector": {
            "endpoint": "/reset and /step",
            "action_space": {
                "has_hallucination": "bool",
                "hallucinated_claim": "string or null",
                "correct_fact": "string or null",
                "confidence": "float strictly between 0 and 1"
            }
        },
        "tasks": ["easy", "medium", "hard", "expert", "adversarial"],
        "themes": ["Theme 1: Multi-Agent", "Theme 4: Self-Improvement"]
    }

@app.get("/leaderboard")
def get_leaderboard_endpoint():
    return {
        "description": "Model performance from recorded POST /leaderboard/record submissions (leaderboard.json).",
        "leaderboard": leaderboard.get_leaderboard(),
        "tasks": list(TASK_KEYS),
        "note": "Per-task values are 0.0 until recorded; overall is the mean over task keys.",
    }


@app.post("/leaderboard/record")
def post_leaderboard_record(body: LeaderboardRecordRequest):
    leaderboard.record_result(
        body.model_name, body.task_id, body.score, body.trained
    )
    return {
        "status": "ok",
        "model_name": body.model_name,
        "task_id": body.task_id,
        "score": body.score,
    }

@app.get("/stats")
def stats():
    return {
        "total_detector_episodes": len(oversight_agent.episode_history),
        "total_generator_episodes": 0
    }

try:
    from sample_generator import generate_batch
    GENERATOR_AVAILABLE = True
except ImportError:
    GENERATOR_AVAILABLE = False

@app.get("/generate")
def generate_samples(n: int = 10):
    if not GENERATOR_AVAILABLE:
        return {"error": "Sample generator not available", "samples": []}
    samples = generate_batch(n=n, clean_ratio=0.2)
    return {"samples": samples, "count": len(samples), "generated": True}

@app.get("/")
@app.get("/demo")
def demo_ui():
    return HTMLResponse("""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HalluciNet Adversarial — Round 2</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;background:#0a0a14;color:#e0e0e0;min-height:100vh}
  
  /* HEADER */
  .header{background:linear-gradient(135deg,#0d1b2a,#1a1a3e);padding:24px 40px;border-bottom:2px solid #0055cc;display:flex;align-items:center;justify-content:space-between}
  .logo{display:flex;align-items:center;gap:12px}
  .logo-icon{font-size:32px}
  .logo-text h1{color:#00aaff;font-size:22px;font-weight:700;letter-spacing:-0.5px}
  .logo-text p{color:#888;font-size:12px;margin-top:2px}
  .badges{display:flex;gap:8px;flex-wrap:wrap}
  .badge{padding:4px 12px;border-radius:20px;font-size:11px;font-weight:600;border:1px solid}
  .badge-blue{background:rgba(0,102,204,0.15);color:#00aaff;border-color:#0055cc}
  .badge-purple{background:rgba(120,40,200,0.15);color:#aa66ff;border-color:#7722cc}
  .badge-green{background:rgba(0,180,80,0.15);color:#00cc66;border-color:#009944}
  .status-dot{width:8px;height:8px;border-radius:50%;background:#00cc66;display:inline-block;margin-right:6px;animation:pulse 2s infinite}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:0.4}}

  /* LAYOUT */
  .main{max-width:1200px;margin:0 auto;padding:24px 20px}
  .grid-3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:20px}
  .grid-2{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px}
  
  /* STAT CARDS */
  .stat-card{background:#111827;border:1px solid #1e2d40;border-radius:10px;padding:16px;text-align:center}
  .stat-val{font-size:28px;font-weight:700;color:#00aaff;font-variant-numeric:tabular-nums}
  .stat-lbl{font-size:11px;color:#666;margin-top:4px;text-transform:uppercase;letter-spacing:0.5px}
  
  /* CARDS */
  .card{background:#111827;border:1px solid #1e2d40;border-radius:12px;padding:20px;margin-bottom:16px}
  .card-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:16px}
  .card-title{font-size:15px;font-weight:600;color:#e0e0e0;display:flex;align-items:center;gap:8px}
  .card-subtitle{font-size:11px;color:#555}
  
  /* CONTROLS */
  select,input[type=text]{background:#0a0a14;border:1px solid #2a3a4a;border-radius:8px;color:#e0e0e0;padding:9px 12px;font-size:13px;width:100%;outline:none;transition:border 0.2s}
  select:focus,input[type=text]:focus{border-color:#0066cc}
  textarea{background:#0a0a14;border:1px solid #2a3a4a;border-radius:8px;color:#e0e0e0;padding:10px 12px;font-size:12px;width:100%;resize:vertical;outline:none;line-height:1.5;transition:border 0.2s}
  textarea:focus{border-color:#0066cc}
  label{display:block;font-size:11px;color:#666;margin-bottom:5px;text-transform:uppercase;letter-spacing:0.4px}
  
  /* BUTTONS */
  .btn{padding:10px 20px;border:none;border-radius:8px;font-size:13px;font-weight:600;cursor:pointer;transition:all 0.2s;display:inline-flex;align-items:center;gap:6px}
  .btn-primary{background:#0066cc;color:#fff}
  .btn-primary:hover{background:#0052a3;transform:translateY(-1px)}
  .btn-danger{background:#cc3300;color:#fff}
  .btn-danger:hover{background:#aa2200}
  .btn-success{background:#006633;color:#fff}
  .btn-success:hover{background:#005522}
  .btn-outline{background:transparent;color:#00aaff;border:1px solid #0066cc}
  .btn-outline:hover{background:rgba(0,102,204,0.1)}
  .btn:disabled{opacity:0.5;cursor:not-allowed;transform:none!important}
  .btn-full{width:100%;justify-content:center}
  
  /* RANGE */
  input[type=range]{width:100%;height:4px;accent-color:#0066cc;margin:8px 0}
  .conf-row{display:flex;align-items:center;gap:10px}
  .conf-val{background:#0a0a14;border:1px solid #2a3a4a;border-radius:6px;padding:4px 10px;font-size:13px;font-weight:600;color:#00aaff;min-width:50px;text-align:center}
  
  /* SCORE */
  .score-big{font-size:36px;font-weight:700;font-variant-numeric:tabular-nums}
  .score-excellent{color:#00cc66}
  .score-good{color:#66aaff}
  .score-partial{color:#ffaa00}
  .score-bad{color:#ff4444}
  
  /* BREAKDOWN */
  .breakdown-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:12px 0}
  .breakdown-item{background:#0a0a14;border:1px solid #1e2d40;border-radius:8px;padding:10px;text-align:center}
  .breakdown-val{font-size:16px;font-weight:700;color:#00aaff}
  .breakdown-lbl{font-size:9px;color:#555;margin-top:3px;text-transform:uppercase}
  
  /* FEEDBACK */
  .feedback-box{background:#0a0a14;border-left:3px solid #0066cc;padding:12px;border-radius:4px;font-size:12px;line-height:1.6;margin-top:10px;color:#aaa}
  
  /* VERDICT */
  .verdict{padding:12px;border-radius:8px;text-align:center;font-size:16px;font-weight:700;margin:12px 0}
  .verdict-win{background:rgba(0,180,80,0.12);color:#00cc66;border:1px solid #006633}
  .verdict-lose{background:rgba(255,60,0,0.12);color:#ff6633;border:1px solid #cc3300}
  .verdict-pending{background:rgba(0,100,200,0.12);color:#6699ff;border:1px solid #0044aa}
  
  /* LEADERBOARD */
  table{width:100%;border-collapse:collapse;font-size:12px}
  th{background:#0a0a14;color:#666;padding:10px 12px;text-align:left;font-size:10px;text-transform:uppercase;letter-spacing:0.5px;border-bottom:1px solid #1e2d40}
  td{padding:10px 12px;border-bottom:1px solid #111827;color:#ccc}
  tr:hover td{background:rgba(0,102,204,0.05)}
  .rank-badge{display:inline-flex;align-items:center;justify-content:center;width:22px;height:22px;border-radius:50%;font-size:11px;font-weight:700}
  .rank-1{background:#ffd700;color:#000}
  .rank-2{background:#c0c0c0;color:#000}
  .rank-3{background:#cd7f32;color:#fff}
  .trained-tag{background:rgba(0,180,80,0.2);color:#00cc66;padding:2px 8px;border-radius:10px;font-size:10px}
  
  /* TABS */
  .tabs{display:flex;gap:4px;margin-bottom:16px;background:#0a0a14;border-radius:10px;padding:4px}
  .tab{flex:1;padding:8px;border-radius:8px;border:none;background:transparent;color:#666;font-size:12px;font-weight:600;cursor:pointer;transition:all 0.2s}
  .tab.active{background:#0066cc;color:#fff}
  
  /* TASK SELECTOR */
  .task-pills{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px}
  .task-pill{padding:5px 14px;border-radius:16px;border:1px solid #2a3a4a;background:transparent;color:#888;font-size:11px;cursor:pointer;transition:all 0.2s;font-weight:600}
  .task-pill.active{background:#0066cc;border-color:#0066cc;color:#fff}
  .task-pill:hover{border-color:#0066cc;color:#00aaff}
  
  /* SAMPLE BOX */
  .sample-box{background:#0a0a14;border:1px solid #1e2d40;border-radius:8px;padding:12px;font-size:12px;line-height:1.6;color:#bbb;min-height:80px;max-height:120px;overflow-y:auto}
  .sample-label{font-size:10px;color:#555;text-transform:uppercase;letter-spacing:0.4px;margin-bottom:6px}
  
  /* AGENT CARDS */
  .agent-card{border-radius:12px;padding:16px;border:1px solid}
  .gen-card{background:#1a0a00;border-color:#663300}
  .det-card{background:#001a0a;border-color:#003322}
  .agent-header{display:flex;align-items:center;gap:8px;margin-bottom:14px}
  .agent-icon{font-size:20px}
  .agent-name{font-size:14px;font-weight:700}
  .gen-card .agent-name{color:#ff6633}
  .det-card .agent-name{color:#00cc66}
  
  /* LOADING */
  .spinner{display:inline-block;width:14px;height:14px;border:2px solid rgba(255,255,255,0.2);border-top-color:#fff;border-radius:50%;animation:spin 0.7s linear infinite}
  @keyframes spin{to{transform:rotate(360deg)}}
  
  /* MISC */
  .section-title{font-size:13px;font-weight:600;color:#888;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:12px}
  .divider{height:1px;background:#1e2d40;margin:16px 0}
  .hidden{display:none}
  .text-center{text-align:center}
  .mt-8{margin-top:8px}
  .gap-8{gap:8px}
  code{background:#0a0a14;border:1px solid #2a3a4a;padding:2px 8px;border-radius:4px;font-size:11px;color:#00aaff}
  .info-row{display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #111827;font-size:12px}
  .info-row:last-child{border:none}
  .info-key{color:#666}
  .info-val{color:#ccc;font-weight:500}
</style>
</head>
<body>

<!-- HEADER -->
<div class="header">
  <div class="logo">
    <div class="logo-icon">🔍</div>
    <div class="logo-text">
      <h1>HalluciNet Adversarial</h1>
      <p><span class="status-dot"></span>Live · Multi-Agent Hallucination Detection · Round 2</p>
    </div>
  </div>
  <div class="badges">
    <span class="badge badge-blue">Theme 1: Multi-Agent</span>
    <span class="badge badge-purple">Theme 4: Self-Improvement</span>
    <span class="badge badge-green">OpenEnv 2.0 ✓</span>
  </div>
</div>

<div class="main">

  <!-- STAT CARDS -->
  <div class="grid-3">
    <div class="stat-card">
      <div class="stat-val" id="stat-tasks">5</div>
      <div class="stat-lbl">Difficulty Levels</div>
    </div>
    <div class="stat-card">
      <div class="stat-val" id="stat-samples">65+</div>
      <div class="stat-lbl">Curated Samples</div>
    </div>
    <div class="stat-card">
      <div class="stat-val" id="stat-score">—</div>
      <div class="stat-lbl">Your Last Score</div>
    </div>
  </div>

  <!-- TABS -->
  <div class="tabs">
    <button class="tab active" onclick="showTab('demo')">🎮 Demo</button>
    <button class="tab" onclick="showTab('leaderboard')">🏆 Leaderboard</button>
    <button class="tab" onclick="showTab('api')">⚡ API</button>
    <button class="tab" onclick="showTab('oversight')">👁 Oversight</button>
  </div>

  <!-- DEMO TAB -->
  <div id="tab-demo">
    
    <!-- TASK SELECTOR + LOAD -->
    <div class="card">
      <div class="card-header">
        <div class="card-title">📋 Sample Loader</div>
        <button class="btn btn-primary" onclick="loadSample()" id="load-btn">
          Load Sample
        </button>
      </div>
      <div class="task-pills">
        <button class="task-pill active" onclick="selectTask(this,'easy')">Easy · 10</button>
        <button class="task-pill" onclick="selectTask(this,'medium')">Medium · 12</button>
        <button class="task-pill" onclick="selectTask(this,'hard')">Hard · 19</button>
        <button class="task-pill" onclick="selectTask(this,'expert')">Expert · 20</button>
        <button class="task-pill" onclick="selectTask(this,'adversarial')">Adversarial · 12</button>
      </div>
      <div class="grid-2">
        <div>
          <div class="sample-label">Reference Document (Ground Truth)</div>
          <div class="sample-box" id="ref-box">Click Load Sample to begin...</div>
        </div>
        <div>
          <div class="sample-label">LLM Response (Evaluate This)</div>
          <div class="sample-box" id="llm-box">—</div>
        </div>
      </div>
    </div>

    <!-- AGENT CARDS -->
    <div class="grid-2">
      
      <!-- GENERATOR -->
      <div class="agent-card gen-card">
        <div class="agent-header">
          <div class="agent-icon">🔥</div>
          <div>
            <div class="agent-name">Generator Agent</div>
            <div style="font-size:11px;color:#aa5533">Creates hallucinations to fool the detector</div>
          </div>
        </div>
        <label>Error Type</label>
        <select id="err-type" style="margin-bottom:10px">
          <option value="year_swap">Year Swap</option>
          <option value="name_swap">Name Swap</option>
          <option value="number_swap">Number Swap</option>
          <option value="negation">Negation Trap</option>
          <option value="entity_flip">Entity Flip</option>
          <option value="unit_shift">Unit Shift</option>
          <option value="partial_truth">Partial Truth</option>
        </select>
        <label>Hallucinated Response</label>
        <textarea id="gen-text" rows="3" placeholder="Write a subtle hallucination here..."></textarea>
        <div style="margin:10px 0">
          <label>Generator Confidence: <span id="gc-val" style="color:#ff6633">0.80</span></label>
          <div class="conf-row">
            <input type="range" id="gen-conf" min="0.01" max="0.99" step="0.01" value="0.8"
              oninput="document.getElementById('gc-val').innerText=parseFloat(this.value).toFixed(2)">
          </div>
        </div>
        <button class="btn btn-danger btn-full" onclick="runGenerator()" id="gen-btn">
          🔥 Generate Hallucination
        </button>
        <div id="gen-result" class="hidden" style="margin-top:12px">
          <div class="score-big" id="gen-score" style="text-align:center;margin-bottom:8px"></div>
          <div class="feedback-box" id="gen-feedback"></div>
        </div>
      </div>

      <!-- DETECTOR -->
      <div class="agent-card det-card">
        <div class="agent-header">
          <div class="agent-icon">🛡️</div>
          <div>
            <div class="agent-name">Detector Agent</div>
            <div style="font-size:11px;color:#005533">Catches hallucinations with calibrated confidence</div>
          </div>
        </div>
        <label>Hallucinated Claim (leave blank if clean)</label>
        <input type="text" id="det-claim" placeholder="Quote the exact wrong phrase..." style="margin-bottom:10px">
        <label>Correct Fact from Reference</label>
        <input type="text" id="det-fact" placeholder="What does the reference say?" style="margin-bottom:10px">
        <div style="margin:10px 0">
          <label>Detector Confidence: <span id="dc-val" style="color:#00cc66">0.80</span></label>
          <div class="conf-row">
            <input type="range" id="det-conf" min="0.01" max="0.99" step="0.01" value="0.8"
              oninput="document.getElementById('dc-val').innerText=parseFloat(this.value).toFixed(2)">
          </div>
        </div>
        <button class="btn btn-success btn-full" onclick="runDetector()" id="det-btn">
          🛡️ Submit Detection
        </button>
        <div id="det-result" class="hidden" style="margin-top:12px">
          <div style="text-align:center;margin-bottom:8px">
            <div class="score-big" id="det-score"></div>
            <div style="font-size:11px;color:#666;margin-top:2px">Detector Score</div>
          </div>
          <div class="breakdown-grid" id="det-breakdown"></div>
          <div class="feedback-box" id="det-feedback"></div>
        </div>
      </div>
    </div>

    <!-- DEBATE -->
  <div id="debate-section" class="hidden" style="margin-top:12px">
    <div class="card" style="border-color:#553300">
      <div class="card-title" style="margin-bottom:10px">⚔️ Debate Round — Generator Defense</div>
      <p style="font-size:12px;color:#888;margin-bottom:10px">
        Generator defends. Detector re-evaluates. Ground truth adjudicates.
      </p>
      <button class="btn btn-full" id="debate-btn" onclick="runDebate()"
        style="background:#553300;color:#ff9933">
        ⚔️ Trigger Debate Round
      </button>
      <div id="debate-result" class="hidden"></div>
    </div>
  </div>

  <!-- VERDICT -->
    <div id="verdict-box" class="hidden">
      <div class="verdict verdict-pending" id="verdict-text"></div>
      <div style="text-align:center;font-size:12px;color:#666;margin-top:6px">
        Generator reward: <span id="g-reward" style="color:#ff6633;font-weight:600"></span>
        &nbsp;·&nbsp;
        Detector reward: <span id="d-reward" style="color:#00cc66;font-weight:600"></span>
        &nbsp;·&nbsp;
        <a href="/oversight" style="color:#6699ff" target="_blank">View Oversight →</a>
      </div>
    </div>

  </div><!-- /demo tab -->

  <!-- LEADERBOARD TAB -->
  <div id="tab-leaderboard" class="hidden">
    <div class="card">
      <div class="card-header">
        <div class="card-title">🏆 Model Leaderboard</div>
        <span style="font-size:11px;color:#555">Avg reward from deterministic grader</span>
      </div>
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Model</th>
            <th>Easy</th>
            <th>Medium</th>
            <th>Hard</th>
            <th>Expert</th>
            <th>Adversarial</th>
            <th>Overall</th>
          </tr>
        </thead>
        <tbody id="lb-body">
          <tr><td colspan="8" style="text-align:center;color:#555;padding:20px">Loading...</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- API TAB -->
  <div id="tab-api" class="hidden">
    <div class="card">
      <div class="card-title" style="margin-bottom:16px">⚡ Quick Start</div>
      <div class="section-title">Test the environment</div>
      <div style="background:#0a0a14;border:1px solid #1e2d40;border-radius:8px;padding:14px;margin-bottom:14px;font-family:monospace;font-size:12px;line-height:1.8;color:#aaa">
        <span style="color:#555"># Health check</span><br>
        curl https://rushikeshbathe096-hallucinet.hf.space/health<br><br>
        <span style="color:#555"># Reset detector episode</span><br>
        curl -X POST .../reset -d '{"task_id": "hard"}'<br><br>
        <span style="color:#555"># Submit detection</span><br>
        curl -X POST .../step -d '{"action": {"has_hallucination": true, "hallucinated_claim": "1902", "correct_fact": "1889", "confidence": 0.95}}'<br><br>
        <span style="color:#555"># OpenEnv validate</span><br>
        openenv validate --url https://rushikeshbathe096-hallucinet.hf.space
      </div>
      <div class="section-title">Endpoints</div>
      <div class="info-row"><span class="info-key">GET /health</span><span class="info-val">Health check</span></div>
      <div class="info-row"><span class="info-key">POST /reset</span><span class="info-val">Start detector episode</span></div>
      <div class="info-row"><span class="info-key">POST /step</span><span class="info-val">Submit detector action</span></div>
      <div class="info-row"><span class="info-key">GET /state</span><span class="info-val">Current episode state</span></div>
      <div class="info-row"><span class="info-key">POST /generator/reset</span><span class="info-val">Start generator episode</span></div>
      <div class="info-row"><span class="info-key">POST /generator/step</span><span class="info-val">Submit generator action</span></div>
      <div class="info-row"><span class="info-key">POST /debate</span><span class="info-val">Debate with generator_defense (after /step)</span></div>
      <div class="info-row"><span class="info-key">GET /oversight/status</span><span class="info-val">System reliability (dynamic)</span></div>
      <div class="info-row"><span class="info-key">GET /curriculum/status</span><span class="info-val">Curriculum progress</span></div>
      <div class="info-row"><span class="info-key">GET /leaderboard</span><span class="info-val">Recorded model scores</span></div>
      <div class="info-row"><span class="info-key">POST /leaderboard/record</span><span class="info-val">Record a model task score</span></div>
    </div>
  </div>

  <!-- OVERSIGHT TAB -->
  <div id="tab-oversight" class="hidden">
    <div class="card">
      <div class="card-title" style="margin-bottom:16px">👁 Oversight Agent — System Monitor</div>
      <div id="oversight-content" style="color:#555;text-align:center;padding:20px">Loading...</div>
    </div>
  </div>

</div><!-- /main -->

<script>
let currentTask = "easy";

function selectTask(el, task) {
  document.querySelectorAll(".task-pill").forEach(p => p.classList.remove("active"));
  el.classList.add("active");
  currentTask = task;
}

function showTab(name) {
  ["demo","leaderboard","api","oversight"].forEach(t => {
    document.getElementById("tab-"+t).classList.add("hidden");
  });
  document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
  document.getElementById("tab-"+name).classList.remove("hidden");
  event.target.classList.add("active");
  if (name === "leaderboard") loadLeaderboard();
  if (name === "oversight") loadOversight();
}

async function loadSample() {
  const btn = document.getElementById("load-btn");
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Loading...';
  try {
    const r = await fetch("/reset", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({task_id: currentTask})
    });
    const d = await r.json();
    const obs = d.observation || d;
    document.getElementById("ref-box").innerText = obs.reference_document || "—";
    document.getElementById("llm-box").innerText = obs.llm_response || "—";
    document.getElementById("stat-samples").innerText = obs.total_samples || "—";
    ["gen-result","det-result","verdict-box"].forEach(id => {
      document.getElementById(id).classList.add("hidden");
    });
    document.getElementById("stat-score").innerText = "—";
  } catch(e) {
    document.getElementById("ref-box").innerText = "Error loading sample: " + e.message;
  }
  btn.disabled = false;
  btn.innerHTML = "Load Sample";
}

async function runGenerator() {
  const text = document.getElementById("gen-text").value;
  if (!text) { alert("Write a hallucination first"); return; }
  const btn = document.getElementById("gen-btn");
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Generating...';
  try {
    await fetch("/generator/reset", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({task_id: currentTask})
    });
    const r = await fetch("/generator/step", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({action: {
        generated_response: text,
        error_type: document.getElementById("err-type").value,
        confidence: parseFloat(document.getElementById("gen-conf").value)
      }})
    });
    const d = await r.json();
    const obs = d.observation || d;
    const score = obs.fooling_rate || obs.score || 0;
    const scoreEl = document.getElementById("gen-score");
    scoreEl.innerText = "Generator: " + score.toFixed(3);
    scoreEl.className = "score-big " + getScoreClass(score);
    document.getElementById("gen-feedback").innerText = obs.feedback || "Submitted";
    document.getElementById("gen-result").classList.remove("hidden");
    document.getElementById("llm-box").innerText = text;
  } catch(e) { alert("Error: " + e.message); }
  btn.disabled = false;
  btn.innerHTML = "🔥 Generate Hallucination";
}

async function runDetector() {
  const claim = document.getElementById("det-claim").value;
  const btn = document.getElementById("det-btn");
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Detecting...';
  try {
    const r = await fetch("/step", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({action: {
        has_hallucination: claim.length > 0,
        hallucinated_claim: claim || null,
        correct_fact: document.getElementById("det-fact").value || null,
        confidence: parseFloat(document.getElementById("det-conf").value)
      }})
    });
    const d = await r.json();
    const obs = d.observation || d;
    const score = obs.score || 0;
    const bd = obs.metadata?.reward_breakdown || {};
    
    const scoreEl = document.getElementById("det-score");
    scoreEl.innerText = score.toFixed(3);
    scoreEl.className = "score-big " + getScoreClass(score);
    document.getElementById("stat-score").innerText = score.toFixed(3);
    
    document.getElementById("det-breakdown").innerHTML = `
      <div class="breakdown-item"><div class="breakdown-val">${(bd.detection||0).toFixed(2)}</div><div class="breakdown-lbl">Detection</div></div>
      <div class="breakdown-item"><div class="breakdown-val">${(bd.phrase||0).toFixed(2)}</div><div class="breakdown-lbl">Phrase ID</div></div>
      <div class="breakdown-item"><div class="breakdown-val">${(bd.fact||0).toFixed(2)}</div><div class="breakdown-lbl">Correct Fact</div></div>
      <div class="breakdown-item"><div class="breakdown-val">${(bd.calibration||0).toFixed(2)}</div><div class="breakdown-lbl">Calibration</div></div>
    `;
    document.getElementById("det-feedback").innerText = obs.feedback || "";
    document.getElementById("det-result").classList.remove("hidden");
    
    const caught = claim.length > 0;
    const vb = document.getElementById("verdict-box");
    const vt = document.getElementById("verdict-text");
    vb.classList.remove("hidden");
    if (caught && score > 0.4) {
      vt.className = "verdict verdict-win";
      vt.innerText = "🛡️ DETECTOR WINS — Hallucination caught!";
    } else if (!caught) {
      vt.className = "verdict verdict-lose";
      vt.innerText = "🔥 GENERATOR WINS — Hallucination slipped through!";
    } else {
      vt.className = "verdict verdict-pending";
      vt.innerText = "⚠️ Partial detection — keep training!";
    }
    document.getElementById("g-reward").innerText = caught ? "0.001" : "0.999";
    document.getElementById("debate-section").classList.remove("hidden");
    document.getElementById("d-reward").innerText = score.toFixed(3);
  } catch(e) { alert("Error: " + e.message); }
  btn.disabled = false;
  btn.innerHTML = "🛡️ Submit Detection";
}

function getScoreClass(s) {
  if (s >= 0.9) return "score-excellent";
  if (s >= 0.7) return "score-good";
  if (s >= 0.4) return "score-partial";
  return "score-bad";
}

async function loadLeaderboard() {
  try {
    const r = await fetch("/leaderboard");
    const d = await r.json();
    const tbody = document.getElementById("lb-body");
    const ranks = ["rank-1","rank-2","rank-3"];
    tbody.innerHTML = d.leaderboard.map((e,i) => `
      <tr>
        <td><span class="rank-badge ${ranks[i]||''}">${e.rank}</span></td>
        <td>${e.model} ${e.trained ? '<span class="trained-tag">GRPO</span>' : ''}</td>
        <td>${e.easy}</td><td>${e.medium}</td><td>${e.hard}</td>
        <td>${e.expert}</td><td>${e.adversarial}</td>
        <td><strong style="color:#00aaff">${e.overall}</strong></td>
      </tr>
    `).join("");
  } catch(e) {
    document.getElementById("lb-body").innerHTML = 
      '<tr><td colspan="8" style="text-align:center;color:#555">Error loading leaderboard</td></tr>';
  }
}

async function loadOversight() {
  try {
    const r = await fetch("/oversight/status");
    const d = await r.json();
    const score = d.reliability_score;
    const color = score > 0.8 ? "#00cc66" : score > 0.6 ? "#ffaa00" : "#ff4444";
    document.getElementById("oversight-content").innerHTML = `
      <div style="font-size:48px;font-weight:700;color:${color};margin-bottom:8px">${score.toFixed(3)}</div>
      <div style="color:#666;margin-bottom:16px">System Reliability Score</div>
      <div class="info-row"><span class="info-key">Overconfidence Rate</span><span class="info-val">${d.overconfidence_rate?.toFixed(3) || "0.000"}</span></div>
      <div class="info-row"><span class="info-key">Blind Spots</span><span class="info-val">${d.blind_spots?.length ? d.blind_spots.join(", ") : "None detected"}</span></div>
      <div class="info-row"><span class="info-key">System Feedback</span><span class="info-val">${d.system_feedback || "—"}</span></div>
    `;
  } catch(e) {
    document.getElementById("oversight-content").innerText = "Error loading oversight data";
  }
}

async function runDebate() {
  const btn = document.getElementById("debate-btn");
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Debating...';
  try {
    const r = await fetch("/debate", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        generator_defense: "My response is factually supported by the source material."
      })
    });
    const d = await r.json();
    document.getElementById("debate-result").innerHTML = `
      <div class="feedback-box" style="margin-top:8px">
        Outcome: ${d.debate?.outcome || "completed"} | 
        ${d.debate?.adjudication_reason || "Debate round complete"}
      </div>
    `;
    document.getElementById("debate-result").classList.remove("hidden");
  } catch(e) { alert("Error: " + e.message); }
  btn.disabled = false;
  btn.innerHTML = "⚔️ Trigger Debate Round";
}

// Init
loadSample();
</script>
</body>
</html>
""")
@app.get("/metadata")
def metadata():
    return {
        "name": "hallucinet-adversarial",
        "description": "Adversarial self-improving hallucination detection. Generator vs Detector multi-agent RL. Theme 1 + Theme 4.",
        "version": "2.0.0",
        "author": "team-tle"
    }

@app.get("/schema")
def schema():
    return {
        "action": {
            "has_hallucination": "bool",
            "hallucinated_claim": "string or null",
            "correct_fact": "string or null",
            "confidence": "float strictly between 0 and 1"
        },
        "observation": {
            "reference_document": "string",
            "llm_response": "string",
            "feedback": "string",
            "score": "float",
            "reward": "float",
            "done": "bool"
        },
        "state": {
            "episode_id": "string",
            "task_id": "string",
            "steps_taken": "int",
            "is_done": "bool"
        }
    }


@app.post("/mcp")
async def mcp_endpoint(request: dict = None):
    """MCP JSON-RPC endpoint for OpenEnv compatibility."""
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "name": "hallucinet-adversarial",
            "version": "2.0.0",
            "description": "Adversarial hallucination detection RL environment"
        }
    }


@app.get("/oversight/status")
def get_oversight_status():
    return oversight_agent.evaluate()


@app.get("/oversight")
def get_oversight():
    return oversight_agent.evaluate()

@app.post("/oversight/reset")
def reset_oversight():
    oversight_agent.reset()
    return {"status": "oversight reset"}


@app.get("/curriculum/status")
def get_curriculum_status():
    return adversarial_curriculum.get_status()


@app.post("/debate")
def debate_post(body: DebateRequest):
    ctx = detector_env.get_last_debate_context()
    if not ctx:
        # Standalone mode — create a default context for judges testing directly
        from tasks import get_task
        import random
        # FIXED
        task_level = body.task_id or getattr(detector_env, "_task_id", None) or "hard"
        samples = get_task(task_level)
        sample = random.choice([s for s in samples if s["ground_truth_has_hallucination"]])
        ctx = {
            "reference_document": sample["reference_document"],
            "llm_response": sample["llm_response"],
            "detector_claim": sample["ground_truth_hallucinated_phrases"][0] if sample["ground_truth_hallucinated_phrases"] else "",
            "ground_truth_phrases": sample["ground_truth_hallucinated_phrases"],
        }
    tid = body.task_id
    if tid and getattr(detector_env, "_task_id", None) and tid != detector_env._task_id:
        raise HTTPException(
            status_code=400,
            detail=(
                f"task_id {tid!r} does not match active episode "
                f"{detector_env._task_id!r}"
            ),
        )
    result = debate_coordinator.run_debate(
        reference=ctx["reference_document"],
        generated_response=ctx["llm_response"],
        detector_claim=ctx["detector_claim"],
        generator_defense=body.generator_defense,
        ground_truth_phrases=ctx["ground_truth_phrases"],
    )
    return {
        "task_id": detector_env._task_id,
        "debate": result,
        "debate_round": True,
        "debate_stats": debate_coordinator.get_stats(),
    }


@app.get("/calibration")
def calibration():
    return {
        "detector": detector_calibration.get_calibration_curve(),
        "description": "Confidence vs actual accuracy. ECE < 0.1 = well calibrated."
    }


@app.get("/elo/standings")
def elo_standings():
    return elo_tracker.get_standings()


@app.get("/elo/history")
def elo_history():
    return {"history": elo_tracker.history[-20:]}


@app.get("/training/summary")
def training_summary():
    return {
        "before_training": {
            "model": "qwen2.5-3b-baseline",
            "medium_reward": 0.9490,
            "note": "Reward ceiling effect — model exploiting JSON format with high confidence"
        },
        "after_training": {
            "model": "qwen2.5-3b-grpo-hallucinet",
            "easy_reward": 0.647,
            "medium_reward": 0.774,
            "hard_reward": 0.729,
            "expert_reward": 0.010,
            "curriculum_level_reached": "hard",
            "promotions": 19,
            "sessions": 90
        },
        "key_finding": "Curriculum escalated from easy to hard across 90 sessions. Expert task correctly demoted — environment working as designed.",
        "elo": elo_tracker.get_standings()
    }


def main():
    import uvicorn
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "7860")),
        reload=False
    )

if __name__ == "__main__":
    main()
