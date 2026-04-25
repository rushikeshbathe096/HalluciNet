import os
import sys

# Add project root to path so all imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional

from server.environment import HallucinationEnvironment
from server.generator_environment import GeneratorEnvironment
from models import HallucinationAction, GeneratorAction

app = FastAPI(title="HalluciNet Adversarial - Round 2")

detector_env = HallucinationEnvironment()
generator_env = GeneratorEnvironment()

class ResetRequest(BaseModel):
    task_id: Optional[str] = "easy"

class DetectorStepRequest(BaseModel):
    action: HallucinationAction

class GeneratorStepRequest(BaseModel):
    action: GeneratorAction

@app.get("/health")
def health():
    return {"status": "healthy", "mode": "adversarial", "version": "2.0"}

@app.post("/reset")
def reset(body: ResetRequest = ResetRequest()):
    obs = detector_env.reset(task_id=body.task_id)
    return {"observation": obs.model_dump(), "reward": None, "done": False}

@app.post("/step")
def step(body: DetectorStepRequest):
    obs = detector_env.step(body.action)
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
        "tasks": ["easy", "medium", "hard", "expert"],
        "themes": ["Theme 1: Multi-Agent", "Theme 4: Self-Improvement"]
    }

@app.get("/leaderboard")
def leaderboard():
    return {"leaderboard": [], "description": "Error types ranked by fool rate"}

@app.get("/stats")
def stats():
    return {"total_detector_episodes": 0, "total_generator_episodes": 0}

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
<html>
<head>
    <title>HalluciNet Adversarial — Round 2</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, Arial; background: #0f1117; color: #e0e0e0; }
        .header { background: linear-gradient(135deg,#1a1a2e,#16213e); padding: 30px; text-align: center; border-bottom: 2px solid #0066cc; }
        h1 { color: #00aaff; font-size: 26px; }
        .sub { color: #888; margin-top: 8px; font-size: 14px; }
        .tags { margin-top: 12px; }
        .tag { display: inline-block; background: rgba(0,102,204,0.2); color: #00aaff; border: 1px solid #0066cc; padding: 3px 10px; border-radius: 20px; font-size: 11px; margin: 3px; }
        .container { max-width: 960px; margin: 24px auto; padding: 0 20px; }
        .card { background: #1a1a2e; border: 1px solid #333; border-radius: 12px; padding: 20px; margin: 16px 0; }
        .card h2 { color: #00aaff; margin-bottom: 12px; font-size: 16px; }
        .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
        textarea { width: 100%; height: 90px; padding: 10px; background: #0f1117; border: 1px solid #444; border-radius: 8px; color: #e0e0e0; font-size: 13px; }
        input[type=text] { width: 100%; padding: 9px; background: #0f1117; border: 1px solid #444; border-radius: 8px; color: #e0e0e0; margin: 5px 0; }
        select { padding: 9px; background: #0f1117; border: 1px solid #444; border-radius: 8px; color: #e0e0e0; }
        button { background: #0066cc; color: white; padding: 9px 18px; border: none; border-radius: 8px; cursor: pointer; font-size: 14px; margin: 4px 0; }
        button:hover { background: #0052a3; }
        button.red { background: #cc3300; } button.green { background: #006633; }
        label { color: #888; font-size: 12px; display: block; margin-top: 10px; margin-bottom: 3px; }
        .score { font-size: 26px; font-weight: bold; color: #00aaff; margin: 8px 0; }
        .breakdown { display: grid; grid-template-columns: repeat(4,1fr); gap: 8px; margin: 10px 0; }
        .metric { background: #0f1117; border: 1px solid #333; padding: 10px; border-radius: 8px; text-align: center; }
        .mv { font-size: 18px; font-weight: bold; color: #00aaff; }
        .ml { font-size: 10px; color: #666; margin-top: 3px; }
        .feedback { background: #0f1117; border-left: 3px solid #0066cc; padding: 10px; border-radius: 4px; font-size: 12px; margin-top: 10px; }
        .winner { font-size: 18px; font-weight: bold; padding: 10px; border-radius: 8px; text-align: center; margin: 8px 0; }
        .gw { background: rgba(255,80,0,0.15); color: #ff6400; border: 1px solid #ff6400; }
        .dw { background: rgba(0,180,80,0.15); color: #00c864; border: 1px solid #00c864; }
        .row { display: flex; align-items: center; gap: 10px; }
        .row input { flex: 1; }
    </style>
</head>
<body>
<div class="header">
    <h1>🔍 HalluciNet Adversarial</h1>
    <div class="sub">Multi-Agent Self-Improving Hallucination Detection — Round 2</div>
    <div class="tags">
        <span class="tag">Theme 1: Multi-Agent</span>
        <span class="tag">Theme 4: Self-Improvement</span>
        <span class="tag">OpenEnv 2.0</span>
        <span class="tag">Adversarial Self-Play</span>
        <span class="tag">4 Difficulty Levels</span>
    </div>
</div>
<div class="container">
    <div class="card">
        <h2>🎮 Adversarial Demo — Generator vs Detector</h2>
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px">
            <select id="task">
                <option value="easy">Easy</option>
                <option value="medium">Medium</option>
                <option value="hard" selected>Hard</option>
                <option value="expert">Expert</option>
            </select>
            <button onclick="loadSample()">Load Sample</button>
        </div>
        <div class="grid2">
            <div>
                <label>Reference Document (ground truth):</label>
                <textarea id="ref" placeholder="Click Load Sample..."></textarea>
            </div>
            <div>
                <label>LLM Response (detector evaluates this):</label>
                <textarea id="llm_resp" placeholder="Will appear after loading..."></textarea>
            </div>
        </div>
        <div class="grid2" style="margin-top:14px">
            <div class="card" style="margin:0;border-color:#cc3300">
                <h2 style="color:#ff6400">🔥 Generator Agent</h2>
                <label>Error Type:</label>
                <select id="err_type">
                    <option value="year_swap">Year Swap</option>
                    <option value="name_swap">Name Swap</option>
                    <option value="number_swap">Number Swap</option>
                    <option value="negation">Negation Trap</option>
                    <option value="entity_flip">Entity Flip</option>
                    <option value="unit_shift">Unit Shift</option>
                    <option value="partial_truth">Partial Truth</option>
                </select>
                <label>Your Hallucination:</label>
                <textarea id="gen_text" placeholder="Write a subtle hallucination..."></textarea>
                <label>Confidence: <span id="gc_v">0.80</span></label>
                <div class="row"><input type="range" id="gen_conf" min="0.01" max="0.99" step="0.01" value="0.8" oninput="document.getElementById('gc_v').innerText=parseFloat(this.value).toFixed(2)"></div>
                <button class="red" onclick="runGenerator()" style="margin-top:10px;width:100%">Generate Hallucination</button>
                <div id="gen_result" style="display:none;margin-top:10px">
                    <div class="score" id="gen_score"></div>
                    <div class="feedback" id="gen_fb"></div>
                </div>
            </div>
            <div class="card" style="margin:0;border-color:#006633">
                <h2 style="color:#00c864">🛡️ Detector Agent</h2>
                <label>Hallucinated Claim (blank if clean):</label>
                <input type="text" id="det_claim" placeholder="Quote the wrong phrase...">
                <label>Correct Fact:</label>
                <input type="text" id="det_fact" placeholder="What does reference say?">
                <label>Confidence: <span id="dc_v">0.80</span></label>
                <div class="row"><input type="range" id="det_conf" min="0.01" max="0.99" step="0.01" value="0.8" oninput="document.getElementById('dc_v').innerText=parseFloat(this.value).toFixed(2)"></div>
                <button class="green" onclick="runDetector()" style="margin-top:10px;width:100%">Submit Detection</button>
                <div id="det_result" style="display:none;margin-top:10px">
                    <div class="score" id="det_score"></div>
                    <div class="breakdown" id="det_bd"></div>
                    <div class="feedback" id="det_fb"></div>
                </div>
            </div>
        </div>
        <div id="battle" style="display:none;margin-top:14px;text-align:center">
            <div class="winner" id="winner"></div>
            <p style="color:#666;font-size:13px;margin-top:6px">
                Generator reward: <span id="gr" style="color:#ff6400"></span> &nbsp;|&nbsp;
                Detector reward: <span id="dr" style="color:#00c864"></span>
            </p>
        </div>
    </div>
    <div class="card">
        <h2>📊 Environment Info</h2>
        <pre id="env_info" style="font-size:12px;color:#888;overflow:auto;max-height:200px">Loading...</pre>
    </div>
</div>
<script>
async function loadSample() {
    const r = await fetch('/reset',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({task_id:document.getElementById('task').value})});
    const d = await r.json();
    const obs = d.observation||d;
    document.getElementById('ref').value = obs.reference_document||'';
    document.getElementById('llm_resp').value = obs.llm_response||'';
    ['gen_result','det_result','battle'].forEach(id=>document.getElementById(id).style.display='none');
}
async function runGenerator() {
    const genText = document.getElementById('gen_text').value;
    if (!genText) { alert('Write a hallucination first'); return; }
    await fetch('/generator/reset',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({task_id:document.getElementById('task').value})});
    const r = await fetch('/generator/step',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:{
        generated_response: genText,
        error_type: document.getElementById('err_type').value,
        confidence: parseFloat(document.getElementById('gen_conf').value)
    }})});
    const d = await r.json();
    const obs = d.observation||d;
    document.getElementById('llm_resp').value = genText;
    document.getElementById('gen_result').style.display='block';
    document.getElementById('gen_score').innerText = 'Generator Score: '+(obs.fooling_rate||0).toFixed(3);
    document.getElementById('gen_fb').innerText = obs.feedback||'';
}
async function runDetector() {
    const claim = document.getElementById('det_claim').value;
    const r = await fetch('/step',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:{
        has_hallucination: claim.length>0,
        hallucinated_claim: claim||null,
        correct_fact: document.getElementById('det_fact').value||null,
        confidence: parseFloat(document.getElementById('det_conf').value)
    }})});
    const d = await r.json();
    const obs = d.observation||d;
    const bd = obs.metadata?.reward_breakdown||{};
    const caught = claim.length>0;
    document.getElementById('det_result').style.display='block';
    document.getElementById('det_score').innerText = 'Detector Score: '+(obs.score||0).toFixed(3);
    document.getElementById('det_bd').innerHTML = `
        <div class="metric"><div class="mv">${(bd.detection||0).toFixed(2)}</div><div class="ml">Detection</div></div>
        <div class="metric"><div class="mv">${(bd.phrase||0).toFixed(2)}</div><div class="ml">Phrase ID</div></div>
        <div class="metric"><div class="mv">${(bd.fact||0).toFixed(2)}</div><div class="ml">Correct Fact</div></div>
        <div class="metric"><div class="mv">${(bd.calibration||0).toFixed(2)}</div><div class="ml">Calibration</div></div>`;
    document.getElementById('det_fb').innerText = obs.feedback||'';
    document.getElementById('battle').style.display='block';
    const el = document.getElementById('winner');
    if(caught){el.className='winner dw';el.innerText='🛡️ DETECTOR WINS — Hallucination caught!';}
    else{el.className='winner gw';el.innerText='🔥 GENERATOR WINS — Hallucination slipped through!';}
    document.getElementById('gr').innerText = (caught?'0.001':'0.999');
    document.getElementById('dr').innerText = (obs.score||0).toFixed(3);
}
fetch('/adversarial/info').then(r=>r.json()).then(d=>{
    document.getElementById('env_info').innerText = JSON.stringify(d,null,2);
}).catch(()=>{ document.getElementById('env_info').innerText = 'Info not available'; });
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
