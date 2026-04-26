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
from tasks import TASKS

app = FastAPI(title="HalluciNet Adversarial - Round 2")
DEMO_UI_HTML = '<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n<meta name="viewport" content="width=device-width, initial-scale=1.0">\n<title>HalluciNet Adversarial — Round 2</title>\n<style>\n@import url(\'https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Syne:wght@400;500;600;700;800&display=swap\');\n\n:root {\n  --bg: #06080f;\n  --bg2: #0c1018;\n  --bg3: #111827;\n  --border: rgba(255,255,255,0.07);\n  --border2: rgba(255,255,255,0.12);\n  --text: #e8eaf0;\n  --muted: #6b7280;\n  --accent: #3b82f6;\n  --accent2: #06b6d4;\n  --green: #10b981;\n  --red: #ef4444;\n  --orange: #f97316;\n  --yellow: #f59e0b;\n  --purple: #8b5cf6;\n  --font: \'Syne\', sans-serif;\n  --mono: \'JetBrains Mono\', monospace;\n}\n\n* { box-sizing: border-box; margin: 0; padding: 0; }\n\nbody {\n  font-family: var(--font);\n  background: var(--bg);\n  color: var(--text);\n  min-height: 100vh;\n  overflow-x: hidden;\n}\n\n/* GRID BACKGROUND */\nbody::before {\n  content: \'\';\n  position: fixed;\n  inset: 0;\n  background-image: \n    linear-gradient(rgba(59,130,246,0.03) 1px, transparent 1px),\n    linear-gradient(90deg, rgba(59,130,246,0.03) 1px, transparent 1px);\n  background-size: 40px 40px;\n  pointer-events: none;\n  z-index: 0;\n}\n\n/* HEADER */\n.header {\n  position: relative;\n  z-index: 10;\n  border-bottom: 1px solid var(--border);\n  background: rgba(6,8,15,0.95);\n  backdrop-filter: blur(12px);\n  padding: 0 32px;\n  display: flex;\n  align-items: center;\n  justify-content: space-between;\n  height: 64px;\n}\n\n.logo-area { display: flex; align-items: center; gap: 16px; }\n\n.logo-icon {\n  width: 36px; height: 36px;\n  background: linear-gradient(135deg, var(--accent), var(--accent2));\n  border-radius: 8px;\n  display: flex; align-items: center; justify-content: center;\n  font-size: 18px;\n}\n\n.logo-text h1 {\n  font-size: 18px;\n  font-weight: 700;\n  letter-spacing: -0.5px;\n  background: linear-gradient(135deg, #fff 0%, var(--accent2) 100%);\n  -webkit-background-clip: text;\n  -webkit-text-fill-color: transparent;\n}\n\n.logo-text p { font-size: 11px; color: var(--muted); font-family: var(--mono); margin-top: 1px; }\n\n.header-badges { display: flex; gap: 6px; flex-wrap: wrap; }\n\n.badge {\n  padding: 3px 10px;\n  border-radius: 20px;\n  font-size: 10px;\n  font-weight: 600;\n  font-family: var(--mono);\n  letter-spacing: 0.3px;\n  border: 1px solid;\n}\n\n.badge-blue { background: rgba(59,130,246,0.1); color: #60a5fa; border-color: rgba(59,130,246,0.3); }\n.badge-cyan { background: rgba(6,182,212,0.1); color: #22d3ee; border-color: rgba(6,182,212,0.3); }\n.badge-purple { background: rgba(139,92,246,0.1); color: #a78bfa; border-color: rgba(139,92,246,0.3); }\n.badge-green { background: rgba(16,185,129,0.1); color: #34d399; border-color: rgba(16,185,129,0.3); }\n.badge-orange { background: rgba(249,115,22,0.1); color: #fb923c; border-color: rgba(249,115,22,0.3); }\n\n.pulse { display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: var(--green); margin-right: 5px; animation: pulse 2s infinite; }\n@keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.3;} }\n\n/* MAIN */\n.main { position: relative; z-index: 1; max-width: 1280px; margin: 0 auto; padding: 24px 24px; }\n\n/* HERO BANNER */\n.hero {\n  background: linear-gradient(135deg, rgba(59,130,246,0.08) 0%, rgba(6,182,212,0.05) 50%, rgba(139,92,246,0.08) 100%);\n  border: 1px solid rgba(59,130,246,0.2);\n  border-radius: 16px;\n  padding: 28px 32px;\n  margin-bottom: 20px;\n  display: grid;\n  grid-template-columns: 1fr auto;\n  gap: 24px;\n  align-items: center;\n}\n\n.hero-title {\n  font-size: 13px;\n  font-family: var(--mono);\n  color: var(--accent2);\n  margin-bottom: 8px;\n  letter-spacing: 1px;\n  text-transform: uppercase;\n}\n\n.hero-headline {\n  font-size: 26px;\n  font-weight: 800;\n  line-height: 1.2;\n  letter-spacing: -0.5px;\n  margin-bottom: 10px;\n}\n\n.hero-headline span { color: var(--accent2); }\n\n.hero-desc { font-size: 13px; color: var(--muted); line-height: 1.6; max-width: 560px; }\n\n.hero-stats { display: flex; gap: 20px; }\n\n.hero-stat { text-align: center; }\n.hero-stat-val { font-size: 28px; font-weight: 800; font-family: var(--mono); color: var(--accent2); line-height: 1; }\n.hero-stat-lbl { font-size: 10px; color: var(--muted); margin-top: 4px; letter-spacing: 0.5px; text-transform: uppercase; }\n\n/* METRICS ROW */\n.metrics-row {\n  display: grid;\n  grid-template-columns: repeat(5, 1fr);\n  gap: 10px;\n  margin-bottom: 20px;\n}\n\n.metric-card {\n  background: var(--bg2);\n  border: 1px solid var(--border);\n  border-radius: 10px;\n  padding: 14px 16px;\n  transition: border-color 0.2s;\n}\n\n.metric-card:hover { border-color: var(--border2); }\n.metric-val { font-size: 22px; font-weight: 700; font-family: var(--mono); color: var(--accent); }\n.metric-lbl { font-size: 10px; color: var(--muted); margin-top: 4px; letter-spacing: 0.5px; text-transform: uppercase; }\n\n/* TABS */\n.tabs {\n  display: flex;\n  gap: 2px;\n  margin-bottom: 20px;\n  background: var(--bg2);\n  border: 1px solid var(--border);\n  border-radius: 10px;\n  padding: 4px;\n}\n\n.tab {\n  flex: 1;\n  padding: 8px 12px;\n  border-radius: 7px;\n  border: none;\n  background: transparent;\n  color: var(--muted);\n  font-size: 12px;\n  font-weight: 600;\n  font-family: var(--font);\n  cursor: pointer;\n  transition: all 0.2s;\n  white-space: nowrap;\n}\n\n.tab:hover { color: var(--text); background: rgba(255,255,255,0.04); }\n.tab.active { background: var(--accent); color: white; }\n\n/* CARDS */\n.card {\n  background: var(--bg2);\n  border: 1px solid var(--border);\n  border-radius: 12px;\n  padding: 20px;\n  margin-bottom: 16px;\n}\n\n.card-title {\n  font-size: 13px;\n  font-weight: 600;\n  color: var(--text);\n  margin-bottom: 16px;\n  display: flex;\n  align-items: center;\n  gap: 8px;\n}\n\n.card-title .icon { font-size: 14px; }\n\n/* TUTORIAL SECTION */\n.tutorial-steps {\n  display: grid;\n  grid-template-columns: repeat(4, 1fr);\n  gap: 12px;\n  margin-bottom: 20px;\n}\n\n.tutorial-step {\n  background: var(--bg2);\n  border: 1px solid var(--border);\n  border-radius: 10px;\n  padding: 16px;\n  position: relative;\n  transition: border-color 0.2s, transform 0.2s;\n}\n\n.tutorial-step:hover { border-color: var(--accent); transform: translateY(-2px); }\n\n.step-num {\n  width: 24px; height: 24px;\n  border-radius: 50%;\n  background: var(--accent);\n  color: white;\n  font-size: 11px;\n  font-weight: 700;\n  font-family: var(--mono);\n  display: flex; align-items: center; justify-content: center;\n  margin-bottom: 10px;\n}\n\n.step-title { font-size: 12px; font-weight: 600; margin-bottom: 4px; }\n.step-desc { font-size: 11px; color: var(--muted); line-height: 1.5; }\n\n.step-arrow {\n  position: absolute;\n  right: -18px;\n  top: 50%;\n  transform: translateY(-50%);\n  color: var(--muted);\n  font-size: 16px;\n  z-index: 2;\n}\n\n/* GRID 2 */\n.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }\n\n/* TASK PILLS */\n.task-pills { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 14px; }\n\n.task-pill {\n  padding: 5px 14px;\n  border-radius: 16px;\n  border: 1px solid var(--border2);\n  background: transparent;\n  color: var(--muted);\n  font-size: 11px;\n  cursor: pointer;\n  transition: all 0.2s;\n  font-weight: 600;\n  font-family: var(--font);\n}\n\n.task-pill.active { background: var(--accent); border-color: var(--accent); color: white; }\n.task-pill:hover:not(.active) { border-color: var(--accent); color: var(--accent); }\n\n/* SAMPLE BOX */\n.sample-box {\n  background: var(--bg);\n  border: 1px solid var(--border);\n  border-radius: 8px;\n  padding: 12px;\n  font-size: 12px;\n  line-height: 1.6;\n  color: #9ca3af;\n  min-height: 80px;\n  max-height: 120px;\n  overflow-y: auto;\n  font-family: var(--mono);\n}\n\n.sample-label { font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; font-family: var(--mono); }\n\n/* AGENT CARDS */\n.agent-card { border-radius: 12px; padding: 18px; border: 1px solid; }\n.gen-card { background: rgba(249,115,22,0.04); border-color: rgba(249,115,22,0.2); }\n.det-card { background: rgba(16,185,129,0.04); border-color: rgba(16,185,129,0.2); }\n\n.agent-header { display: flex; align-items: center; gap: 10px; margin-bottom: 14px; }\n.agent-icon { font-size: 20px; }\n.agent-name { font-size: 13px; font-weight: 700; }\n.agent-sub { font-size: 11px; color: var(--muted); margin-top: 2px; }\n.gen-card .agent-name { color: var(--orange); }\n.det-card .agent-name { color: var(--green); }\n\n/* FORM ELEMENTS */\nlabel { display: block; font-size: 10px; color: var(--muted); margin-bottom: 5px; text-transform: uppercase; letter-spacing: 0.5px; font-family: var(--mono); }\n\nselect, input[type=text] {\n  background: var(--bg);\n  border: 1px solid var(--border2);\n  border-radius: 7px;\n  color: var(--text);\n  padding: 8px 11px;\n  font-size: 12px;\n  width: 100%;\n  outline: none;\n  transition: border 0.2s;\n  margin-bottom: 10px;\n  font-family: var(--font);\n}\n\nselect:focus, input[type=text]:focus { border-color: var(--accent); }\n\ntextarea {\n  background: var(--bg);\n  border: 1px solid var(--border2);\n  border-radius: 7px;\n  color: var(--text);\n  padding: 9px 11px;\n  font-size: 12px;\n  width: 100%;\n  resize: vertical;\n  outline: none;\n  line-height: 1.5;\n  transition: border 0.2s;\n  font-family: var(--mono);\n}\n\ntextarea:focus { border-color: var(--accent); }\n\n/* RANGE */\ninput[type=range] { width: 100%; height: 4px; accent-color: var(--accent); margin: 8px 0; }\n\n/* BUTTONS */\n.btn {\n  padding: 9px 18px;\n  border: none;\n  border-radius: 8px;\n  font-size: 12px;\n  font-weight: 600;\n  cursor: pointer;\n  transition: all 0.2s;\n  display: inline-flex;\n  align-items: center;\n  gap: 6px;\n  font-family: var(--font);\n}\n\n.btn-primary { background: var(--accent); color: white; }\n.btn-primary:hover { background: #2563eb; transform: translateY(-1px); }\n.btn-danger { background: rgba(239,68,68,0.15); color: var(--red); border: 1px solid rgba(239,68,68,0.3); }\n.btn-danger:hover { background: rgba(239,68,68,0.25); }\n.btn-success { background: rgba(16,185,129,0.15); color: var(--green); border: 1px solid rgba(16,185,129,0.3); }\n.btn-success:hover { background: rgba(16,185,129,0.25); }\n.btn-debate { background: rgba(249,115,22,0.15); color: var(--orange); border: 1px solid rgba(249,115,22,0.3); }\n.btn-debate:hover { background: rgba(249,115,22,0.25); }\n.btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none !important; }\n.btn-full { width: 100%; justify-content: center; }\n\n/* SCORE */\n.score-big { font-size: 40px; font-weight: 800; font-family: var(--mono); }\n.score-excellent { color: var(--green); }\n.score-good { color: var(--accent); }\n.score-partial { color: var(--yellow); }\n.score-bad { color: var(--red); }\n\n/* BREAKDOWN */\n.breakdown-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 8px; margin: 12px 0; }\n.breakdown-item { background: var(--bg); border: 1px solid var(--border); border-radius: 8px; padding: 10px; text-align: center; }\n.breakdown-val { font-size: 16px; font-weight: 700; color: var(--accent); font-family: var(--mono); }\n.breakdown-lbl { font-size: 9px; color: var(--muted); margin-top: 3px; text-transform: uppercase; letter-spacing: 0.5px; }\n\n/* FEEDBACK */\n.feedback-box { background: var(--bg); border-left: 2px solid var(--accent); padding: 11px 14px; border-radius: 0 6px 6px 0; font-size: 12px; line-height: 1.6; margin-top: 10px; color: #9ca3af; font-family: var(--mono); }\n\n/* VERDICT */\n.verdict { padding: 12px; border-radius: 8px; text-align: center; font-size: 15px; font-weight: 700; margin: 12px 0; }\n.verdict-win { background: rgba(16,185,129,0.1); color: var(--green); border: 1px solid rgba(16,185,129,0.3); }\n.verdict-lose { background: rgba(239,68,68,0.1); color: var(--red); border: 1px solid rgba(239,68,68,0.3); }\n.verdict-pending { background: rgba(59,130,246,0.1); color: var(--accent); border: 1px solid rgba(59,130,246,0.3); }\n\n/* LEADERBOARD */\ntable { width: 100%; border-collapse: collapse; font-size: 12px; }\nth { background: var(--bg); color: var(--muted); padding: 10px 12px; text-align: left; font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px; border-bottom: 1px solid var(--border); font-family: var(--mono); }\ntd { padding: 10px 12px; border-bottom: 1px solid var(--border); color: #d1d5db; }\ntr:hover td { background: rgba(59,130,246,0.04); }\n.rank-badge { display: inline-flex; align-items: center; justify-content: center; width: 22px; height: 22px; border-radius: 50%; font-size: 11px; font-weight: 700; }\n.rank-1 { background: #f59e0b; color: #000; }\n.rank-2 { background: #9ca3af; color: #000; }\n.rank-3 { background: #92400e; color: #fff; }\n.trained-tag { background: rgba(16,185,129,0.15); color: var(--green); padding: 2px 7px; border-radius: 10px; font-size: 10px; font-family: var(--mono); }\n\n/* SPINNER */\n.spinner { display: inline-block; width: 13px; height: 13px; border: 2px solid rgba(255,255,255,0.2); border-top-color: #fff; border-radius: 50%; animation: spin 0.7s linear infinite; }\n@keyframes spin { to { transform: rotate(360deg); } }\n\n/* INFO ROW */\n.info-row { display: flex; justify-content: space-between; padding: 7px 0; border-bottom: 1px solid var(--border); font-size: 12px; }\n.info-row:last-child { border: none; }\n.info-key { color: var(--muted); font-family: var(--mono); }\n.info-val { color: var(--text); font-weight: 500; }\n\n/* SECTION TITLE */\n.section-title { font-size: 10px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 10px; margin-top: 16px; font-family: var(--mono); }\n\n/* HIDDEN */\n.hidden { display: none; }\n\n/* ENDPOINT GRID */\n.endpoint-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }\n.endpoint-item { background: var(--bg); border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; }\n.endpoint-method { font-size: 9px; font-weight: 700; font-family: var(--mono); padding: 2px 6px; border-radius: 4px; margin-right: 6px; }\n.method-get { background: rgba(16,185,129,0.15); color: var(--green); }\n.method-post { background: rgba(249,115,22,0.15); color: var(--orange); }\n.endpoint-path { font-size: 11px; font-family: var(--mono); color: var(--accent2); }\n.endpoint-desc { font-size: 10px; color: var(--muted); margin-top: 4px; }\n\n/* NEW ENDPOINTS HIGHLIGHT */\n.new-tag { font-size: 9px; background: rgba(139,92,246,0.15); color: #a78bfa; padding: 1px 6px; border-radius: 4px; font-family: var(--mono); margin-left: 6px; }\n\n/* WORLD MODEL */\n.wm-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }\n.wm-section { background: var(--bg); border: 1px solid var(--border); border-radius: 10px; padding: 16px; }\n.wm-title { font-size: 11px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 12px; font-family: var(--mono); }\n\n/* ELO DISPLAY */\n.elo-bars { display: flex; gap: 12px; align-items: flex-end; height: 80px; margin: 12px 0; }\n.elo-bar-wrap { flex: 1; display: flex; flex-direction: column; align-items: center; gap: 4px; height: 100%; justify-content: flex-end; }\n.elo-bar { width: 100%; border-radius: 4px 4px 0 0; min-height: 8px; transition: height 0.5s ease; }\n.elo-bar-det { background: linear-gradient(180deg, var(--green), rgba(16,185,129,0.5)); }\n.elo-bar-gen { background: linear-gradient(180deg, var(--orange), rgba(249,115,22,0.5)); }\n.elo-bar-val { font-size: 12px; font-weight: 700; font-family: var(--mono); }\n.elo-bar-lbl { font-size: 10px; color: var(--muted); }\n\n/* TRAINING SUMMARY */\n.training-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }\n.training-col { background: var(--bg); border: 1px solid var(--border); border-radius: 10px; padding: 14px; }\n.training-col-title { font-size: 10px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 10px; font-family: var(--mono); }\n.training-before { border-left: 2px solid var(--red); }\n.training-after { border-left: 2px solid var(--green); }\n\n/* CURRICULUM FLOW */\n.curriculum-flow { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; margin: 12px 0; }\n.curr-level { padding: 4px 12px; border-radius: 20px; font-size: 11px; font-weight: 600; font-family: var(--mono); border: 1px solid; }\n.curr-easy { background: rgba(16,185,129,0.1); color: var(--green); border-color: rgba(16,185,129,0.3); }\n.curr-medium { background: rgba(59,130,246,0.1); color: var(--accent); border-color: rgba(59,130,246,0.3); }\n.curr-hard { background: rgba(249,115,22,0.1); color: var(--orange); border-color: rgba(249,115,22,0.3); }\n.curr-expert { background: rgba(239,68,68,0.1); color: var(--red); border-color: rgba(239,68,68,0.3); }\n.curr-arrow { color: var(--muted); font-size: 14px; }\n\n/* CODE BLOCK */\n.code-block { background: var(--bg); border: 1px solid var(--border); border-radius: 8px; padding: 14px; font-family: var(--mono); font-size: 11px; line-height: 1.8; color: #9ca3af; overflow-x: auto; }\n.code-comment { color: #4b5563; }\n.code-key { color: var(--accent2); }\n.code-val { color: #a78bfa; }\n.code-str { color: #34d399; }\n\n/* PROGRESS BAR */\n.progress-bar { height: 6px; background: var(--bg); border-radius: 3px; overflow: hidden; margin: 6px 0; }\n.progress-fill { height: 100%; border-radius: 3px; transition: width 0.5s ease; }\n\n</style>\n</head>\n<body>\n\n<!-- HEADER -->\n<div class="header">\n  <div class="logo-area">\n    <div class="logo-icon">🔍</div>\n    <div class="logo-text">\n      <h1>HalluciNet Adversarial</h1>\n      <p><span class="pulse"></span>live · multi-agent hallucination detection · round 2</p>\n    </div>\n  </div>\n  <div class="header-badges">\n    <span class="badge badge-blue">Theme 1: Multi-Agent</span>\n    <span class="badge badge-cyan">Theme 3: World Modeling</span>\n    <span class="badge badge-purple">Theme 4: Self-Improvement</span>\n    <span class="badge badge-green">OpenEnv 2.0 ✓</span>\n    <span class="badge badge-orange">GRPO Trained ✓</span>\n  </div>\n</div>\n\n<div class="main">\n\n  <!-- HERO -->\n  <div class="hero">\n    <div>\n      <div class="hero-title">// adversarial rl environment</div>\n      <div class="hero-headline">Two agents compete.<br><span>One lies. One detects.</span></div>\n      <div class="hero-desc">Generator creates subtle hallucinations. Detector catches them with calibrated confidence. Oversight monitors the system. Curriculum escalates difficulty. No LLM judge — fully deterministic reward.</div>\n    </div>\n    <div class="hero-stats">\n      <div class="hero-stat">\n        <div class="hero-stat-val">5</div>\n        <div class="hero-stat-lbl">Difficulty Levels</div>\n      </div>\n      <div class="hero-stat">\n        <div class="hero-stat-val" id="stat-samples">73</div>\n        <div class="hero-stat-lbl">Curated Samples</div>\n      </div>\n      <div class="hero-stat">\n        <div class="hero-stat-val" id="stat-score">—</div>\n        <div class="hero-stat-lbl">Last Score</div>\n      </div>\n    </div>\n  </div>\n\n  <!-- HOW IT WORKS -->\n  <div class="tutorial-steps">\n    <div class="tutorial-step">\n      <div class="step-num">1</div>\n      <div class="step-title">Load Sample</div>\n      <div class="step-desc">Choose difficulty (Easy → Adversarial). Load a reference document and an LLM response that may contain errors.</div>\n      <div class="step-arrow">→</div>\n    </div>\n    <div class="tutorial-step">\n      <div class="step-num">2</div>\n      <div class="step-title">Generator Acts</div>\n      <div class="step-desc">Optionally write a hallucination to fool the detector. Generator is rewarded when detector misses it.</div>\n      <div class="step-arrow">→</div>\n    </div>\n    <div class="tutorial-step">\n      <div class="step-num">3</div>\n      <div class="step-title">Detector Responds</div>\n      <div class="step-desc">Submit detection — flag the hallucination, quote the wrong phrase, provide the correct fact. Confidence calibration matters.</div>\n      <div class="step-arrow">→</div>\n    </div>\n    <div class="tutorial-step">\n      <div class="step-num">4</div>\n      <div class="step-title">Debate + Oversight</div>\n      <div class="step-desc">Generator defends its response. Detector re-evaluates. Oversight tracks patterns across episodes. ELO updates.</div>\n    </div>\n  </div>\n\n  <!-- METRICS ROW -->\n  <div class="metrics-row">\n    <div class="metric-card">\n      <div class="metric-val" id="met-elo-det">1000</div>\n      <div class="metric-lbl">Detector ELO</div>\n    </div>\n    <div class="metric-card">\n      <div class="metric-val" id="met-elo-gen">1000</div>\n      <div class="metric-lbl">Generator ELO</div>\n    </div>\n    <div class="metric-card">\n      <div class="metric-val" id="met-rounds">0</div>\n      <div class="metric-lbl">Total Rounds</div>\n    </div>\n    <div class="metric-card">\n      <div class="metric-val" id="met-reliability">—</div>\n      <div class="metric-lbl">Reliability Score</div>\n    </div>\n    <div class="metric-card">\n      <div class="metric-val" id="met-episodes">0</div>\n      <div class="metric-lbl">Episodes Monitored</div>\n    </div>\n  </div>\n\n  <!-- TABS -->\n  <div class="tabs">\n    <button class="tab active" onclick="showTab(\'demo\', this)">🎮 Demo</button>\n    <button class="tab" onclick="showTab(\'training\', this)">📈 Training Results</button>\n    <button class="tab" onclick="showTab(\'leaderboard\', this)">🏆 Leaderboard</button>\n    <button class="tab" onclick="showTab(\'worldmodel\', this)">🌍 World Model</button>\n    <button class="tab" onclick="showTab(\'api\', this)">⚡ API Reference</button>\n    <button class="tab" onclick="showTab(\'oversight\', this)">👁 Oversight</button>\n    <button class="tab" onclick="showTab(\'taxonomy\', this)">🧬 Taxonomy</button>\n  </div>\n\n  <!-- ===== DEMO TAB ===== -->\n  <div id="tab-demo">\n    <div class="card">\n      <div class="card-title"><span class="icon">📋</span> Sample Loader\n        <span style="margin-left:auto">\n          <button class="btn btn-primary" onclick="loadSample()" id="load-btn" style="font-size:12px;padding:7px 16px;">Load Sample</button>\n        </span>\n      </div>\n      <div class="task-pills">\n        <button class="task-pill active" onclick="selectTask(this,\'easy\')">Easy · 10</button>\n        <button class="task-pill" onclick="selectTask(this,\'medium\')">Medium · 12</button>\n        <button class="task-pill" onclick="selectTask(this,\'hard\')">Hard · 19</button>\n        <button class="task-pill" onclick="selectTask(this,\'expert\')">Expert · 20</button>\n        <button class="task-pill" onclick="selectTask(this,\'adversarial\')">Adversarial · 12</button>\n      </div>\n      <div class="grid-2">\n        <div>\n          <div class="sample-label">Reference Document (Ground Truth)</div>\n          <div class="sample-box" id="ref-box">Click "Load Sample" to begin. Choose a difficulty level above.</div>\n        </div>\n        <div>\n          <div class="sample-label">LLM Response (Evaluate This)</div>\n          <div class="sample-box" id="llm-box">The LLM response will appear here. Find the hallucination.</div>\n        </div>\n      </div>\n    </div>\n\n    <div class="grid-2">\n      <!-- GENERATOR -->\n      <div class="agent-card gen-card">\n        <div class="agent-header">\n          <div class="agent-icon">🔥</div>\n          <div>\n            <div class="agent-name">Generator Agent</div>\n            <div class="agent-sub">Creates hallucinations to fool the detector</div>\n          </div>\n        </div>\n        <label>Error Type</label>\n        <select id="err-type">\n          <option value="year_swap">Year Swap — wrong dates</option>\n          <option value="name_swap">Name Swap — wrong person</option>\n          <option value="number_swap">Number Swap — digit errors</option>\n          <option value="negation">Negation Trap — flipped meaning</option>\n          <option value="entity_flip">Entity Flip — wrong subject</option>\n          <option value="unit_shift">Unit Shift — wrong units</option>\n          <option value="partial_truth">Partial Truth — mixed facts</option>\n        </select>\n        <label>Hallucinated Response</label>\n        <textarea id="gen-text" rows="3" placeholder="Write a subtle hallucination here to fool the detector..."></textarea>\n        <div style="margin:10px 0">\n          <label>Generator Confidence: <span id="gc-val" style="color:var(--orange);font-family:var(--mono)">0.80</span></label>\n          <input type="range" id="gen-conf" min="0.01" max="0.99" step="0.01" value="0.8"\n            oninput="document.getElementById(\'gc-val\').innerText=parseFloat(this.value).toFixed(2)">\n        </div>\n        <button class="btn btn-danger btn-full" onclick="runGenerator()" id="gen-btn">🔥 Generate Hallucination</button>\n        <div id="gen-result" class="hidden" style="margin-top:12px">\n          <div class="score-big" id="gen-score" style="text-align:center;margin-bottom:8px;font-size:28px"></div>\n          <div class="feedback-box" id="gen-feedback"></div>\n        </div>\n      </div>\n\n      <!-- DETECTOR -->\n      <div class="agent-card det-card">\n        <div class="agent-header">\n          <div class="agent-icon">🛡️</div>\n          <div>\n            <div class="agent-name">Detector Agent</div>\n            <div class="agent-sub">Catches hallucinations with calibrated confidence</div>\n          </div>\n        </div>\n        <label>Hallucinated Claim (leave blank if clean)</label>\n        <input type="text" id="det-claim" placeholder="Quote the exact wrong phrase...">\n        <label>Correct Fact from Reference</label>\n        <input type="text" id="det-fact" placeholder="What does the reference actually say?">\n        <div style="margin:10px 0">\n          <label>Detector Confidence: <span id="dc-val" style="color:var(--green);font-family:var(--mono)">0.80</span></label>\n          <input type="range" id="det-conf" min="0.01" max="0.99" step="0.01" value="0.8"\n            oninput="document.getElementById(\'dc-val\').innerText=parseFloat(this.value).toFixed(2)">\n        </div>\n        <button class="btn btn-success btn-full" onclick="runDetector()" id="det-btn">🛡️ Submit Detection</button>\n        <div id="det-result" class="hidden" style="margin-top:12px">\n          <div style="text-align:center;margin-bottom:8px">\n            <div class="score-big" id="det-score"></div>\n            <div style="font-size:11px;color:var(--muted);margin-top:2px;font-family:var(--mono)">Detector Score</div>\n          </div>\n          <div class="breakdown-grid" id="det-breakdown"></div>\n          <div class="feedback-box" id="det-feedback"></div>\n        </div>\n      </div>\n    </div>\n\n    <!-- DEBATE SECTION -->\n    <div id="debate-section" class="hidden">\n      <div class="card" style="border-color:rgba(249,115,22,0.3)">\n        <div class="card-title"><span class="icon">⚔️</span> Debate Round — Generator Defense</div>\n        <p style="font-size:12px;color:var(--muted);margin-bottom:14px;line-height:1.6">Generator gets one chance to defend its response. Detector re-evaluates with new information. Ground truth adjudicates. This is <strong style="color:var(--text)">multi-step reasoning</strong> — state persists across turns.</p>\n        <button class="btn btn-debate btn-full" id="debate-btn" onclick="runDebate()">⚔️ Trigger Debate Round</button>\n        <div id="debate-result" class="hidden" style="margin-top:12px"></div>\n      </div>\n    </div>\n\n    <!-- VERDICT -->\n    <div id="verdict-box" class="hidden">\n      <div class="verdict verdict-pending" id="verdict-text"></div>\n      <div style="text-align:center;font-size:11px;color:var(--muted);margin-top:8px;font-family:var(--mono)">\n        Generator reward: <span id="g-reward" style="color:var(--orange);font-weight:600"></span>\n        &nbsp;·&nbsp;\n        Detector reward: <span id="d-reward" style="color:var(--green);font-weight:600"></span>\n        &nbsp;·&nbsp;\n        ELO updated: <span id="elo-update" style="color:var(--accent)">—</span>\n      </div>\n    </div>\n  </div>\n\n  <!-- ===== TRAINING TAB ===== -->\n  <div id="tab-training" class="hidden">\n    <div class="card">\n      <div class="card-title"><span class="icon">📈</span> GRPO Training Results — Qwen2.5-3B</div>\n      <div class="curriculum-flow">\n        <div class="curr-level curr-easy">easy</div>\n        <div class="curr-arrow">→ promoted</div>\n        <div class="curr-level curr-medium">medium</div>\n        <div class="curr-arrow">→ promoted</div>\n        <div class="curr-level curr-hard">hard</div>\n        <div class="curr-arrow">→ tried</div>\n        <div class="curr-level curr-expert">expert</div>\n        <div class="curr-arrow">→ demoted (correct)</div>\n        <div class="curr-level curr-hard">hard</div>\n      </div>\n      <p style="font-size:12px;color:var(--muted);margin-bottom:16px;line-height:1.6;font-family:var(--mono)">// 90 training steps · 3 epochs · 120 samples · Tesla T4 · GRPO via Unsloth</p>\n\n      <div class="training-grid">\n        <div class="training-col training-before">\n          <div class="training-col-title">Before Training — Baseline</div>\n          <div class="info-row"><span class="info-key">model</span><span class="info-val">qwen2.5-3b-baseline</span></div>\n          <div class="info-row"><span class="info-key">medium_reward</span><span style="color:var(--red);font-weight:600;font-family:var(--mono)">0.9490</span></div>\n          <div class="info-row"><span class="info-key">finding</span><span class="info-val" style="font-size:11px">Reward ceiling — exploiting JSON format</span></div>\n          <div class="info-row"><span class="info-key">curriculum</span><span class="info-val">easy (starting level)</span></div>\n        </div>\n        <div class="training-col training-after">\n          <div class="training-col-title">After Training — GRPO</div>\n          <div class="info-row"><span class="info-key">model</span><span class="info-val">qwen2.5-3b-grpo</span></div>\n          <div class="info-row"><span class="info-key">medium_reward</span><span style="color:var(--green);font-weight:600;font-family:var(--mono)">0.774</span></div>\n          <div class="info-row"><span class="info-key">hard_reward</span><span style="color:var(--green);font-weight:600;font-family:var(--mono)">0.729</span></div>\n          <div class="info-row"><span class="info-key">curriculum</span><span style="color:var(--orange);font-weight:600">hard (19 promotions)</span></div>\n        </div>\n      </div>\n\n      <div class="section-title">Per-Task Performance</div>\n      <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:8px">\n        <div style="background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:12px">\n          <div style="font-size:10px;color:var(--muted);font-family:var(--mono);margin-bottom:6px">easy</div>\n          <div style="font-size:18px;font-weight:700;font-family:var(--mono);color:var(--green)">0.647</div>\n          <div class="progress-bar"><div class="progress-fill" style="width:64.7%;background:var(--green)"></div></div>\n        </div>\n        <div style="background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:12px">\n          <div style="font-size:10px;color:var(--muted);font-family:var(--mono);margin-bottom:6px">medium</div>\n          <div style="font-size:18px;font-weight:700;font-family:var(--mono);color:var(--accent)">0.774</div>\n          <div class="progress-bar"><div class="progress-fill" style="width:77.4%;background:var(--accent)"></div></div>\n        </div>\n        <div style="background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:12px">\n          <div style="font-size:10px;color:var(--muted);font-family:var(--mono);margin-bottom:6px">hard</div>\n          <div style="font-size:18px;font-weight:700;font-family:var(--mono);color:var(--yellow)">0.729</div>\n          <div class="progress-bar"><div class="progress-fill" style="width:72.9%;background:var(--yellow)"></div></div>\n        </div>\n        <div style="background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:12px">\n          <div style="font-size:10px;color:var(--muted);font-family:var(--mono);margin-bottom:6px">expert</div>\n          <div style="font-size:18px;font-weight:700;font-family:var(--mono);color:var(--red)">0.010</div>\n          <div class="progress-bar"><div class="progress-fill" style="width:1%;background:var(--red)"></div></div>\n          <div style="font-size:9px;color:var(--muted);margin-top:4px">correctly demoted</div>\n        </div>\n      </div>\n\n      <div style="background:rgba(59,130,246,0.06);border:1px solid rgba(59,130,246,0.2);border-radius:8px;padding:14px;margin-top:16px">\n        <div style="font-size:11px;font-weight:600;color:var(--accent);margin-bottom:6px;font-family:var(--mono)">// key finding</div>\n        <div style="font-size:12px;color:#9ca3af;line-height:1.6">Baseline scored 0.949 on medium — a reward ceiling effect where the untrained model exploited JSON formatting. GRPO training broke this shortcut and forced genuine hallucination detection. The curriculum correctly escalated to hard difficulty across 90 sessions, with expert-level demotion validating the system works as designed.</div>\n      </div>\n    </div>\n  </div>\n\n  <!-- ===== LEADERBOARD TAB ===== -->\n  <div id="tab-leaderboard" class="hidden">\n    <div class="card">\n      <div class="card-title"><span class="icon">🏆</span> Model Leaderboard\n        <span style="margin-left:auto;font-size:11px;color:var(--muted);font-family:var(--mono)">deterministic grader · no LLM judge</span>\n      </div>\n      <table>\n        <thead>\n          <tr>\n            <th>#</th><th>Model</th><th>Easy</th><th>Medium</th><th>Hard</th><th>Expert</th><th>Adversarial</th><th>Overall</th>\n          </tr>\n        </thead>\n        <tbody id="lb-body">\n          <tr><td colspan="8" style="text-align:center;color:var(--muted);padding:20px;font-family:var(--mono)">Loading...</td></tr>\n        </tbody>\n      </table>\n      <div style="margin-top:12px;padding:10px;background:var(--bg);border-radius:8px;font-size:11px;color:var(--muted);font-family:var(--mono)">\n        POST /leaderboard/record to add any model\'s scores. Overall = mean across all 5 task levels.\n      </div>\n    </div>\n  </div>\n\n  <!-- ===== WORLD MODEL TAB ===== -->\n  <div id="tab-worldmodel" class="hidden">\n    <div class="card">\n      <div class="card-title"><span class="icon">🌍</span> World Model — Theme 3: World Modeling\n        <span class="new-tag">new</span>\n      </div>\n      <p style="font-size:12px;color:var(--muted);margin-bottom:16px;line-height:1.6">The oversight agent builds a persistent internal model of agent behavior across episodes — tracking reliability, blind spots, and calibration drift. This endpoint shows the system\'s current world model state.</p>\n      \n      <div class="wm-grid">\n        <div class="wm-section">\n          <div class="wm-title">Agent Model</div>\n          <div class="info-row"><span class="info-key">detector_reliability</span><span class="info-val" id="wm-reliability">—</span></div>\n          <div class="info-row"><span class="info-key">overconfidence_rate</span><span class="info-val" id="wm-overconf">—</span></div>\n          <div class="info-row"><span class="info-key">calibration_error</span><span class="info-val" id="wm-ece">—</span></div>\n          <div class="info-row"><span class="info-key">blind_spots</span><span class="info-val" id="wm-blindspots">—</span></div>\n          <div class="info-row"><span class="info-key">episodes_monitored</span><span class="info-val" id="wm-episodes">—</span></div>\n        </div>\n        <div class="wm-section">\n          <div class="wm-title">Environment Model</div>\n          <div class="info-row"><span class="info-key">current_difficulty</span><span class="info-val" id="wm-difficulty">—</span></div>\n          <div class="info-row"><span class="info-key">detector_elo</span><span class="info-val" id="wm-det-elo">—</span></div>\n          <div class="info-row"><span class="info-key">generator_elo</span><span class="info-val" id="wm-gen-elo">—</span></div>\n          <div class="info-row"><span class="info-key">total_rounds</span><span class="info-val" id="wm-rounds">—</span></div>\n          <div class="info-row"><span class="info-key">predicted_next_action</span><span class="info-val" id="wm-action">—</span></div>\n        </div>\n      </div>\n\n      <div class="section-title">ELO Ratings — Live</div>\n      <div style="background:var(--bg);border:1px solid var(--border);border-radius:10px;padding:16px">\n        <div class="elo-bars" id="elo-bars">\n          <div class="elo-bar-wrap">\n            <div class="elo-bar-val" id="elo-det-val" style="color:var(--green)">1000</div>\n            <div class="elo-bar elo-bar-det" id="elo-det-bar" style="height:50%"></div>\n            <div class="elo-bar-lbl">Detector</div>\n          </div>\n          <div class="elo-bar-wrap">\n            <div class="elo-bar-val" id="elo-gen-val" style="color:var(--orange)">1000</div>\n            <div class="elo-bar elo-bar-gen" id="elo-gen-bar" style="height:50%"></div>\n            <div class="elo-bar-lbl">Generator</div>\n          </div>\n        </div>\n      </div>\n\n      <div class="section-title">Multi-Step Episode Tracking</div>\n      <div style="background:var(--bg);border:1px solid var(--border);border-radius:10px;padding:16px">\n        <div class="info-row"><span class="info-key">total_episodes</span><span class="info-val" id="wm-total-ep">—</span></div>\n        <div class="info-row"><span class="info-key">episode_flow</span><span style="color:var(--accent2);font-family:var(--mono);font-size:11px">generator → detector → debate → oversight → curriculum</span></div>\n        <div class="info-row"><span class="info-key">system_health</span><span class="info-val" id="wm-health" style="font-size:11px">—</span></div>\n      </div>\n\n      <div style="text-align:right;margin-top:12px">\n        <button class="btn btn-primary" onclick="loadWorldModel()" style="font-size:11px;padding:7px 14px">↻ Refresh World Model</button>\n      </div>\n    </div>\n  </div>\n\n  <!-- ===== API TAB ===== -->\n  <div id="tab-api" class="hidden">\n    <div class="card">\n      <div class="card-title"><span class="icon">⚡</span> API Reference</div>\n      \n      <div class="section-title">Quick Start</div>\n      <div class="code-block">\n<span class="code-comment"># 1. health check</span>\n<span class="code-key">curl</span> https://rushikeshbathe096-hallucinet.hf.space/health\n\n<span class="code-comment"># 2. reset detector episode</span>\n<span class="code-key">curl</span> -X POST .../reset -d <span class="code-str">\'{"task_id": "hard"}\'</span>\n\n<span class="code-comment"># 3. submit detection</span>\n<span class="code-key">curl</span> -X POST .../step -d <span class="code-str">\'{"action": {"has_hallucination": true, "hallucinated_claim": "1902", "correct_fact": "1889", "confidence": 0.95}}\'</span>\n\n<span class="code-comment"># 4. trigger debate</span>\n<span class="code-key">curl</span> -X POST .../debate -d <span class="code-str">\'{"generator_defense": "My response is accurate"}\'</span>\n\n<span class="code-comment"># 5. openenv validate</span>\n<span class="code-key">openenv validate</span> --url https://rushikeshbathe096-hallucinet.hf.space\n      </div>\n\n      <div class="section-title">Core Endpoints</div>\n      <div class="endpoint-grid">\n        <div class="endpoint-item">\n          <div><span class="endpoint-method method-get">GET</span><span class="endpoint-path">/health</span></div>\n          <div class="endpoint-desc">Environment health check</div>\n        </div>\n        <div class="endpoint-item">\n          <div><span class="endpoint-method method-post">POST</span><span class="endpoint-path">/reset</span></div>\n          <div class="endpoint-desc">Start detector episode {task_id}</div>\n        </div>\n        <div class="endpoint-item">\n          <div><span class="endpoint-method method-post">POST</span><span class="endpoint-path">/step</span></div>\n          <div class="endpoint-desc">Submit detection action</div>\n        </div>\n        <div class="endpoint-item">\n          <div><span class="endpoint-method method-get">GET</span><span class="endpoint-path">/state</span></div>\n          <div class="endpoint-desc">Current episode state</div>\n        </div>\n        <div class="endpoint-item">\n          <div><span class="endpoint-method method-post">POST</span><span class="endpoint-path">/generator/reset</span></div>\n          <div class="endpoint-desc">Start generator episode</div>\n        </div>\n        <div class="endpoint-item">\n          <div><span class="endpoint-method method-post">POST</span><span class="endpoint-path">/generator/step</span></div>\n          <div class="endpoint-desc">Submit generator action</div>\n        </div>\n        <div class="endpoint-item">\n          <div><span class="endpoint-method method-post">POST</span><span class="endpoint-path">/debate</span><span class="new-tag">new</span></div>\n          <div class="endpoint-desc">Generator defends, detector re-evaluates</div>\n        </div>\n        <div class="endpoint-item">\n          <div><span class="endpoint-method method-get">GET</span><span class="endpoint-path">/elo/standings</span><span class="new-tag">new</span></div>\n          <div class="endpoint-desc">Live ELO ratings for both agents</div>\n        </div>\n        <div class="endpoint-item">\n          <div><span class="endpoint-method method-get">GET</span><span class="endpoint-path">/calibration</span><span class="new-tag">new</span></div>\n          <div class="endpoint-desc">Confidence vs accuracy (ECE metric)</div>\n        </div>\n        <div class="endpoint-item">\n          <div><span class="endpoint-method method-get">GET</span><span class="endpoint-path">/world/model</span><span class="new-tag">new</span></div>\n          <div class="endpoint-desc">Persistent world model state</div>\n        </div>\n        <div class="endpoint-item">\n          <div><span class="endpoint-method method-get">GET</span><span class="endpoint-path">/training/summary</span><span class="new-tag">new</span></div>\n          <div class="endpoint-desc">Before/after GRPO training results</div>\n        </div>\n        <div class="endpoint-item">\n          <div><span class="endpoint-method method-get">GET</span><span class="endpoint-path">/oversight/status</span></div>\n          <div class="endpoint-desc">System reliability metrics</div>\n        </div>\n        <div class="endpoint-item">\n          <div><span class="endpoint-method method-get">GET</span><span class="endpoint-path">/curriculum/status</span></div>\n          <div class="endpoint-desc">Curriculum progression state</div>\n        </div>\n        <div class="endpoint-item">\n          <div><span class="endpoint-method method-get">GET</span><span class="endpoint-path">/leaderboard</span></div>\n          <div class="endpoint-desc">Dynamic model rankings</div>\n        </div>\n      </div>\n    </div>\n  </div>\n\n  <!-- ===== OVERSIGHT TAB ===== -->\n  <div id="tab-oversight" class="hidden">\n    <div class="card">\n      <div class="card-title"><span class="icon">👁</span> Oversight Agent — System Monitor</div>\n      <div id="oversight-content" style="color:var(--muted);text-align:center;padding:20px;font-family:var(--mono)">Loading oversight data...</div>\n    </div>\n  </div>\n\n\n  <!-- ===== TAXONOMY TAB ===== -->\n  <div id="tab-taxonomy" class="hidden">\n    <div class="card">\n      <div class="card-title"><span class="icon">🧬</span> Hallucination Error Taxonomy\n        <span class="new-tag">dynamic</span>\n        <span style="margin-left:auto;font-size:11px;color:var(--muted);font-family:var(--mono)" id="tax-summary"></span>\n      </div>\n      <p style="font-size:12px;color:var(--muted);margin-bottom:16px;line-height:1.6">All error types are dynamically scanned from tasks.py. Each card shows the error type, its difficulty tier, description, example, and sample count.</p>\n      <div id="taxonomy-content" style="color:var(--muted);text-align:center;padding:20px;font-family:var(--mono)">Loading taxonomy...</div>\n    </div>\n  </div>\n\n</div><!-- /main -->\n\n<script>\nlet currentTask = "easy";\n\nfunction selectTask(el, task) {\n  document.querySelectorAll(".task-pill").forEach(p => p.classList.remove("active"));\n  el.classList.add("active");\n  currentTask = task;\n}\n\nfunction showTab(name, btn) {\n  ["demo","training","leaderboard","worldmodel","api","oversight","taxonomy"].forEach(t => {\n    document.getElementById("tab-"+t).classList.add("hidden");\n  });\n  document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));\n  document.getElementById("tab-"+name).classList.remove("hidden");\n  if (btn) btn.classList.add("active");\n  if (name === "leaderboard") loadLeaderboard();\n  if (name === "oversight") loadOversight();\n  if (name === "worldmodel") loadWorldModel();\n  if (name === "taxonomy") loadTaxonomy();\n}\n\nasync function loadSample() {\n  const btn = document.getElementById("load-btn");\n  btn.disabled = true;\n  btn.innerHTML = \'<span class="spinner"></span> Loading...\';\n  try {\n    const r = await fetch("/reset", {\n      method:"POST", headers:{"Content-Type":"application/json"},\n      body: JSON.stringify({task_id: currentTask})\n    });\n    const d = await r.json();\n    const obs = d.observation || d;\n    document.getElementById("ref-box").innerText = obs.reference_document || "—";\n    document.getElementById("llm-box").innerText = obs.llm_response || "—";\n    document.getElementById("stat-samples").innerText = obs.total_samples || "—";\n    ["gen-result","det-result","verdict-box","debate-section"].forEach(id => {\n      document.getElementById(id).classList.add("hidden");\n    });\n    document.getElementById("stat-score").innerText = "—";\n  } catch(e) {\n    document.getElementById("ref-box").innerText = "Error: " + e.message;\n  }\n  btn.disabled = false;\n  btn.innerHTML = "Load Sample";\n}\n\nasync function runGenerator() {\n  const text = document.getElementById("gen-text").value;\n  if (!text) { alert("Write a hallucination first"); return; }\n  const btn = document.getElementById("gen-btn");\n  btn.disabled = true;\n  btn.innerHTML = \'<span class="spinner"></span> Generating...\';\n  try {\n    await fetch("/generator/reset", {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({task_id:currentTask})});\n    const r = await fetch("/generator/step", {\n      method:"POST", headers:{"Content-Type":"application/json"},\n      body: JSON.stringify({action:{generated_response:text,error_type:document.getElementById("err-type").value,confidence:parseFloat(document.getElementById("gen-conf").value)}})\n    });\n    const d = await r.json();\n    const obs = d.observation || d;\n    const score = obs.fooling_rate || obs.score || 0;\n    const scoreEl = document.getElementById("gen-score");\n    scoreEl.innerText = "Generator: " + score.toFixed(3);\n    scoreEl.className = "score-big " + getScoreClass(score);\n    document.getElementById("gen-feedback").innerText = obs.feedback || "Submitted";\n    document.getElementById("gen-result").classList.remove("hidden");\n    document.getElementById("llm-box").innerText = text;\n  } catch(e) { alert("Error: " + e.message); }\n  btn.disabled = false;\n  btn.innerHTML = "🔥 Generate Hallucination";\n}\n\nasync function runDetector() {\n  const claim = document.getElementById("det-claim").value;\n  const btn = document.getElementById("det-btn");\n  btn.disabled = true;\n  btn.innerHTML = \'<span class="spinner"></span> Detecting...\';\n  try {\n    const r = await fetch("/step", {\n      method:"POST", headers:{"Content-Type":"application/json"},\n      body: JSON.stringify({action:{\n        has_hallucination: claim.length > 0,\n        hallucinated_claim: claim || null,\n        correct_fact: document.getElementById("det-fact").value || null,\n        confidence: parseFloat(document.getElementById("det-conf").value)\n      }})\n    });\n    const d = await r.json();\n    const obs = d.observation || d;\n    const score = obs.score || 0;\n    const bd = obs.metadata?.reward_breakdown || {};\n    \n    document.getElementById("det-score").innerText = score.toFixed(3);\n    document.getElementById("det-score").className = "score-big " + getScoreClass(score);\n    document.getElementById("stat-score").innerText = score.toFixed(3);\n    \n    document.getElementById("det-breakdown").innerHTML = `\n      <div class="breakdown-item"><div class="breakdown-val">${(bd.detection||0).toFixed(2)}</div><div class="breakdown-lbl">Detection</div></div>\n      <div class="breakdown-item"><div class="breakdown-val">${(bd.phrase||0).toFixed(2)}</div><div class="breakdown-lbl">Phrase ID</div></div>\n      <div class="breakdown-item"><div class="breakdown-val">${(bd.fact||0).toFixed(2)}</div><div class="breakdown-lbl">Correct Fact</div></div>\n      <div class="breakdown-item"><div class="breakdown-val">${(bd.calibration||0).toFixed(2)}</div><div class="breakdown-lbl">Calibration</div></div>\n    `;\n    document.getElementById("det-feedback").innerText = obs.feedback || "";\n    document.getElementById("det-result").classList.remove("hidden");\n    \n    const caught = claim.length > 0;\n    const vb = document.getElementById("verdict-box");\n    const vt = document.getElementById("verdict-text");\n    vb.classList.remove("hidden");\n    if (caught && score > 0.4) {\n      vt.className = "verdict verdict-win";\n      vt.innerText = "🛡️ DETECTOR WINS — Hallucination caught!";\n    } else if (!caught) {\n      vt.className = "verdict verdict-lose";\n      vt.innerText = "🔥 GENERATOR WINS — Hallucination slipped through!";\n    } else {\n      vt.className = "verdict verdict-pending";\n      vt.innerText = "⚠️ Partial detection — keep training!";\n    }\n    document.getElementById("g-reward").innerText = caught ? "0.001" : "0.999";\n    document.getElementById("d-reward").innerText = score.toFixed(3);\n    document.getElementById("debate-section").classList.remove("hidden");\n    \n    // Update metrics\n    updateMetrics();\n  } catch(e) { alert("Error: " + e.message); }\n  btn.disabled = false;\n  btn.innerHTML = "🛡️ Submit Detection";\n}\n\nasync function runDebate() {\n  const btn = document.getElementById("debate-btn");\n  btn.disabled = true;\n  btn.innerHTML = \'<span class="spinner"></span> Debating...\';\n  try {\n    const r = await fetch("/debate", {\n      method:"POST", headers:{"Content-Type":"application/json"},\n      body: JSON.stringify({generator_defense:"My response is factually supported by the source material and follows standard interpretation."})\n    });\n    const d = await r.json();\n    const debate = d.debate || {};\n    const win = debate.outcome === "detector_wins";\n    document.getElementById("debate-result").innerHTML = `\n      <div class="verdict ${win ? \'verdict-win\' : \'verdict-pending\'}" style="margin-top:8px">\n        ${win ? \'🛡️ DETECTOR WINS DEBATE\' : \'🔥 DEBATE INCONCLUSIVE\'}\n      </div>\n      <div class="feedback-box" style="margin-top:8px">\n        Defense score: ${(debate.generator_defense_score||0).toFixed(3)} · \n        Reward delta: ${(debate.generator_final_reward_delta||0).toFixed(3)} · \n        ${debate.adjudication_reason || "Debate complete"}\n      </div>\n    `;\n    document.getElementById("debate-result").classList.remove("hidden");\n    document.getElementById("elo-update").innerText = win ? "↑ detector" : "↑ generator";\n    updateMetrics();\n  } catch(e) { alert("Error: " + e.message); }\n  btn.disabled = false;\n  btn.innerHTML = "⚔️ Trigger Debate Round";\n}\n\nfunction getScoreClass(s) {\n  if (s >= 0.9) return "score-excellent";\n  if (s >= 0.7) return "score-good";\n  if (s >= 0.4) return "score-partial";\n  return "score-bad";\n}\n\nfunction clamp01(v) {\n  return Math.max(0, Math.min(1, Number.isFinite(v) ? v : 0));\n}\n\nfunction computeFrontendReliability(oversight, calibration) {\n  const accuracy = clamp01(oversight?.reliability_score ?? 0);\n  const overconfidence = clamp01(oversight?.overconfidence_rate ?? 0);\n  const ece = calibration?.detector?.calibration_error;\n  const calibrationQuality = Number.isFinite(ece) ? clamp01(1 - ece) : clamp01(1 - overconfidence);\n  const blindSpotCount = Array.isArray(oversight?.blind_spots) ? oversight.blind_spots.length : 0;\n  const coverage = clamp01(1 - (blindSpotCount / 5));\n  const blended = clamp01((0.5 * accuracy) + (0.3 * calibrationQuality) + (0.2 * coverage));\n  return { blended, accuracy, calibrationQuality, coverage };\n}\n\nasync function updateMetrics() {\n  try {\n    const elo = await fetch("/elo/standings").then(r=>r.json());\n    document.getElementById("met-elo-det").innerText = Math.round(elo.detector_elo);\n    document.getElementById("met-elo-gen").innerText = Math.round(elo.generator_elo);\n    document.getElementById("met-rounds").innerText = elo.total_rounds;\n  } catch(e) {}\n  try {\n    const [ov, cal] = await Promise.all([\n      fetch("/oversight/status").then(r=>r.json()),\n      fetch("/calibration").then(r=>r.json()).catch(() => ({}))\n    ]);\n    const rel = computeFrontendReliability(ov, cal);\n    document.getElementById("met-reliability").innerText = rel.blended.toFixed(2);\n    document.getElementById("met-episodes").innerText = ov.episodes_monitored || 0;\n  } catch(e) {}\n}\n\nasync function loadLeaderboard() {\n  try {\n    const d = await fetch("/leaderboard").then(r=>r.json());\n    const tbody = document.getElementById("lb-body");\n    const ranks = ["rank-1","rank-2","rank-3"];\n    tbody.innerHTML = d.leaderboard.map((e,i) => `\n      <tr>\n        <td><span class="rank-badge ${ranks[i]||\'\'}">${e.rank}</span></td>\n        <td style="font-family:var(--mono)">${e.model} ${e.trained ? \'<span class="trained-tag">GRPO</span>\' : \'\'}</td>\n        <td>${e.easy}</td><td>${e.medium}</td><td>${e.hard}</td>\n        <td>${e.expert}</td><td>${e.adversarial}</td>\n        <td><strong style="color:var(--accent);font-family:var(--mono)">${e.overall}</strong></td>\n      </tr>\n    `).join("");\n  } catch(e) {\n    document.getElementById("lb-body").innerHTML = \'<tr><td colspan="8" style="text-align:center;color:var(--muted);font-family:var(--mono)">Error loading leaderboard</td></tr>\';\n  }\n}\n\nasync function loadOversight() {\n  try {\n    const [d, cal] = await Promise.all([\n      fetch("/oversight/status").then(r=>r.json()),\n      fetch("/calibration").then(r=>r.json()).catch(() => ({}))\n    ]);\n    const rel = computeFrontendReliability(d, cal);\n    const score = rel.blended;\n    const color = score > 0.8 ? "var(--green)" : score > 0.6 ? "var(--yellow)" : "var(--red)";\n    document.getElementById("oversight-content").innerHTML = `\n      <div style="font-size:56px;font-weight:800;color:${color};margin-bottom:8px;font-family:var(--mono)">${score.toFixed(3)}</div>\n      <div style="color:var(--muted);margin-bottom:20px;font-size:12px">Frontend Reliability Score (50% accuracy · 30% calibration · 20% coverage)</div>\n      <div class="info-row"><span class="info-key">Accuracy Component</span><span class="info-val">${rel.accuracy.toFixed(3)}</span></div>\n      <div class="info-row"><span class="info-key">Calibration Component</span><span class="info-val">${rel.calibrationQuality.toFixed(3)}</span></div>\n      <div class="info-row"><span class="info-key">Coverage Component</span><span class="info-val">${rel.coverage.toFixed(3)}</span></div>\n      <div class="info-row"><span class="info-key">Overconfidence Rate</span><span class="info-val">${(d.overconfidence_rate||0).toFixed(3)}</span></div>\n      <div class="info-row"><span class="info-key">Blind Spots</span><span class="info-val">${d.blind_spots?.length ? d.blind_spots.join(", ") : "None detected"}</span></div>\n      <div class="info-row"><span class="info-key">Episodes Monitored</span><span class="info-val">${d.episodes_monitored || 0}</span></div>\n      <div class="info-row"><span class="info-key">Fleet AI Bonus</span><span class="info-val" style="color:var(--green)">${d.fleet_ai_bonus ? "✓ enabled" : "—"}</span></div>\n      <div style="background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:12px;margin-top:14px;font-size:12px;color:#9ca3af;font-family:var(--mono)">${d.system_feedback || "—"}</div>\n    `;\n  } catch(e) {\n    document.getElementById("oversight-content").innerText = "Error loading oversight data";\n  }\n}\n\nasync function loadWorldModel() {\n  try {\n    const d = await fetch("/world/model").then(r=>r.json());\n    const am = d.agent_model || {};\n    const em = d.environment_model || {};\n    const ms = d.multi_step_episodes || {};\n    \n    document.getElementById("wm-reliability").innerText = (am.detector_reliability||0).toFixed(3);\n    document.getElementById("wm-overconf").innerText = (am.overconfidence_rate||0).toFixed(3);\n    document.getElementById("wm-ece").innerText = `${(am.calibration_error||0).toFixed(4)} (${am.calibration_interpretation||"N/A"})`;\n    document.getElementById("wm-blindspots").innerText = am.detector_blind_spots?.length ? am.detector_blind_spots.join(", ") : "None detected";\n    document.getElementById("wm-episodes").innerText = am.episodes_monitored || 0;\n    document.getElementById("wm-difficulty").innerText = em.current_difficulty || "—";\n    document.getElementById("wm-det-elo").innerText = em.detector_elo || 1000;\n    document.getElementById("wm-gen-elo").innerText = em.generator_elo || 1000;\n    document.getElementById("wm-rounds").innerText = em.total_rounds || 0;\n    document.getElementById("wm-action").innerText = d.predicted_next_action || "—";\n    document.getElementById("wm-total-ep").innerText = ms.total_episodes || 0;\n    document.getElementById("wm-health").innerText = d.system_health || "—";\n    \n    // ELO bars\n    const detElo = em.detector_elo || 1000;\n    const genElo = em.generator_elo || 1000;\n    const maxElo = Math.max(detElo, genElo, 1001);\n    const minElo = Math.min(detElo, genElo, 999);\n    const range = maxElo - minElo + 100;\n    const detH = Math.max(10, Math.round(((detElo - minElo + 50) / range) * 80));\n    const genH = Math.max(10, Math.round(((genElo - minElo + 50) / range) * 80));\n    \n    document.getElementById("elo-det-bar").style.height = detH + "%";\n    document.getElementById("elo-gen-bar").style.height = genH + "%";\n    document.getElementById("elo-det-val").innerText = Math.round(detElo);\n    document.getElementById("elo-gen-val").innerText = Math.round(genElo);\n\n    // Update header metrics too\n    document.getElementById("met-elo-det").innerText = Math.round(detElo);\n    document.getElementById("met-elo-gen").innerText = Math.round(genElo);\n    document.getElementById("met-rounds").innerText = em.total_rounds || 0;\n  } catch(e) {\n    console.error("World model error:", e);\n  }\n}\n\nasync function loadTaxonomy() {\n  try {\n    const d = await fetch("/taxonomy").then(r=>r.json());\n    const diffColors = {easy:"var(--green)",medium:"var(--yellow)",hard:"var(--orange)",expert:"var(--red)"};\n    const diffBg = {easy:"rgba(16,185,129,0.1)",medium:"rgba(245,158,11,0.1)",hard:"rgba(249,115,22,0.1)",expert:"rgba(239,68,68,0.1)"};\n    const diffBorder = {easy:"rgba(16,185,129,0.3)",medium:"rgba(245,158,11,0.3)",hard:"rgba(249,115,22,0.3)",expert:"rgba(239,68,68,0.3)"};\n    document.getElementById("tax-summary").innerText = d.total_error_types + " types · " + d.total_samples_scanned + " samples";\n    let html = "";\n    const tax = d.taxonomy || {};\n    const counts = d.sample_counts || {};\n    for (const [cat, subs] of Object.entries(tax)) {\n      html += `<div style="margin-bottom:20px">`;\n      const catIcon = cat.includes("Factual") ? "📊" : cat.includes("Logical") ? "🧠" : "⚡";\n      html += `<div style="font-size:13px;font-weight:700;color:var(--text);margin-bottom:12px;display:flex;align-items:center;gap:8px;cursor:pointer" onclick="this.nextElementSibling.classList.toggle(\'hidden\')">`;\n      html += `${catIcon} ${cat} <span style="font-size:10px;color:var(--muted);font-family:var(--mono)">click to toggle</span></div>`;\n      html += `<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:10px">`;\n      for (const [sub, types] of Object.entries(subs)) {\n        for (const [etype, info] of Object.entries(types)) {\n          const dc = diffColors[info.difficulty] || "var(--muted)";\n          const db = diffBg[info.difficulty] || "rgba(107,114,128,0.1)";\n          const dbr = diffBorder[info.difficulty] || "rgba(107,114,128,0.3)";\n          const sc = counts[etype] || info.sample_count || 0;\n          html += `<div style="background:var(--bg);border:1px solid var(--border);border-radius:10px;padding:14px;transition:border-color 0.2s" onmouseover="this.style.borderColor=\'${dc}\'" onmouseout="this.style.borderColor=\'\'">`;\n          html += `<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">`;\n          html += `<span style="font-size:12px;font-weight:700;font-family:var(--mono);color:var(--accent2)">${etype}</span>`;\n          html += `<span style="padding:2px 8px;border-radius:12px;font-size:9px;font-weight:700;font-family:var(--mono);background:${db};color:${dc};border:1px solid ${dbr}">${info.difficulty}</span>`;\n          html += `</div>`;\n          html += `<div style="font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;font-family:var(--mono)">${sub}</div>`;\n          html += `<div style="font-size:11px;color:#d1d5db;line-height:1.5;margin-bottom:8px">${info.description}</div>`;\n          html += `<div style="background:var(--bg2);border:1px solid var(--border);border-radius:6px;padding:8px;font-size:10px;font-family:var(--mono);color:#9ca3af;margin-bottom:6px">${info.example}</div>`;\n          html += `<div style="font-size:10px;color:var(--muted);font-family:var(--mono)">${sc} sample${sc!==1?"s":""}</div>`;\n          if (info.is_clean) html += `<div style="margin-top:4px;font-size:9px;padding:2px 6px;display:inline-block;border-radius:4px;background:rgba(139,92,246,0.15);color:#a78bfa;font-family:var(--mono)">clean sample</div>`;\n          html += `</div>`;\n        }\n      }\n      html += `</div></div>`;\n    }\n    document.getElementById("taxonomy-content").innerHTML = html;\n  } catch(e) {\n    document.getElementById("taxonomy-content").innerText = "Error loading taxonomy: " + e.message;\n  }\n}\n\n// Init\nloadSample();\nupdateMetrics();\nsetInterval(updateMetrics, 30000);\n</script>\n</body>\n</html>\n'

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
            "episode_id": detector_env._episode_id,
            "error_type": detector_env._samples[max(0, detector_env._index-1)].get("error_type", "unknown") if detector_env._samples else "unknown",
            "detector_confidence": body.action.confidence,
            "detector_correct": (obs.score or 0) > 0.5,
            "generator_confidence": 0.5,
            "generator_won": (obs.score or 0) < 0.3,
            "task_id": detector_env._task_id,
            "step": detector_env._steps
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
        "themes": ["Theme 1: Multi-Agent", "Theme 3: World Modeling", "Theme 4: Self-Improvement"]
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

@app.get("/taxonomy")
def taxonomy():
    # ── Static lookup: maps error_type → metadata ──
    ERROR_TYPE_META = {
        "year_swap": {
            "category": "Factual Errors", "subcategory": "Temporal",
            "difficulty": "easy",
            "description": "Change a year by plausible amount",
            "example": "completed in 1889 → completed in 1902"
        },
        "number_swap": {
            "category": "Factual Errors", "subcategory": "Quantitative",
            "difficulty": "easy",
            "description": "Alter a quantity slightly",
            "example": "21,196 → 8,000"
        },
        "name_swap": {
            "category": "Factual Errors", "subcategory": "Entity",
            "difficulty": "medium",
            "description": "Replace person with similar name",
            "example": "Guido van Rossum → Dennis Ritchie"
        },
        "location_swap": {
            "category": "Factual Errors", "subcategory": "Entity",
            "difficulty": "medium",
            "description": "Wrong location",
            "example": "Agra → New Delhi"
        },
        "negation": {
            "category": "Logical Errors", "subcategory": "Negation",
            "difficulty": "hard",
            "description": "Add or remove a negation to flip meaning",
            "example": "was ratified → was not ratified"
        },
        "entity_flip": {
            "category": "Logical Errors", "subcategory": "Causal",
            "difficulty": "hard",
            "description": "Reverse who did what to whom",
            "example": "France gifted → America gifted"
        },
        "unit_shift": {
            "category": "Factual Errors", "subcategory": "Quantitative",
            "difficulty": "hard",
            "description": "Same number, wrong unit",
            "example": "384,400 kilometres → 384,400 metres"
        },
        "partial_truth": {
            "category": "Logical Errors", "subcategory": "Causal",
            "difficulty": "hard",
            "description": "Mostly correct, one wrong detail embedded",
            "example": "correct context, wrong specific fact"
        },
        "date_arithmetic": {
            "category": "Factual Errors", "subcategory": "Temporal",
            "difficulty": "expert",
            "description": "Multi-step date calculation error",
            "example": "Feb 28 + 60 days in leap year → wrong date"
        },
        "adversarial_clean": {
            "category": "Adversarial", "subcategory": "Clean",
            "difficulty": "expert",
            "description": "Sounds wrong but is actually correct — tests false positive resistance",
            "example": "counterintuitive but factually accurate statement",
            "is_clean": True
        },
    }

    # ── Inference rules: detect error_type from hint/content when missing ──
    def _infer_error_type(sample: dict, task_id: str) -> str:
        if sample.get("error_type"):
            return sample["error_type"]
        if sample.get("is_clean") and not sample.get("ground_truth_has_hallucination"):
            return "adversarial_clean"
        hint = (sample.get("hint") or "").lower()
        phrases = sample.get("ground_truth_hallucinated_phrases") or []
        phrase_str = " ".join(phrases).lower()
        # Year / date keywords
        if any(k in hint for k in ["year", "date", "completion year", "introduction year"]):
            if task_id == "expert" or "arithmetic" in hint or "leap" in hint:
                return "date_arithmetic"
            return "year_swap"
        # Unit keywords
        if any(k in hint for k in ["unit", "metres", "meters", "feet", "kilomet"]):
            return "unit_shift"
        # Name / person keywords
        if any(k in hint for k in ["name", "creator", "person", "astronaut", "who"]):
            if any(k in hint for k in ["who bought", "who did what", "swap"]):
                return "entity_flip"
            return "name_swap"
        # Location keywords
        if any(k in hint for k in ["city", "location", "country"]):
            return "location_swap"
        # Number / figure keywords
        if any(k in hint for k in ["figure", "number", "digit", "height", "population", "length",
                                    "layer", "area", "period", "percentage", "average"]):
            return "number_swap"
        # Negation keywords
        if any(k in hint for k in ["negat", "not ", "increase or decrease", "sufficient",
                                    "liable", "excluded", "inclusive", "exclusion"]):
            return "negation"
        # Entity flip keywords
        if any(k in hint for k in ["bought whom", "reversed", "swapped", "confused",
                                    "merged", "direction", "reactant", "product"]):
            return "entity_flip"
        # Partial truth for adversarial-tier content
        if task_id == "adversarial":
            return "partial_truth"
        # Expert multi-hop
        if task_id == "expert":
            if any(k in hint for k in ["swap", "reverse", "confuse"]):
                return "entity_flip"
            return "partial_truth"
        # Hard tier defaults
        if task_id == "hard":
            if any(k in hint for k in ["organ", "bacteria", "virus", "category"]):
                return "partial_truth"
            return "entity_flip"
        return "partial_truth"

    # ── Scan all samples dynamically ──
    sample_counts: dict = {}
    difficulty_map: dict = {}

    for task_id, samples in TASKS.items():
        for sample in samples:
            et = _infer_error_type(sample, task_id)
            sample_counts[et] = sample_counts.get(et, 0) + 1

    # Build difficulty_distribution dynamically
    for et, meta in ERROR_TYPE_META.items():
        diff = meta["difficulty"]
        if diff not in difficulty_map:
            difficulty_map[diff] = []
        if et not in difficulty_map[diff]:
            difficulty_map[diff].append(et)

    # ── Build structured taxonomy tree ──
    taxonomy_tree: dict = {}
    for et, meta in ERROR_TYPE_META.items():
        cat = meta["category"]
        sub = meta["subcategory"]
        if cat not in taxonomy_tree:
            taxonomy_tree[cat] = {}
        if sub not in taxonomy_tree[cat]:
            taxonomy_tree[cat][sub] = {}
        entry = {
            "difficulty": meta["difficulty"],
            "description": meta["description"],
            "example": meta["example"],
            "sample_count": sample_counts.get(et, 0),
        }
        if meta.get("is_clean"):
            entry["is_clean"] = True
        taxonomy_tree[cat][sub][et] = entry

    return {
        "description": "Hallucination error type taxonomy used in HalluciNet",
        "total_error_types": len(ERROR_TYPE_META),
        "total_samples_scanned": sum(len(s) for s in TASKS.values()),
        "taxonomy": taxonomy_tree,
        "difficulty_distribution": difficulty_map,
        "sample_counts": sample_counts,
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


@app.get("/tasks/summary")
def tasks_summary():
    counts = {task_id: len(samples) for task_id, samples in TASKS.items()}
    return {
        "tasks": counts,
        "total_samples": sum(counts.values()),
        "task_count": len(counts),
    }

@app.get("/")
@app.get("/demo")
def demo_ui():
    return HTMLResponse(DEMO_UI_HTML)
@app.get("/metadata")
def metadata():
    return {
        "name": "hallucinet-adversarial",
        "description": "Adversarial self-improving hallucination detection. Generator vs Detector multi-agent RL. Theme 1 + Theme 3 + Theme 4.",
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
    # Feed debate outcome back into oversight
    try:
        oversight_agent.record_episode({
            "episode_id": detector_env._episode_id,
            "error_type": "debate_round",
            "detector_confidence": 0.8,
            "detector_correct": result.get("outcome") == "detector_wins",
            "generator_confidence": result.get("generator_defense_score", 0.5),
            "generator_won": result.get("outcome") != "detector_wins",
            "task_id": detector_env._task_id,
            "step": "debate",
            "debate_delta": result.get("generator_final_reward_delta", 0)
        })
    except Exception as e:
        print(f"[DEBATE OVERSIGHT ERROR] {e}")

    return {
        "task_id": detector_env._task_id,
        "episode_id": detector_env._episode_id,
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


@app.get("/world/model")
def world_model():
    eval_data = oversight_agent.evaluate()
    curriculum_data = adversarial_curriculum.get_status()
    calibration_data = detector_calibration.get_calibration_curve()
    elo_data = elo_tracker.get_standings()
    
    # Group oversight records by episode_id
    episodes = {}
    for rec in oversight_agent.episode_history:
        eid = rec.get("episode_id", "unknown")
        if eid not in episodes:
            episodes[eid] = []
        episodes[eid].append(rec)
    
    return {
        "theme": "Theme 2: Long-Horizon Planning + Theme 3: World Modeling",
        "description": "5-step adversarial episodes with persistent state tracking",
        "multi_step_episodes": {
            "total_episodes": len(episodes),
            "steps_per_episode": {
                eid: len(steps) 
                for eid, steps in list(episodes.items())[-5:]
            },
            "episode_flow": "generator \u2192 detector \u2192 debate \u2192 oversight \u2192 curriculum"
        },
        "agent_model": {
            "detector_reliability": eval_data["reliability_score"],
            "detector_blind_spots": eval_data["blind_spots"],
            "overconfidence_rate": eval_data["overconfidence_rate"],
            "calibration_error": calibration_data["calibration_error"],
            "calibration_interpretation": calibration_data["interpretation"],
            "episodes_monitored": eval_data["episodes_monitored"]
        },
        "environment_model": {
            "current_difficulty": curriculum_data.get("current_task"),
            "detector_elo": elo_data["detector_elo"],
            "generator_elo": elo_data["generator_elo"],
            "current_leader": elo_data["current_leader"],
            "total_rounds": elo_data["total_rounds"]
        },
        "predicted_next_action": (
            "inject_adversarial_sample" if oversight_agent.should_inject_adversarial()
            else "continue_current_difficulty"
        ),
        "system_health": eval_data["system_feedback"]
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
