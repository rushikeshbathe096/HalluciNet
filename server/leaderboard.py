# server/leaderboard.py
"""
Persistent per-model scores for GET /leaderboard. Data comes only from
recorded results (leaderboard.json), never from hardcoded benchmark tables.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

LEADERBOARD_PATH = os.path.join(_ROOT, "leaderboard.json")

TASK_KEYS = ("easy", "medium", "hard", "expert", "adversarial")


class Leaderboard:
    """In-memory + JSON file storage for model_name -> per-task scores."""

    def __init__(self, path: str = LEADERBOARD_PATH) -> None:
        self.path = path
        # model_name -> {"trained": bool, "tasks": {task_id: float}}
        self.results: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not os.path.isfile(self.path):
            self.results = {}
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self.results = data
            else:
                self.results = {}
        except (json.JSONDecodeError, OSError):
            self.results = {}

    def _save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, sort_keys=True)

    def record_result(
        self,
        model_name: str,
        task_id: str,
        score: float,
        trained: bool = False,
    ) -> None:
        if model_name not in self.results:
            self.results[model_name] = {"trained": trained, "tasks": {}}
        self.results[model_name]["trained"] = bool(trained)
        self.results[model_name].setdefault("tasks", {})[str(task_id)] = float(score)
        self._save()

    def get_or_default(self, model_name: str, task_id: str) -> float:
        m = self.results.get(model_name, {})
        tasks = m.get("tasks", {}) if isinstance(m, dict) else {}
        return float(tasks.get(str(task_id), 0.0))

    def get_leaderboard(self) -> List[Dict[str, Any]]:
        """Sorted by **overall** (mean of recorded task scores), best first."""
        rows: List[Dict[str, Any]] = []
        for name, rec in self.results.items():
            tasks = rec.get("tasks", {}) if isinstance(rec, dict) else {}
            vals: List[float] = []
            row: Dict[str, Any] = {
                "model": name,
                "trained": bool(rec.get("trained", False)),
            }
            for k in TASK_KEYS:
                v = float(tasks.get(k, 0.0)) if k in tasks else 0.0
                row[k] = round(v, 4)
                vals.append(v)
            # overall: mean over task keys (including zeros for missing = conservative)
            overall = sum(vals) / max(len(TASK_KEYS), 1) if vals else 0.0
            row["overall"] = round(overall, 4)
            rows.append(row)
        rows.sort(key=lambda r: r["overall"], reverse=True)
        for i, r in enumerate(rows, start=1):
            r["rank"] = i
        return rows
