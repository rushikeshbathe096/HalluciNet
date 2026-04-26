# server/oversight_agent.py
"""
Fleet-oversight agent: aggregates detector episodes from record_episode() calls
and surfaces reliability, overconfidence, and blind-spot patterns. All values are
derived from self.episode_history (no static scores).
"""

from __future__ import annotations

import sys
import os
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class OversightAgent:
    """Monitors cross-episode behavior; state lives in instance memory only."""

    def __init__(self) -> None:
        self.episode_history: List[Dict[str, Any]] = []

    def record_episode(self, episode_result: dict) -> None:
        """Append one completed-episode summary (or step-level) record."""
        self.episode_history.append(dict(episode_result))

    def detect_blind_spots(self) -> List[str]:
        """
        error_types for which the detector failed 3 or more times **consecutively**
        in chronological order (scans `error_type` + `detector_correct` on each record).
        """
        blind: set[str] = set()
        streak_error: Optional[str] = None
        streak_fails = 0

        for rec in self.episode_history:
            err = str(rec.get("error_type", "unknown"))
            ok = rec.get("detector_correct", False)
            if not ok:
                if streak_error == err:
                    streak_fails += 1
                else:
                    streak_error = err
                    streak_fails = 1
                if streak_fails >= 3:
                    blind.add(err)
            else:
                streak_error = None
                streak_fails = 0

        return sorted(blind)

    def evaluate(self) -> dict:
        """Dynamic metrics from episode_history."""
        total = len(self.episode_history)
        if total == 0:
            return {
                "reliability_score": 1.0,
                "overconfidence_rate": 0.0,
                "blind_spots": [],
                "system_feedback": "No episode data yet — run detector episodes to populate oversight.",
                "episodes_monitored": 0,
                "fleet_ai_bonus": True,
            }

        overconfident_wrong: List[Dict[str, Any]] = [
            r
            for r in self.episode_history
            if float(r.get("detector_confidence", 0.0)) > 0.8
            and not r.get("detector_correct", True)
        ]
        ovw_n = len(overconfident_wrong)
        reliability = 1.0 - (ovw_n / max(total, 1))
        ovw_rate = ovw_n / max(total, 1)
        blind = self.detect_blind_spots()

        if reliability > 0.8:
            feedback = "System operating within normal reliability based on recent episodes."
        elif reliability > 0.6:
            feedback = (
                f"Moderate reliability (score {reliability:.2f}). "
                f"Watch overconfidence when wrong — rate {ovw_rate:.2f}."
            )
        else:
            feedback = (
                f"Reliability is low (score {reliability:.2f}). "
                f"Overconfidence while incorrect: {ovw_rate:.2f}. "
                f"Blind spot types: {blind or '—'}."
            )

        return {
            "reliability_score": round(reliability, 4),
            "overconfidence_rate": round(ovw_rate, 4),
            "blind_spots": blind,
            "system_feedback": feedback,
            "episodes_monitored": total,
            "fleet_ai_bonus": True,
        }

    def should_inject_adversarial(self) -> bool:
        if len(self.episode_history) < 3:
            return False
        last_three = self.episode_history[-3:]
        return all(
            float(r.get("detector_confidence", 0)) > 0.7
            and not r.get("detector_correct", True)
            for r in last_three
        )

    def reset(self) -> None:
        self.episode_history = []
