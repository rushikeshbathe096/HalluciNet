# client.py
import httpx
from dataclasses import dataclass
from typing import Optional
from models import HallucinationAction, HallucinationObservation, HallucinationState


@dataclass
class StepResult:
    observation: HallucinationObservation
    reward: Optional[float]
    done: bool


class HallucinationEnvClient:
    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")
        self._client = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(timeout=60.0)
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()

    def _parse_response(self, data: dict) -> StepResult:
        obs_data = data.get("observation", data)
        obs = HallucinationObservation(
            done=obs_data.get("done", False),
            reward=obs_data.get("reward"),
            task_id=obs_data.get("task_id", ""),
            sample_index=obs_data.get("sample_index", 0),
            total_samples=obs_data.get("total_samples", 0),
            reference_document=obs_data.get("reference_document", ""),
            llm_response=obs_data.get("llm_response", ""),
            feedback=obs_data.get("feedback"),
            score=obs_data.get("score", 0.0),
            steps_taken=obs_data.get("steps_taken", 0),
            max_steps=obs_data.get("max_steps", 10),
            metadata=obs_data.get("metadata", {})
        )
        return StepResult(
            observation=obs,
            reward=data.get("reward", obs.reward),
            done=data.get("done", obs.done)
        )

    async def reset(self, task_id: str = "easy") -> StepResult:
        resp = await self._client.post(
            f"{self.base_url}/reset",
            json={"task_id": task_id}
        )
        resp.raise_for_status()
        return self._parse_response(resp.json())

    async def step(self, action: HallucinationAction) -> StepResult:
        resp = await self._client.post(
            f"{self.base_url}/step",
            json={"action": action.model_dump()}
        )
        resp.raise_for_status()
        return self._parse_response(resp.json())

    async def state(self) -> HallucinationState:
        resp = await self._client.get(f"{self.base_url}/state")
        resp.raise_for_status()
        data = resp.json()
        return HallucinationState(**data)