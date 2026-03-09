from __future__ import annotations

import json
import logging
from typing import Any

from app.autotune.schemas import CritiqueFeedback, GenerationStrategy, PocketInput
from app.autotune.utils import clamp, extract_json_object


logger = logging.getLogger(__name__)


class ScientistAgent:
    """Designs strategy (conditions + guidance) and updates it from feedback."""

    def __init__(self, *, llm: Any | None = None, use_llm: bool = False) -> None:
        self._llm = llm
        self._use_llm = use_llm and llm is not None

    async def propose_initial_strategy(self, pocket: PocketInput) -> tuple[GenerationStrategy, str]:
        if self._use_llm:
            strategy, rationale = await self._llm_propose_strategy(pocket, prev_strategy=None, feedback=None)
            return strategy, rationale

        # Heuristic defaults: moderate guidance, avoid overly hydrophobic.
        strategy = GenerationStrategy(
            guidance_mode="implicit",
            center_guidance_weight=0.65,
            edge_hbond_weight=0.60,
            hydrophobic_weight=0.52,
            aromatic_weight=0.45,
            conformation_flexibility=0.40,
            num_steps=50,
            guidance_scale=2.0,
            temperature=1.0,
        )
        rationale = (
            "基于默认启发式：先用隐式引导探索口袋，边缘氢键权重略高以鼓励极性互作，"
            "疏水性权重保持中等以避免过强疏水导致对接分数劣化。"
        )
        return strategy, rationale

    async def revise_strategy(
        self,
        *,
        pocket: PocketInput,
        prev_strategy: GenerationStrategy,
        feedback: CritiqueFeedback,
    ) -> tuple[GenerationStrategy, str]:
        if self._use_llm:
            strategy, rationale = await self._llm_propose_strategy(pocket, prev_strategy=prev_strategy, feedback=feedback)
            return strategy, rationale

        # Apply numeric deltas then clamp to safe ranges.
        data = prev_strategy.model_dump()
        for k, delta in (feedback.parameter_deltas or {}).items():
            if k in data and isinstance(data[k], (int, float)):
                data[k] = float(data[k]) + float(delta)

        # Clamp key params.
        data["center_guidance_weight"] = clamp(float(data.get("center_guidance_weight", 0.65)), 0.05, 0.95)
        data["edge_hbond_weight"] = clamp(float(data.get("edge_hbond_weight", 0.55)), 0.0, 1.0)
        data["hydrophobic_weight"] = clamp(float(data.get("hydrophobic_weight", 0.55)), 0.0, 1.0)
        data["aromatic_weight"] = clamp(float(data.get("aromatic_weight", 0.40)), 0.0, 1.0)
        data["conformation_flexibility"] = clamp(float(data.get("conformation_flexibility", 0.35)), 0.0, 1.0)

        # Keep steps/scale within reasonable bounds.
        data["num_steps"] = int(clamp(float(data.get("num_steps", 50)), 10, 200))
        data["guidance_scale"] = clamp(float(data.get("guidance_scale", 2.0)), 0.5, 10.0)
        data["temperature"] = clamp(float(data.get("temperature", 1.0)), 0.1, 2.0)

        new_strategy = GenerationStrategy(**data)
        rationale = "根据 Critic 的量化反馈应用参数增量并做边界裁剪，进入下一轮生成。"
        return new_strategy, rationale

    async def _llm_propose_strategy(
        self,
        pocket: PocketInput,
        *,
        prev_strategy: GenerationStrategy | None,
        feedback: CritiqueFeedback | None,
    ) -> tuple[GenerationStrategy, str]:
        # Designed to work with app.services.dashscope_client.DashScopeClient
        system_prompt = (
            "你是构象与条件设计专家(Scientist Agent)。你的工作是为条件扩散生成小分子制定策略，"
            "并在收到评估反馈后自主调参迭代。\n"
            "你必须只输出 JSON 对象，包含字段：\n"
            "{\n"
            "  'strategy': { ...GenerationStrategy fields... },\n"
            "  'rationale': '...中文简短理由...'\n"
            "}\n"
            "strategy 字段允许的键包括：guidance_mode(center/edge/hydrophobic/aromatic/conformation_flexibility/num_steps/guidance_scale/temperature/explicit_constraints)。\n"
            "所有 weight/flexibility 取值范围 [0,1]。docking 目标越低越好。禁止编造不存在的测量数据。"
        )

        user_payload: dict[str, Any] = {
            "pocket": pocket.model_dump(),
            "prev_strategy": prev_strategy.model_dump() if prev_strategy else None,
            "critic_feedback": feedback.model_dump() if feedback else None,
        }
        user_prompt = "请基于以下输入给出下一版策略 JSON：\n" + json.dumps(user_payload, ensure_ascii=False)

        # DashScopeClient.chat takes list[dict]
        raw = await self._llm.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=False,
        )

        obj = extract_json_object(raw)
        strat_obj = obj.get("strategy") if isinstance(obj, dict) else None
        rationale = str(obj.get("rationale") or "") if isinstance(obj, dict) else ""

        if not isinstance(strat_obj, dict):
            logger.warning("Scientist LLM returned non-strategy JSON; falling back. raw=%r", raw[:300])
            return await self.propose_initial_strategy(pocket)

        try:
            strategy = GenerationStrategy(**strat_obj)
        except Exception as e:
            logger.warning("Scientist LLM strategy parse failed; falling back: %s", repr(e))
            return await self.propose_initial_strategy(pocket)

        if not rationale:
            rationale = "LLM 生成的策略（未提供 rationale）。"
        return strategy, rationale
