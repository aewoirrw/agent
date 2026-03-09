from __future__ import annotations

import json
import logging
from typing import Any

from app.autotune.adapters.diffgui import DiffGuiBaselineRunner
from app.autotune.adapters.evaluators import MoleculeEvaluator
from app.autotune.schemas import (
    CritiqueFeedback,
    GenerationStrategy,
    MoleculeCandidate,
    PocketInput,
    ScoredCandidate,
)


logger = logging.getLogger(__name__)


class CriticAgent:
    """Evaluates generated molecules, compares to baseline, and returns feedback."""

    def __init__(
        self,
        *,
        evaluator: MoleculeEvaluator,
        baseline_runner: DiffGuiBaselineRunner,
        llm: Any | None = None,
        use_llm: bool = False,
    ) -> None:
        self._evaluator = evaluator
        self._baseline_runner = baseline_runner
        self._llm = llm
        self._use_llm = use_llm and llm is not None

    async def score_candidates(
        self,
        *,
        pocket: PocketInput,
        candidates: list[MoleculeCandidate],
    ) -> list[ScoredCandidate]:
        scored: list[ScoredCandidate] = []
        for c in candidates:
            ev = await self._evaluator.evaluate(pocket=pocket, candidate=c)
            total = float(ev.docking_score) + 1.2 * float(ev.steric_clash_count)
            scored.append(ScoredCandidate(candidate=c, evaluation=ev, total_score=total))
        scored.sort(key=lambda x: x.total_score)
        return scored

    async def run_baseline(
        self,
        *,
        pocket: PocketInput,
        n: int,
        iteration: int,
    ) -> list[MoleculeCandidate]:
        return await self._baseline_runner.generate_baseline(pocket=pocket, n=n, iteration=iteration)

    async def critique(
        self,
        *,
        pocket: PocketInput,
        strategy: GenerationStrategy,
        scored_generated: list[ScoredCandidate],
        scored_baseline: list[ScoredCandidate],
        min_improvement_margin: float,
    ) -> CritiqueFeedback:
        best_gen = scored_generated[0] if scored_generated else None
        best_base = scored_baseline[0] if scored_baseline else None

        if best_gen is None or best_base is None:
            return CritiqueFeedback(
                summary="生成或基线结果为空，无法评估；请检查生成/基线适配器。",
                suggestions=["确认生成器与 DiffGui 基线均能输出候选分子"],
                parameter_deltas={"temperature": 0.1},
                should_stop=True,
                stop_reason="empty_candidates",
            )

        margin = float(best_base.total_score - best_gen.total_score)
        improved = margin >= float(min_improvement_margin)

        if improved:
            summary = (
                f"本轮最优生成分子优于 DiffGui 基线，提升幅度 {margin:.3f}（越大越好）。"
                "可以停止迭代并输出最终结果。"
            )
            return CritiqueFeedback(summary=summary, suggestions=[], parameter_deltas={}, should_stop=True, stop_reason="improved")

        # Not improved: generate strict feedback.
        gen_ev = best_gen.evaluation
        base_ev = best_base.evaluation

        deltas: dict[str, float] = {}
        suggestions: list[str] = []

        # Rules based on evaluation.
        if gen_ev.steric_clash_count >= 6:
            suggestions.append("位阻冲突偏多：降低中心区域引导权重，释放更多构象空间")
            deltas["center_guidance_weight"] = -0.10
            deltas["conformation_flexibility"] = +0.12

        if gen_ev.hydrophobicity_proxy > 0.70:
            suggestions.append("整体疏水性偏强：降低疏水性引导权重，提升边缘氢键利用")
            deltas["hydrophobic_weight"] = -0.10
            deltas["edge_hbond_weight"] = +0.08

        # If docking still poor relative to baseline, push hbond weight.
        if gen_ev.docking_score >= base_ev.docking_score - 0.1:
            suggestions.append("对接分数未明显改善：提高边缘氢键权重，适度降低刚性约束")
            deltas.setdefault("edge_hbond_weight", 0.0)
            deltas["edge_hbond_weight"] += 0.06
            deltas.setdefault("conformation_flexibility", 0.0)
            deltas["conformation_flexibility"] += 0.06

        # If no obvious issue, adjust exploration.
        if not suggestions:
            suggestions.append("未观察到明确问题：增加采样探索（temperature 上调）并轻微降低 guidance_scale")
            deltas["temperature"] = +0.10
            deltas["guidance_scale"] = -0.20

        summary = (
            "生成分子未超过 DiffGui 基线。"
            f"当前 best_generated 总分 {best_gen.total_score:.3f} / best_baseline 总分 {best_base.total_score:.3f}。"
        )

        if self._use_llm:
            try:
                nl = await self._llm_refine_feedback(
                    pocket=pocket,
                    strategy=strategy,
                    best_gen=best_gen,
                    best_base=best_base,
                    suggestions=suggestions,
                    deltas=deltas,
                )
                summary = nl
            except Exception as e:
                logger.warning("Critic LLM refine failed: %s", repr(e))

        return CritiqueFeedback(summary=summary, suggestions=suggestions, parameter_deltas=deltas, should_stop=False)

    async def _llm_refine_feedback(
        self,
        *,
        pocket: PocketInput,
        strategy: GenerationStrategy,
        best_gen: ScoredCandidate,
        best_base: ScoredCandidate,
        suggestions: list[str],
        deltas: dict[str, float],
    ) -> str:
        system_prompt = (
            "你是苛刻的评估裁判(Critic Agent)。你要把量化指标转成严格、具体的中文反馈，"
            "并明确指出下一轮应该如何调参（给出理由）。禁止编造不存在的实验。输出只要一段话。"
        )

        user_payload = {
            "pocket": pocket.model_dump(),
            "strategy": strategy.model_dump(),
            "best_generated": {
                "total_score": best_gen.total_score,
                "evaluation": best_gen.evaluation.model_dump(),
            },
            "best_baseline": {
                "total_score": best_base.total_score,
                "evaluation": best_base.evaluation.model_dump(),
            },
            "suggestions": suggestions,
            "parameter_deltas": deltas,
        }
        user_prompt = "请基于输入写出 Critic 反馈：\n" + json.dumps(user_payload, ensure_ascii=False)

        text = await self._llm.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=False,
        )
        return str(text or "").strip()
