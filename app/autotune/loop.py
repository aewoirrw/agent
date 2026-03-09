from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from app.autotune.agents.critic import CriticAgent
from app.autotune.agents.executor import ExecutorAgent
from app.autotune.agents.scientist import ScientistAgent
from app.autotune.schemas import (
    CritiqueFeedback,
    GenerationStrategy,
    IterationRecord,
    LoopConfig,
    PocketInput,
    ScoredCandidate,
)


logger = logging.getLogger(__name__)


@dataclass
class LoopResult:
    best_candidate: ScoredCandidate | None
    best_strategy: GenerationStrategy | None
    history: list[IterationRecord]
    stop_reason: str


class AutotuneLoop:
    """Closed-loop: Scientist -> Executor -> Critic -> Scientist ..."""

    def __init__(
        self,
        *,
        scientist: ScientistAgent,
        executor: ExecutorAgent,
        critic: CriticAgent,
        config: LoopConfig | None = None,
    ) -> None:
        self._scientist = scientist
        self._executor = executor
        self._critic = critic
        self._config = config or LoopConfig()

    async def run(
        self,
        *,
        pocket: PocketInput,
        out_jsonl: str | Path | None = None,
    ) -> LoopResult:
        cfg = self._config

        history: list[IterationRecord] = []
        best_overall: ScoredCandidate | None = None
        best_strategy: GenerationStrategy | None = None
        no_improve_rounds = 0
        total_candidates = 0

        strategy, _ = await self._scientist.propose_initial_strategy(pocket)

        out_path = Path(out_jsonl) if out_jsonl else None
        if out_path:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if out_path.exists():
                out_path.unlink()

        stop_reason = "max_iterations"

        for iteration in range(1, cfg.max_iterations + 1):
            n_gen = min(cfg.candidates_per_iteration, cfg.max_total_candidates - total_candidates)
            if n_gen <= 0:
                stop_reason = "max_total_candidates"
                break

            generated = await self._executor.run_generation(
                pocket=pocket,
                strategy=strategy,
                n=n_gen,
                iteration=iteration,
            )
            total_candidates += len(generated)

            baseline_raw = await self._critic.run_baseline(
                pocket=pocket,
                n=cfg.baseline_candidates,
                iteration=iteration,
            )

            scored_generated = await self._critic.score_candidates(pocket=pocket, candidates=generated)
            scored_baseline = await self._critic.score_candidates(pocket=pocket, candidates=baseline_raw)

            best_gen = scored_generated[0] if scored_generated else None
            best_base = scored_baseline[0] if scored_baseline else None

            improvement_margin = 0.0
            improved = False
            if best_gen and best_base:
                improvement_margin = float(best_base.total_score - best_gen.total_score)
                improved = improvement_margin >= cfg.min_improvement_margin

            critic_feedback = await self._critic.critique(
                pocket=pocket,
                strategy=strategy,
                scored_generated=scored_generated,
                scored_baseline=scored_baseline,
                min_improvement_margin=cfg.min_improvement_margin,
            )

            record = IterationRecord(
                iteration=iteration,
                pocket=pocket,
                strategy=strategy,
                generated=scored_generated,
                baseline=scored_baseline,
                best_generated=best_gen,
                best_baseline=best_base,
                improved_over_baseline=improved,
                improvement_margin=improvement_margin,
                critic_feedback=critic_feedback,
            )
            history.append(record)

            if out_path:
                out_path.write_text(
                    "\n".join(json.dumps(r.model_dump(), ensure_ascii=False) for r in history) + "\n",
                    encoding="utf-8",
                )

            # Update best.
            if best_gen and (best_overall is None or best_gen.total_score < best_overall.total_score):
                best_overall = best_gen
                best_strategy = strategy

            if improved:
                stop_reason = "improved_over_baseline"
                break

            if isinstance(critic_feedback, CritiqueFeedback) and critic_feedback.should_stop:
                stop_reason = critic_feedback.stop_reason or "critic_stop"
                break

            # Patience-based stopping.
            if improvement_margin > 0:
                no_improve_rounds = 0
            else:
                no_improve_rounds += 1
                if no_improve_rounds >= cfg.patience:
                    stop_reason = "patience"
                    break

            # Ask Scientist to revise.
            strategy, _ = await self._scientist.revise_strategy(
                pocket=pocket,
                prev_strategy=strategy,
                feedback=critic_feedback,
            )

        return LoopResult(
            best_candidate=best_overall,
            best_strategy=best_strategy,
            history=history,
            stop_reason=stop_reason,
        )
