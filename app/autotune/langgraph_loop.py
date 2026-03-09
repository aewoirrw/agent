from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, TypedDict, Union, cast

from app.autotune.agents.critic import CriticAgent
from app.autotune.agents.executor import ExecutorAgent
from app.autotune.agents.scientist import ScientistAgent
from app.autotune.loop import LoopResult
from app.autotune.schemas import (
    CritiqueFeedback,
    GenerationStrategy,
    IterationRecord,
    LoopConfig,
    PocketInput,
    ScoredCandidate,
)


logger = logging.getLogger(__name__)


try:
    from langgraph.graph import END, StateGraph  # type: ignore
except Exception:  # pragma: no cover
    END = cast(Any, "__END__")
    StateGraph = None  # type: ignore


class _GraphState(TypedDict, total=False):
    pocket: PocketInput
    config: LoopConfig
    out_path: Optional[Path]

    iteration: int
    total_candidates: int
    no_improve_rounds: int

    strategy: GenerationStrategy

    generated_raw: list[Any]
    baseline_raw: list[Any]

    scored_generated: list[ScoredCandidate]
    scored_baseline: list[ScoredCandidate]

    best_overall: Optional[ScoredCandidate]
    best_strategy: Optional[GenerationStrategy]

    critic_feedback: CritiqueFeedback
    stop_reason: str

    history: list[IterationRecord]


@dataclass
class LangGraphAutotuneLoop:
    """LangGraph-based closed loop (Scientist -> Executor -> Critic -> ...).

    This is an alternative orchestrator to `AutotuneLoop`. It uses LangGraph
    to represent the feedback loop as an explicit state machine.
    """

    scientist: ScientistAgent
    executor: ExecutorAgent
    critic: CriticAgent
    config: LoopConfig | None = None

    def __post_init__(self) -> None:
        if StateGraph is None:
            raise ImportError(
                "langgraph 未安装：请在 python_app 环境中执行 `pip install langgraph`（或重新安装项目依赖）。"
            )

    async def run(self, *, pocket: PocketInput, out_jsonl: Union[str, Path, None] = None) -> LoopResult:
        cfg = self.config or LoopConfig()
        out_path = Path(out_jsonl) if out_jsonl else None
        if out_path:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if out_path.exists():
                out_path.unlink()

        graph = self._build_graph()

        initial_state: _GraphState = {
            "pocket": pocket,
            "config": cfg,
            "out_path": out_path,
            "iteration": 0,
            "total_candidates": 0,
            "no_improve_rounds": 0,
            "history": [],
            "best_overall": None,
            "best_strategy": None,
            "stop_reason": "max_iterations",
        }

        final_state = cast(_GraphState, await graph.ainvoke(initial_state))

        return LoopResult(
            best_candidate=final_state.get("best_overall"),
            best_strategy=final_state.get("best_strategy"),
            history=final_state.get("history", []),
            stop_reason=str(final_state.get("stop_reason") or "unknown"),
        )

    def _build_graph(self):
        sg = StateGraph(_GraphState)  # type: ignore[arg-type]

        sg.add_node("init", self._node_init)
        sg.add_node("generate", self._node_generate)
        sg.add_node("baseline", self._node_baseline)
        sg.add_node("score", self._node_score)
        sg.add_node("critique", self._node_critique)
        sg.add_node("record", self._node_record)
        sg.add_node("decide", self._node_decide)
        sg.add_node("revise", self._node_revise)

        sg.set_entry_point("init")
        sg.add_edge("init", "generate")
        sg.add_edge("generate", "baseline")
        sg.add_edge("baseline", "score")
        sg.add_edge("score", "critique")
        sg.add_edge("critique", "record")
        sg.add_edge("record", "decide")

        def _route(state: _GraphState) -> Union[Literal["revise"], Any]:
            reason = str(state.get("stop_reason") or "")
            if reason and reason != "continue":
                return END
            return "revise"

        sg.add_conditional_edges("decide", _route, {"revise": "revise", END: END})
        sg.add_edge("revise", "generate")

        return sg.compile()

    async def _node_init(self, state: _GraphState) -> _GraphState:
        pocket = state["pocket"]
        strategy, _rationale = await self.scientist.propose_initial_strategy(pocket)
        return {
            **state,
            "strategy": strategy,
            "iteration": 1,
            "stop_reason": "continue",
        }

    async def _node_generate(self, state: _GraphState) -> _GraphState:
        cfg = state["config"]
        pocket = state["pocket"]
        strategy = state["strategy"]
        iteration = int(state["iteration"])
        total_candidates = int(state.get("total_candidates", 0))

        n_gen = min(int(cfg.candidates_per_iteration), int(cfg.max_total_candidates) - total_candidates)
        if n_gen <= 0:
            return {**state, "stop_reason": "max_total_candidates"}

        generated = await self.executor.run_generation(
            pocket=pocket,
            strategy=strategy,
            n=n_gen,
            iteration=iteration,
        )

        return {
            **state,
            "generated_raw": generated,
            "total_candidates": total_candidates + len(generated),
        }

    async def _node_baseline(self, state: _GraphState) -> _GraphState:
        cfg = state["config"]
        pocket = state["pocket"]
        iteration = int(state["iteration"])

        baseline_raw = await self.critic.run_baseline(
            pocket=pocket,
            n=int(cfg.baseline_candidates),
            iteration=iteration,
        )
        return {**state, "baseline_raw": baseline_raw}

    async def _node_score(self, state: _GraphState) -> _GraphState:
        pocket = state["pocket"]
        generated_raw = cast(list, state.get("generated_raw") or [])
        baseline_raw = cast(list, state.get("baseline_raw") or [])

        scored_generated = await self.critic.score_candidates(pocket=pocket, candidates=generated_raw)
        scored_baseline = await self.critic.score_candidates(pocket=pocket, candidates=baseline_raw)

        return {
            **state,
            "scored_generated": scored_generated,
            "scored_baseline": scored_baseline,
        }

    async def _node_critique(self, state: _GraphState) -> _GraphState:
        cfg = state["config"]
        pocket = state["pocket"]
        strategy = state["strategy"]
        scored_generated = state.get("scored_generated", [])
        scored_baseline = state.get("scored_baseline", [])

        feedback = await self.critic.critique(
            pocket=pocket,
            strategy=strategy,
            scored_generated=scored_generated,
            scored_baseline=scored_baseline,
            min_improvement_margin=float(cfg.min_improvement_margin),
        )
        return {**state, "critic_feedback": feedback}

    async def _node_record(self, state: _GraphState) -> _GraphState:
        cfg = state["config"]
        pocket = state["pocket"]
        strategy = state["strategy"]
        iteration = int(state["iteration"])
        scored_generated = state.get("scored_generated", [])
        scored_baseline = state.get("scored_baseline", [])
        feedback = state["critic_feedback"]

        best_gen = scored_generated[0] if scored_generated else None
        best_base = scored_baseline[0] if scored_baseline else None

        improvement_margin = 0.0
        improved = False
        if best_gen and best_base:
            improvement_margin = float(best_base.total_score - best_gen.total_score)
            improved = improvement_margin >= float(cfg.min_improvement_margin)

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
            critic_feedback=feedback,
        )

        history = list(state.get("history", []))
        history.append(record)

        out_path = state.get("out_path")
        if out_path:
            out_path.write_text(
                "\n".join(json.dumps(r.model_dump(), ensure_ascii=False) for r in history) + "\n",
                encoding="utf-8",
            )

        best_overall = state.get("best_overall")
        best_strategy = state.get("best_strategy")
        if best_gen and (best_overall is None or best_gen.total_score < best_overall.total_score):
            best_overall = best_gen
            best_strategy = strategy

        return {
            **state,
            "history": history,
            "best_overall": best_overall,
            "best_strategy": best_strategy,
        }

    async def _node_decide(self, state: _GraphState) -> _GraphState:
        cfg = state["config"]
        iteration = int(state["iteration"])
        no_improve_rounds = int(state.get("no_improve_rounds", 0))

        history = state.get("history", [])
        last = history[-1] if history else None

        if last is None:
            return {**state, "stop_reason": "empty_history"}

        # Stop conditions
        if last.improved_over_baseline:
            return {**state, "stop_reason": "improved_over_baseline"}

        feedback = state.get("critic_feedback")
        if isinstance(feedback, CritiqueFeedback) and feedback.should_stop:
            return {**state, "stop_reason": str(feedback.stop_reason or "critic_stop")}

        if iteration >= int(cfg.max_iterations):
            return {**state, "stop_reason": "max_iterations"}

        # Patience logic: treat positive margin as "improving"
        if float(last.improvement_margin) > 0:
            no_improve_rounds = 0
        else:
            no_improve_rounds += 1
            if no_improve_rounds >= int(cfg.patience):
                return {**state, "no_improve_rounds": no_improve_rounds, "stop_reason": "patience"}

        return {**state, "no_improve_rounds": no_improve_rounds, "stop_reason": "continue"}

    async def _node_revise(self, state: _GraphState) -> _GraphState:
        pocket = state["pocket"]
        prev_strategy = state["strategy"]
        feedback = state["critic_feedback"]
        iteration = int(state["iteration"])

        new_strategy, _rationale = await self.scientist.revise_strategy(
            pocket=pocket,
            prev_strategy=prev_strategy,
            feedback=feedback,
        )

        return {
            **state,
            "strategy": new_strategy,
            "iteration": iteration + 1,
        }
