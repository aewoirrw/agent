from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from app.autotune.adapters import MockCompositeEvaluator, MockDiffGuiBaselineRunner, MockDiffusionModel
from app.autotune.agents import CriticAgent, ExecutorAgent, ScientistAgent
from app.autotune.loop import AutotuneLoop
from app.autotune.schemas import LoopConfig, PocketInput
from app.services.dashscope_client import DashScopeClient


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Scientist-Executor-Critic closed-loop autotuning (mock adapters by default).")
    p.add_argument("--pocket", type=str, required=True, help="Path to pocket JSON (PocketInput)")
    p.add_argument("--out", type=str, default="autotune_outputs/loop_history.jsonl", help="Output JSONL path")
    p.add_argument("--iters", type=int, default=8)
    p.add_argument("--n", type=int, default=24, help="Generated candidates per iteration")
    p.add_argument("--baseline-n", type=int, default=24)
    p.add_argument("--margin", type=float, default=0.25, help="Min improvement margin vs baseline")
    p.add_argument("--patience", type=int, default=3)

    p.add_argument("--use-llm", action="store_true", help="Use DashScope LLM to draft strategy/critic feedback (optional)")
    return p.parse_args()


async def main() -> None:
    args = _parse_args()

    pocket_path = Path(args.pocket)
    pocket = PocketInput(**json.loads(pocket_path.read_text(encoding="utf-8")))

    dashscope = DashScopeClient()
    use_llm = bool(args.use_llm and dashscope.enabled)

    scientist = ScientistAgent(llm=dashscope, use_llm=use_llm)
    executor = ExecutorAgent(model=MockDiffusionModel(seed=0))
    critic = CriticAgent(
        evaluator=MockCompositeEvaluator(),
        baseline_runner=MockDiffGuiBaselineRunner(seed=1337),
        llm=dashscope,
        use_llm=use_llm,
    )

    cfg = LoopConfig(
        max_iterations=int(args.iters),
        candidates_per_iteration=int(args.n),
        baseline_candidates=int(args.baseline_n),
        min_improvement_margin=float(args.margin),
        patience=int(args.patience),
    )

    loop = AutotuneLoop(scientist=scientist, executor=executor, critic=critic, config=cfg)
    result = await loop.run(pocket=pocket, out_jsonl=args.out)

    print("stop_reason:", result.stop_reason)
    if result.best_candidate:
        print("best_total_score:", f"{result.best_candidate.total_score:.3f}")
        print("best_smiles:", result.best_candidate.candidate.smiles[:120])
        print("best_eval:", result.best_candidate.evaluation.model_dump())
    print("history_len:", len(result.history))
    print("history_jsonl:", str(Path(args.out).resolve()))


if __name__ == "__main__":
    asyncio.run(main())
