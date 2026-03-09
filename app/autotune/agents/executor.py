from __future__ import annotations

from app.autotune.adapters.diffusion import ConditionalDiffusionModel
from app.autotune.schemas import GenerationStrategy, MoleculeCandidate, PocketInput


class ExecutorAgent:
    """Executes generation strictly according to Scientist's strategy."""

    def __init__(self, *, model: ConditionalDiffusionModel) -> None:
        self._model = model

    async def run_generation(
        self,
        *,
        pocket: PocketInput,
        strategy: GenerationStrategy,
        n: int,
        iteration: int,
    ) -> list[MoleculeCandidate]:
        return await self._model.generate(pocket=pocket, strategy=strategy, n=n, iteration=iteration)
