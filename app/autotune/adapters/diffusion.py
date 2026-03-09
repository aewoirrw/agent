from __future__ import annotations

import hashlib
import random
from abc import ABC, abstractmethod

from app.autotune.schemas import GenerationStrategy, MoleculeCandidate, PocketInput


class ConditionalDiffusionModel(ABC):
    """Adapter interface for a conditional diffusion generator."""

    @abstractmethod
    async def generate(
        self,
        *,
        pocket: PocketInput,
        strategy: GenerationStrategy,
        n: int,
        iteration: int,
    ) -> list[MoleculeCandidate]:
        raise NotImplementedError


class MockDiffusionModel(ConditionalDiffusionModel):
    """Deterministic mock generator.

    Generates fake SMILES-like strings to validate the closed loop.
    The distribution shifts slightly with strategy weights so you can
    see the loop "improve" under the mock evaluator.
    """

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed

    async def generate(
        self,
        *,
        pocket: PocketInput,
        strategy: GenerationStrategy,
        n: int,
        iteration: int,
    ) -> list[MoleculeCandidate]:
        # Make output stable across runs given (seed, pocket, iteration).
        key = f"{self._seed}|{pocket.target_id}|{pocket.pocket_id}|{iteration}"
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        base_seed = int(digest[:8], 16)

        rng = random.Random(base_seed)

        # Simple mapping: more hydrophobic_weight -> more 'C'; more edge_hbond_weight -> more 'N'/'O'
        c_bias = 2 + int(6 * max(0.0, strategy.hydrophobic_weight))
        hbond_bias = 1 + int(5 * max(0.0, strategy.edge_hbond_weight))
        aromatic_bias = 1 + int(4 * max(0.0, strategy.aromatic_weight))

        candidates: list[MoleculeCandidate] = []
        for idx in range(n):
            # Create a fake linear "SMILES"
            length = rng.randint(18, 42)
            chars: list[str] = []
            for _ in range(length):
                r = rng.random()
                if r < 0.45:
                    chars.append("C" if rng.random() < 0.75 else "c")
                elif r < 0.45 + 0.20:
                    chars.append("N")
                elif r < 0.45 + 0.20 + 0.15:
                    chars.append("O")
                else:
                    chars.append("c" if rng.random() < 0.6 else "C")

            # Inject bias by appending blocks
            chars.extend(["C"] * c_bias)
            chars.extend(["N", "O"] * max(0, hbond_bias // 2))
            chars.extend(["c"] * aromatic_bias)

            smiles = "".join(chars)
            candidate_id = f"gen_{iteration}_{idx}"
            candidates.append(
                MoleculeCandidate(
                    candidate_id=candidate_id,
                    smiles=smiles,
                    source="generated",
                    iteration=iteration,
                    strategy=strategy,
                    extra={
                        "mock": True,
                        "seed": base_seed,
                    },
                )
            )

        return candidates
