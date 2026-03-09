from __future__ import annotations

import hashlib
import random
from abc import ABC, abstractmethod

from app.autotune.schemas import MoleculeCandidate, PocketInput


class DiffGuiBaselineRunner(ABC):
    """Adapter interface to generate DiffGui baseline candidates."""

    @abstractmethod
    async def generate_baseline(
        self,
        *,
        pocket: PocketInput,
        n: int,
        iteration: int,
    ) -> list[MoleculeCandidate]:
        raise NotImplementedError


class MockDiffGuiBaselineRunner(DiffGuiBaselineRunner):
    """Mock DiffGui baseline.

    Deterministic and intentionally "ok but not great" so the loop can
    surpass it under the mock evaluator.
    """

    def __init__(self, seed: int = 1337) -> None:
        self._seed = seed

    async def generate_baseline(
        self,
        *,
        pocket: PocketInput,
        n: int,
        iteration: int,
    ) -> list[MoleculeCandidate]:
        key = f"diffgui|{self._seed}|{pocket.target_id}|{pocket.pocket_id}|{iteration}"
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        base_seed = int(digest[:8], 16)
        rng = random.Random(base_seed)

        candidates: list[MoleculeCandidate] = []
        for idx in range(n):
            length = rng.randint(20, 40)
            # Baseline slightly more hydrophobic and less polar.
            chars = ["C" if rng.random() < 0.85 else "c" for _ in range(length)]
            if rng.random() < 0.4:
                chars.extend(["N"] * rng.randint(0, 2))
            if rng.random() < 0.3:
                chars.extend(["O"] * rng.randint(0, 2))
            smiles = "".join(chars)
            candidates.append(
                MoleculeCandidate(
                    candidate_id=f"base_{iteration}_{idx}",
                    smiles=smiles,
                    source="baseline",
                    iteration=iteration,
                    strategy=None,
                    extra={"mock": True, "seed": base_seed},
                )
            )

        return candidates
