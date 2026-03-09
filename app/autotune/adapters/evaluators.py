from __future__ import annotations

from abc import ABC, abstractmethod

from app.autotune.schemas import EvaluationResult, MoleculeCandidate, PocketInput


class MoleculeEvaluator(ABC):
    """Adapter interface for scoring candidates (docking, clash, etc.)."""

    @abstractmethod
    async def evaluate(
        self,
        *,
        pocket: PocketInput,
        candidate: MoleculeCandidate,
    ) -> EvaluationResult:
        raise NotImplementedError


def _hydrophobicity_proxy(smiles: str) -> float:
    if not smiles:
        return 0.0
    # Very rough proxy: fraction of carbon/aromatic carbons.
    c = smiles.count("C") + smiles.count("c")
    return c / max(1, len(smiles))


class MockCompositeEvaluator(MoleculeEvaluator):
    """A mock evaluator with a tunable landscape.

    The idea is to make strategy parameters causally affect the score:
    - Too high hydrophobicity worsens docking.
    - Higher edge_hbond_weight helps docking.
    - Too rigid (low flexibility) increases steric clashes.
    - Excessively strong center guidance can also cause clashes.

    This lets the closed-loop system demonstrate self-tuning without any
    external chemistry toolchain.
    """

    def __init__(
        self,
        *,
        steric_penalty: float = 1.2,
        hydrophobic_penalty: float = 8.0,
        hbond_bonus: float = 6.5,
    ) -> None:
        self._steric_penalty = steric_penalty
        self._hydrophobic_penalty = hydrophobic_penalty
        self._hbond_bonus = hbond_bonus

    async def evaluate(
        self,
        *,
        pocket: PocketInput,
        candidate: MoleculeCandidate,
    ) -> EvaluationResult:
        smiles = candidate.smiles
        hyd = _hydrophobicity_proxy(smiles)

        # "Docking" base: prefer moderate length, avoid too hydrophobic.
        length_term = abs(len(smiles) - 38) / 10.0
        docking = 7.0 + length_term + self._hydrophobic_penalty * max(0.0, hyd - 0.62)

        # Strategy effects only for generated candidates (baseline has None).
        strategy = candidate.strategy
        if strategy is not None:
            docking -= self._hbond_bonus * strategy.edge_hbond_weight
            docking += 2.0 * max(0.0, strategy.hydrophobic_weight - 0.65)
            docking += 1.0 * max(0.0, 0.45 - strategy.aromatic_weight)

        # Clash landscape: rigidity + too-strong center guidance.
        clash = 0
        if strategy is not None:
            rigidity = max(0.0, 0.45 - strategy.conformation_flexibility)
            center_over = max(0.0, strategy.center_guidance_weight - 0.72)
            clash = int((rigidity * 10.0) + (center_over * 18.0))

        notes: list[str] = []
        if hyd > 0.7:
            notes.append("hydrophobicity_high")
        if clash >= 6:
            notes.append("steric_clash_high")

        return EvaluationResult(
            docking_score=float(docking),
            steric_clash_count=int(clash),
            hydrophobicity_proxy=float(hyd),
            notes=notes,
        )
