from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, ConfigDict


GuidanceMode = Literal["implicit", "explicit", "hybrid"]


class PocketInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    target_id: str = Field(default="", description="Target identifier (e.g., PDBbind id)")
    pocket_id: str = Field(default="", description="Pocket identifier")
    pocket_description: str = Field(default="", description="Free-form pocket description")

    # Optional additional signals
    residues: list[str] = Field(default_factory=list, description="Residues near pocket")
    center_xyz: list[float] | None = Field(default=None, description="Pocket center (x,y,z)")


class GenerationStrategy(BaseModel):
    model_config = ConfigDict(extra="ignore")

    guidance_mode: GuidanceMode = "implicit"

    # High-level weights in [0, 1]
    center_guidance_weight: float = 0.65
    edge_hbond_weight: float = 0.55
    hydrophobic_weight: float = 0.55
    aromatic_weight: float = 0.40

    # "Release" conformation / exploration vs constraint
    conformation_flexibility: float = 0.35

    # Explicit constraint placeholders (SMARTS / pharmacophore constraints)
    explicit_constraints: list[str] = Field(default_factory=list)

    # Diffusion params (kept generic so you can map to your model)
    num_steps: int = 50
    guidance_scale: float = 2.0
    temperature: float = 1.0


class MoleculeCandidate(BaseModel):
    model_config = ConfigDict(extra="ignore")

    candidate_id: str
    smiles: str
    source: Literal["generated", "baseline"] = "generated"
    iteration: int = 0
    strategy: GenerationStrategy | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class EvaluationResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    docking_score: float = Field(description="Lower is better")
    steric_clash_count: int = 0
    hydrophobicity_proxy: float = 0.0
    notes: list[str] = Field(default_factory=list)


class ScoredCandidate(BaseModel):
    model_config = ConfigDict(extra="ignore")

    candidate: MoleculeCandidate
    evaluation: EvaluationResult
    total_score: float = Field(description="Lower is better")


class CritiqueFeedback(BaseModel):
    model_config = ConfigDict(extra="ignore")

    summary: str
    suggestions: list[str] = Field(default_factory=list)

    # Parameter deltas to apply to next strategy (simple numeric deltas)
    parameter_deltas: dict[str, float] = Field(
        default_factory=dict,
        description="Mapping of strategy fields to additive deltas, e.g. {'center_guidance_weight': -0.1}",
    )

    should_stop: bool = False
    stop_reason: str | None = None


class IterationRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    iteration: int
    pocket: PocketInput
    strategy: GenerationStrategy

    generated: list[ScoredCandidate]
    baseline: list[ScoredCandidate]

    best_generated: ScoredCandidate | None = None
    best_baseline: ScoredCandidate | None = None

    improved_over_baseline: bool = False
    improvement_margin: float = 0.0

    critic_feedback: CritiqueFeedback


class LoopConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    max_iterations: int = 8
    candidates_per_iteration: int = 24
    baseline_candidates: int = 24

    # Improvement definition: best_generated must beat best_baseline by >= margin
    min_improvement_margin: float = 0.25

    # Stop if no improvement for this many iterations
    patience: int = 3

    # Hard safety bounds
    max_total_candidates: int = 512
