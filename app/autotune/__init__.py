"""Autonomous closed-loop tuning (multi-agent) for molecular generation.

This package provides a pluggable Scientist / Executor / Critic architecture
with a feedback loop. Concrete generators/evaluators (diffusion, DiffGui,
docking) are injected via adapters.
"""

from app.autotune.loop import AutotuneLoop

try:
	from app.autotune.langgraph_loop import LangGraphAutotuneLoop

	__all__ = ["AutotuneLoop", "LangGraphAutotuneLoop"]
except Exception:
	__all__ = ["AutotuneLoop"]
