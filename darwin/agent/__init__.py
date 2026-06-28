"""Darwin Phase B2 — the Worker Agent (the atom).

One generic, model-agnostic agent that takes an ``AgentSpec`` (the role the
Architect authors in B4) and an ``AgentInput``, calls a model through the
registry-backed ``ModelClient``, and returns strict structured output as an
``AgentResult`` — never free text, never a fitness number.

Four frozen surfaces are the handoff contract to B3: ``AgentSpec``,
``AgentResult``, ``ModelClient``, and the six output schemas.
"""

from darwin.agent.client import ModelClient, ModelResponse, Usage
from darwin.agent.outputs import (
    ArbitrationOutput,
    ConstraintReportOutput,
    CritiqueOutput,
    DecompositionOutput,
    FullSolutionOutput,
    PartialSolutionOutput,
)
from darwin.agent.registry import ModelEntry, ModelRegistry, Provider, default_registry
from darwin.agent.spec import AgentSpec, InputKind, OutputKind, ThinkingLevel
from darwin.agent.telemetry import InMemoryTelemetrySink, MongoTelemetrySink, NullTelemetrySink
from darwin.agent.worker import AgentInput, AgentResult, WorkerAgent

__all__ = [
    "AgentSpec",
    "InputKind",
    "OutputKind",
    "ThinkingLevel",
    "AgentInput",
    "AgentResult",
    "WorkerAgent",
    "ModelClient",
    "ModelResponse",
    "Usage",
    "ModelRegistry",
    "ModelEntry",
    "Provider",
    "default_registry",
    "FullSolutionOutput",
    "PartialSolutionOutput",
    "CritiqueOutput",
    "ConstraintReportOutput",
    "ArbitrationOutput",
    "DecompositionOutput",
    "InMemoryTelemetrySink",
    "MongoTelemetrySink",
    "NullTelemetrySink",
]
