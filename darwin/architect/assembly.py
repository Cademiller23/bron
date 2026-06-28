"""Deterministic assembly: an ``ArchitectTeamDesign`` -> a B3 ``TeamGenome``.

Pure and deterministic (no model calls). The Architect's creativity is fully
contained in the model call; this code builds the structure so it can be tested
exactly: assign agent_ids, embed full B2 ``AgentSpec``s, resolve role-name wiring
to ``Edge``s, set the arbiter, and compute org-chart layout positions.
"""

import re
from typing import Any, Dict, List, Optional

from darwin.agent.spec import AgentSpec, OutputKind
from darwin.architect.schemas import ArchitectTeamDesign
from darwin.team.genome import AgentNode, Edge, MutationActor, TeamGenome

_SLUG_CLEAN = re.compile(r"[^a-z0-9]+")


class AssemblyError(ValueError):
    """Raised when a design cannot be assembled (unresolvable arbiter / edges)."""


def _slugify(name: str) -> str:
    slug = _SLUG_CLEAN.sub("_", name.strip().lower()).strip("_-")
    if not slug or not slug[0].isalnum():
        slug = "agent_" + slug if slug else "agent"
    return slug


def _layout_for(output_contract: OutputKind, is_arbiter: bool) -> int:
    """A y-band for the org chart: proposers up top, checkers in the middle, the
    arbiter at the sink."""
    if is_arbiter:
        return 2
    if output_contract in (OutputKind.CRITIQUE, OutputKind.CONSTRAINT_REPORT):
        return 1
    return 0


def assemble(
    design: ArchitectTeamDesign, instance_id: str, registry: Any = None, *, generation: int = 0
) -> TeamGenome:
    """Convert a validated design into a fresh (version-1, DRAFT) ``TeamGenome``."""
    slug_counts: Dict[str, int] = {}
    role_to_id: Dict[str, str] = {}
    nodes: List[AgentNode] = []
    band_index: Dict[int, int] = {}

    for draft in design.agents:
        slug = _slugify(draft.role_name)
        n = slug_counts.get(slug, 0)
        slug_counts[slug] = n + 1
        agent_id = slug if n == 0 else f"{slug}-{n + 1}"

        # build a full B2 AgentSpec (validates model_id against the registry
        # context). role_name == the de-duplicated agent_id, so role_names are
        # unique even when distinct authored roles slugify to the same slug.
        spec = AgentSpec.model_validate(
            {
                "agent_id": agent_id,
                "role_name": agent_id,
                "role_description": draft.role_description,
                "input_contract": draft.input_contract.value,
                "output_contract": draft.output_contract.value,
                "model_id": draft.model_id,
                "thinking_level": draft.thinking_level.value,
                "created_by": "architect",
                "spec_version": "1.0.0",
            },
            context={"registry": registry},
        )

        is_arbiter = draft.role_name == design.arbiter_role_name
        band = _layout_for(draft.output_contract, is_arbiter)
        col = band_index.get(band, 0)
        band_index[band] = col + 1
        layout = {"x": float(col), "y": float(band)}

        # exact original role_name -> agent_id (first occurrence wins for dup originals)
        role_to_id.setdefault(draft.role_name, agent_id)
        nodes.append(AgentNode(agent_id=agent_id, spec=spec, layout=layout))

    # resolve wiring
    edges: List[Edge] = []
    for ed in design.edges:
        src = role_to_id.get(ed.from_role_name)
        dst = role_to_id.get(ed.to_role_name)
        if src is None or dst is None:
            raise AssemblyError(
                f"edge references unknown role: {ed.from_role_name!r} -> {ed.to_role_name!r}"
            )
        edges.append(Edge(from_agent_id=src, to_agent_id=dst, edge_type=ed.edge_type))

    arbiter_id = role_to_id.get(design.arbiter_role_name)
    if arbiter_id is None:
        raise AssemblyError(f"arbiter_role_name {design.arbiter_role_name!r} matches no authored agent")

    return TeamGenome.create(
        instance_id=instance_id,
        agents=nodes,
        edges=edges,
        arbiter_id=arbiter_id,
        actor=MutationActor.ARCHITECT,
        description=design.design_rationale or "Initial team curated by the Architect",
        generation=generation,
    )
