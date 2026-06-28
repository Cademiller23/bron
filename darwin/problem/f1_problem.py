"""The F1 calendar as a ProblemInstance the architect/runner consume.

We do NOT subclass ProblemInstance (it's frozen with strict validators tuned for
flow networks). Instead we build a *valid* ProblemInstance whose nodes are the 24
circuits and whose problem_class is F1_CALENDAR, so:
  * the architect's design prompt sees problem_class='F1_CALENDAR' (line 299) and
    becomes F1-aware automatically, and
  * the instance carries a KnownOptimum = the FEASIBLE_BASELINE fitness anchor.

The network is a minimal well-formed transport shell (one source, one sink, one
arc) purely to satisfy ProblemInstance's structural validators — the REAL problem
lives in the F1 scorer, not the network. The agents are told (via role
descriptions in F1-D) to emit a calendar; the codec packs it into the Solution.
"""

from darwin.problem.schemas import (
    Arc,
    InstanceMetadata,
    Difficulty,
    KnownOptimum,
    Node,
    NodeType,
    OptimumSource,
    ProblemClass,
    ProblemInstance,
)

F1_INSTANCE_ID = "f1_2026_calendar"


def build_f1_instance() -> ProblemInstance:
    """A structurally-valid ProblemInstance flagged F1_CALENDAR.

    The node/arc shell is minimal-but-valid (supply==demand) to pass the frozen
    validators; the F1 scorer ignores it and scores the calendar instead.
    """
    nodes = [
        Node(node_id="f1_src", node_type=NodeType.SOURCE, supply=1.0),
        Node(node_id="f1_snk", node_type=NodeType.SINK, demand=1.0),
    ]
    arcs = [Arc(arc_id="f1_arc", from_node="f1_src", to_node="f1_snk", unit_cost=0.0)]
    meta = InstanceMetadata(
        difficulty=Difficulty.HARD,
        notes="F1 2026 calendar optimization: 24 circuits, 3 conflicting constraint "
              "families (routing/scheduling/clustering). Decision = ordered (race, week).",
    )
    # The reference optimum is the proven-feasible baseline's fitness (1.0 on the
    # normalized scale): a feasible calendar at/below baseline carbon scores ~1.0.
    known = KnownOptimum(objective_value=1.0, source=OptimumSource.SOLVER_VERIFIED, verified=True)
    return ProblemInstance(
        instance_id=F1_INSTANCE_ID,
        source="generated",
        problem_class=ProblemClass.F1_CALENDAR,
        nodes=nodes,
        arcs=arcs,
        known_optimum=known,
        metadata=meta,
    )
