"""Darwin — recursive self-improving swarm for supply-chain optimization.

Phase B1 (``darwin.problem``) is the foundation: a canonical, validated
problem/solution data model (the *contract*) plus a purely deterministic
fitness scorer. Every other phase (B2–B8) is built strictly against the
B1 contract.
"""

__version__ = "0.1.0"
