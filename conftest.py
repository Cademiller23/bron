"""Root pytest configuration.

Registers a Hypothesis profile that disables the per-example deadline so the
property-based invariant tests (which do real solver work) never flake on a slow
machine, while keeping example diversity.
"""

from hypothesis import settings

settings.register_profile("darwin", deadline=None)
settings.load_profile("darwin")
