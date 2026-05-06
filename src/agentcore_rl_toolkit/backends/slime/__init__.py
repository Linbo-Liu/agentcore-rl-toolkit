"""Slime training backend for agentcore-rl-toolkit.

Primary entry point is :class:`SlimeRunner`. The ``integration/`` subpackage
and ``patches/`` module are implementation detail — users shouldn't import
them directly, but they are load-bearing (slime loads
``agentcore_rl_toolkit.backends.slime.integration.rollout.generate_rollout``
via ``--rollout-function-path`` at job-submit time).
"""

from .runner import SlimeRunner

__all__ = ["SlimeRunner"]
