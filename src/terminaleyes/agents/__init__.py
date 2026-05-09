"""Agent layer.

Tiered architecture (CLAUDE.md "Agent architecture"):

- Tier 1 (atomic): capture, cursor, target, move, verify
- Tier 2 (action): click, type, wake
- Tier 3 (workflow): focus, navigate, login, search
- Tier 4 (storage): vault

Each agent is a small, testable unit with a typed ``Outcome`` return.
The shared :class:`AgentContext` carries capture/keyboard/mouse/vision
clients and the optional :class:`Vault`. Higher-tier agents compose
lower-tier ones rather than reimplementing primitives.
"""
