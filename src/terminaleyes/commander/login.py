"""Backwards-compat shim. The real implementation lives in
:mod:`terminaleyes.agents.login` now.

The ``LoginFlow`` class and ``resolve_password`` function are
re-exported here so existing imports keep working. New code should
prefer ``LoginAgent`` directly.
"""

from __future__ import annotations

import logging

from terminaleyes.agents.login import (
    LOGIN_QUESTION,
    LoginAgent,
    LoginOutcome,
    resolve_password,
)

logger = logging.getLogger(__name__)


__all__ = [
    "LOGIN_QUESTION",
    "LoginAgent",
    "LoginFlow",
    "LoginOutcome",
    "resolve_password",
]


class LoginFlow:
    """Old facade preserved for the existing ``terminaleyes login`` CLI.

    Wraps :class:`LoginAgent` so the CLI's argparse args still flow
    through ``flow.login(password=…, wake=…, …)``. New code should use
    :class:`LoginAgent` directly.
    """

    def __init__(self, *, mouse, keyboard, session=None) -> None:
        self._mouse = mouse
        self._keyboard = keyboard
        self._session = session

    async def login(
        self,
        password: str,
        *,
        wake: bool = True,
        click_input: bool = False,
        submit: bool = True,
        verify: bool = True,
        verify_attempts: int = 6,
        verify_interval: float = 1.0,
    ) -> bool:
        """Run the login flow with an explicit password (already resolved).

        Builds an :class:`AgentContext` from the legacy session/mouse/
        keyboard fields and dispatches to :class:`LoginAgent`.
        """
        from terminaleyes.agents.context import AgentContext

        ctx = AgentContext(
            mouse=self._mouse,
            keyboard=self._keyboard,
            capture=getattr(self._session, "_capture", None),
            vision_client=getattr(self._session, "_client", None),
            vision_model=getattr(self._session, "_model", "") or "",
            evaluator=getattr(self._session, "_evaluator", None),
        )
        agent = LoginAgent(ctx)
        outcome = await agent.run(
            password=password,
            wake=wake,
            verify=verify,
            verify_attempts=verify_attempts,
            verify_interval=verify_interval,
            click_input=click_input,
            submit=submit,
        )
        if not outcome:
            print(f"Login NOT sent — {outcome.reason}")
            return False
        return True
