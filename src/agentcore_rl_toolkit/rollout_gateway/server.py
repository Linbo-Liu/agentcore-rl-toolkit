"""Run a :class:`RolloutGateway` on a background thread, for synchronous trainers.

A sync trainer (slime, verl) blocks its thread waiting for episode results, but the
gateway must keep answering the agent's LLM calls the whole time. Served on the same
thread, the block would freeze the event loop and deadlock the episode — so this
class serves ``gateway.app`` on its own daemon thread with a long-lived loop. The
session API (``create_session`` / ``finish_session`` / ``drop_session``) stays safe
to call from the trainer thread. Async trainers don't need this: mount
``gateway.app`` on your own loop and ``await`` instead.

The serving mechanics are adapted from slime's ``slime/agent/aiohttp_threaded.py``
(baseline commit ``fa3c990a``; see NOTICE).

Requires ``aiohttp`` (the ``[gateway]`` extra), like :class:`RolloutGateway` itself.
"""

from __future__ import annotations

import asyncio
import logging
import threading

from aiohttp import web
from aiohttp.web_log import AccessLogger

logger = logging.getLogger(__name__)


class FilteredAccessLogger(AccessLogger):
    """Log only failures and slow requests; healthy fast traffic is noise."""

    SLOW_THRESHOLD_SEC = 120.0

    def log(self, request, response, time):
        if request.method == "HEAD":
            return
        if response.status == 200 and time <= self.SLOW_THRESHOLD_SEC:
            return
        super().log(request, response, time)


class ThreadedGatewayServer:
    """Serve an assembled :class:`RolloutGateway` on a background thread.

    ``start()`` blocks until the server is bound (or raises if binding fails),
    so callers can hand out :attr:`base_url` immediately afterwards. ``port=0``
    binds an OS-assigned port, reflected in :attr:`port`/:attr:`base_url` after
    ``start()``.

    Session identity rides in the api-key / Bearer slot of each request, so all
    sessions share the single fixed :attr:`base_url`.
    """

    def __init__(self, gateway, *, host: str, port: int = 0, startup_timeout: float = 120.0):
        self.gateway = gateway
        self.host = host
        self.port = port
        self.startup_timeout = startup_timeout
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._runner: web.AppRunner | None = None

    @property
    def base_url(self) -> str:
        """Fixed OpenAI-compatible ``base_url`` shared by all sessions."""
        return f"http://{self.host}:{self.port}/v1"

    def start(self) -> None:
        started = threading.Event()
        startup_error: list[BaseException] = []

        def _serve() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # handler_cancellation=True: a client disconnect cancels the
                # in-flight handler coroutine, so an agent that dies mid-request
                # doesn't leave an orphaned generate call in the sampling backend.
                runner = web.AppRunner(
                    self.gateway.app,
                    handler_cancellation=True,
                    access_log_class=FilteredAccessLogger,
                )
                loop.run_until_complete(runner.setup())
                site = web.TCPSite(runner, host=self.host, port=self.port)
                loop.run_until_complete(site.start())
                for sock in site._server.sockets:  # resolve OS-assigned port when port=0
                    self.port = sock.getsockname()[1]
                    break
                self._loop = loop
                self._runner = runner
                started.set()
                loop.run_forever()
            except BaseException as e:  # surface bind/setup failures to the caller
                startup_error.append(e)
                started.set()

        self._thread = threading.Thread(target=_serve, name="rollout-gateway", daemon=True)
        self._thread.start()
        if not started.wait(timeout=self.startup_timeout):
            raise TimeoutError(f"Rollout gateway did not start within {self.startup_timeout}s")
        if startup_error:
            raise RuntimeError(f"Rollout gateway failed to start on {self.host}:{self.port}") from startup_error[0]
        logger.info("Rollout gateway serving at %s", self.base_url)

    def shutdown(self) -> None:
        if self._loop is None:
            return
        # Clean up on the live loop first (graceful connection shutdown +
        # release of the listening socket), then stop the loop.
        try:
            fut = asyncio.run_coroutine_threadsafe(self._runner.cleanup(), self._loop)
            fut.result(timeout=10)
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=10)
        self._loop = None
