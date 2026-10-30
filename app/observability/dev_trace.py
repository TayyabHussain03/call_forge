"""Dev-only typed trace sink for live-conversation diagnostics.

Opt-in via `DEV_LIVE_TRACE=1`. When enabled, `make_stderr_dev_trace_sink`
returns a callable that `ProductionTurnProcessor` invokes once per turn
with a bounded structured record. Output goes to stderr as one JSON
line per turn — easy to grep, easy to pipe to a file, never crosses
the public HTTP/WS boundary.

Explicit non-goals:
    - no persistent file writer (operator pipes stderr if they want one)
    - no network sink
    - no PII/secret filtering (the sink is only given the already-bounded
      record built by `_build_dev_trace`, which excludes credentials,
      raw prompts, and chain-of-thought)
"""

from __future__ import annotations

import json
import os
import sys
from typing import Callable


def is_dev_trace_enabled() -> bool:
    """`DEV_LIVE_TRACE` env var controls opt-in. Any truthy string enables."""
    value = os.environ.get("DEV_LIVE_TRACE", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def make_stderr_dev_trace_sink() -> Callable[[dict], None]:
    """One JSON line per turn, prefixed so operators can grep it out."""
    def _sink(record: dict) -> None:
        line = "[dev-trace] " + json.dumps(record, sort_keys=True, default=str)
        print(line, file=sys.stderr, flush=True)
    return _sink


def resolve_dev_trace_sink() -> Callable[[dict], None] | None:
    """Returns a sink when `DEV_LIVE_TRACE=1`, else None.

    This is the single call site the composition root uses — the
    toggle lives in one place and is explicit at construction time.
    """
    if not is_dev_trace_enabled():
        return None
    return make_stderr_dev_trace_sink()
