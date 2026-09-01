"""State machine configuration loading and fail-fast validation.

Ye module `conversation_config.yaml` ko typed structures mein load karta hai aur
STARTUP par validate karta hai — runtime par nahi. Agar config mein unknown
state/action, duplicate, invalid transition, ya terminal-with-outgoing ho, to
app boot hi nahi hoti (fail-fast).

DESIGN: ye sirf structure aur references validate karta hai. Context-sensitive
business validation (email candidate exists? DNC override?) yahan NAHI — woh
Sitting 2 ka `action_validator.py` karega. Ye separation guide ka requirement
hai.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from app.core.constants import AgentAction, ConversationState
from app.core.exceptions import ConfigurationError

# Special sentinel target: HANDLE_QUESTION jaise detour states previous state
# par wapas jaate hain. Config mein ye string se represent hota hai.
PREVIOUS_STATE_SENTINEL = "__previous__"


@dataclass(frozen=True)
class TransitionRule:
    """One action → target-state rule within a state.

    Attributes:
        action: Jo action ye transition trigger karta hai.
        target: Target state, ya PREVIOUS_STATE_SENTINEL (detour restore).
        outcome: Optional explicit outcome label jab target terminal ho. Ye
            field extensible hai — future mein call-outcome vs lead-status alag
            karne ke liye. Abhi ek string label.
        requires: Context preconditions (Sitting 2 validator inhe evaluate
            karega). Abhi sirf declared, enforced nahi.
        retry_bucket: Agar ye ek bounded-retry transition hai (clarification),
            to kaunse limit bucket se count hoti hai. None agar retry nahi.
        preserve_interest: True jab ye transition lead ka interest barqarar
            rakhe (busy→callback).
    """

    action: AgentAction
    target: str  # ConversationState value ya PREVIOUS_STATE_SENTINEL
    outcome: str | None = None
    requires: tuple[str, ...] = ()
    retry_bucket: str | None = None
    preserve_interest: bool = False


@dataclass(frozen=True)
class StateDefinition:
    """A single conversational state and its rules.

    Attributes:
        name: The ConversationState this defines.
        allowed_actions: Structural whitelist of AgentActions in this state.
        transitions: Mapping of action → TransitionRule.
        requires: Context preconditions to ENTER this state (validator-evaluated).
        is_detour: True for temporary states (HANDLE_QUESTION) that restore the
            previous meaningful state.
        is_terminal: True if this state has no outgoing transitions.
    """

    name: ConversationState
    allowed_actions: frozenset[AgentAction]
    transitions: dict[AgentAction, TransitionRule]
    requires: tuple[str, ...] = ()
    is_detour: bool = False
    is_terminal: bool = False


@dataclass(frozen=True)
class FallbackMapping:
    """A single deterministic fallback: which safe action + response identifier.

    Attributes:
        action: Safe recovery action to take.
        response_key: Stable machine-readable identifier for the response
            (natural-language wording is filled later by the LLM/prompt layer).
    """

    action: AgentAction
    response_key: str


@dataclass(frozen=True)
class FallbackConfig:
    """All configured fallback mappings.

    Attributes:
        by_state: State-specific fallbacks (state → mapping).
        by_category: Category-specific fallbacks (category string → mapping),
            e.g. dnc_conflict, unknown_requirement. These override by_state.
        default: The safe-close fallback used when nothing more specific applies.
    """

    by_state: dict[ConversationState, FallbackMapping]
    by_category: dict[str, FallbackMapping]
    default: FallbackMapping


@dataclass(frozen=True)
class MachineConfig:
    """The fully-parsed, validated state machine configuration.

    Attributes:
        initial_state: Where a fresh session starts.
        states: All state definitions keyed by ConversationState.
        terminal_states: Set of terminal states.
        limits: Named integer limits (retry/clarification bounds).
        response_policy: Per-state response policy metadata (LLM layer later).
        fallbacks: Deterministic fallback mappings (Sitting 2C-A).
    """

    initial_state: ConversationState
    states: dict[ConversationState, StateDefinition]
    terminal_states: frozenset[ConversationState]
    limits: dict[str, int] = field(default_factory=dict)
    response_policy: dict[str, Any] = field(default_factory=dict)
    fallbacks: FallbackConfig | None = None


def _coerce_state(value: str) -> ConversationState:
    """Convert a config string to a ConversationState, or fail fast.

    Args:
        value: The raw state string from YAML.

    Returns:
        ConversationState: The matching enum member.

    Raises:
        ConfigurationError: Agar value kisi known state se match na kare.
    """
    try:
        return ConversationState(value)
    except ValueError as exc:
        raise ConfigurationError(f"Unknown state in config: {value!r}") from exc


def _coerce_action(value: str) -> AgentAction:
    """Convert a config string to an AgentAction, or fail fast.

    Args:
        value: The raw action string from YAML.

    Returns:
        AgentAction: The matching enum member.

    Raises:
        ConfigurationError: Agar value kisi known action se match na kare.
    """
    try:
        return AgentAction(value)
    except ValueError as exc:
        raise ConfigurationError(f"Unknown action in config: {value!r}") from exc


def load_config(path: str | Path) -> MachineConfig:
    """Load and validate the conversation config, failing fast on any error.

    Ye function saari structural validation karta hai: unknown states/actions,
    duplicate keys, invalid transition targets, terminal-with-outgoing, aur
    actions jo allowed-list mein hain lekin transition define nahi karte (aur
    vice-versa). Koi bhi masla → ConfigurationError, boot ruk jaata hai.

    Args:
        path: Path to the YAML config file.

    Returns:
        MachineConfig: The parsed, validated configuration.

    Raises:
        ConfigurationError: On any structural problem in the config.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigurationError(f"Config file not found: {config_path}")

    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    # ── terminal states ──
    terminal_raw = raw.get("terminal_states", [])
    terminal_states = frozenset(_coerce_state(s) for s in terminal_raw)

    # ── states ──
    states_raw = raw.get("states", {})
    if not states_raw:
        raise ConfigurationError("Config has no states defined.")

    # duplicate detection: YAML keys already unique, but guard anyway on the
    # coerced enum set vs raw count.
    if len(states_raw) != len({_coerce_state(k) for k in states_raw}):
        raise ConfigurationError("Duplicate state definitions detected.")

    states: dict[ConversationState, StateDefinition] = {}

    for state_key, body in states_raw.items():
        state = _coerce_state(state_key)
        body = body or {}

        allowed = frozenset(_coerce_action(a) for a in body.get("actions", []))
        requires = tuple(body.get("requires", []))
        is_detour = bool(body.get("is_detour", False))
        is_terminal = state in terminal_states

        transitions_raw = body.get("transitions", {}) or {}
        transitions: dict[AgentAction, TransitionRule] = {}

        for action_key, rule_body in transitions_raw.items():
            action = _coerce_action(action_key)
            rule_body = rule_body or {}
            target = rule_body.get("to")
            if target is None:
                raise ConfigurationError(
                    f"State {state.value!r} action {action.value!r} has no 'to' target."
                )
            # target ya to valid state ho ya previous-sentinel.
            if target != PREVIOUS_STATE_SENTINEL:
                _coerce_state(target)  # fail-fast if unknown target

            transitions[action] = TransitionRule(
                action=action,
                target=target,
                outcome=rule_body.get("outcome"),
                requires=tuple(rule_body.get("requires", [])),
                retry_bucket=rule_body.get("retry"),
                preserve_interest=bool(rule_body.get("preserve_interest", False)),
            )

        # ── invariant checks ──
        # terminal state must have zero outgoing transitions.
        if is_terminal and transitions:
            raise ConfigurationError(
                f"Terminal state {state.value!r} must have no outgoing transitions."
            )
        # every transition action must be in the allowed-actions whitelist.
        for action in transitions:
            if action not in allowed:
                raise ConfigurationError(
                    f"State {state.value!r} transitions on {action.value!r} "
                    f"but it is not in allowed actions."
                )

        states[state] = StateDefinition(
            name=state,
            allowed_actions=allowed,
            transitions=transitions,
            requires=requires,
            is_detour=is_detour,
            is_terminal=is_terminal,
        )

    # ── initial state ──
    initial_raw = raw.get("initial_state")
    if initial_raw is None:
        raise ConfigurationError("Config missing 'initial_state'.")
    initial_state = _coerce_state(initial_raw)
    if initial_state not in states:
        raise ConfigurationError(
            f"initial_state {initial_state.value!r} is not a defined state."
        )
    if initial_state in terminal_states:
        raise ConfigurationError("initial_state cannot be a terminal state.")

    # ── cross-reference: every non-sentinel transition target must be defined ──
    for state_def in states.values():
        for rule in state_def.transitions.values():
            if rule.target == PREVIOUS_STATE_SENTINEL:
                continue
            target_state = ConversationState(rule.target)
            if target_state not in states:
                raise ConfigurationError(
                    f"State {state_def.name.value!r} transitions to undefined "
                    f"state {rule.target!r}."
                )
            # outcome only meaningful when target is terminal.
            if rule.outcome is not None and target_state not in terminal_states:
                raise ConfigurationError(
                    f"State {state_def.name.value!r} action {rule.action.value!r} "
                    f"declares outcome but target {rule.target!r} is not terminal."
                )
        # retry buckets must reference a defined limit.
        for rule in state_def.transitions.values():
            if rule.retry_bucket is not None:
                limit_key = f"{rule.retry_bucket}_max_attempts"
                if limit_key not in raw.get("limits", {}):
                    raise ConfigurationError(
                        f"Retry bucket {rule.retry_bucket!r} has no matching "
                        f"limit {limit_key!r} in config."
                    )

    # ── fallbacks (Sitting 2C-A) — optional section ──
    raw_fallbacks = raw.get("fallbacks")
    fallbacks = (
        _parse_and_validate_fallbacks(raw_fallbacks, states) if raw_fallbacks else None
    )

    return MachineConfig(
        initial_state=initial_state,
        states=states,
        terminal_states=terminal_states,
        limits=dict(raw.get("limits", {})),
        response_policy=dict(raw.get("response_policy", {})),
        fallbacks=fallbacks,
    )


def _parse_mapping(body: dict[str, Any], where: str) -> FallbackMapping:
    """Parse one fallback mapping entry, failing fast on problems.

    Args:
        body: The raw {action, response_key} dict.
        where: Context string for error messages.

    Returns:
        FallbackMapping: The parsed mapping.

    Raises:
        ConfigurationError: On missing/invalid fields.
    """
    action_raw = body.get("action")
    key = body.get("response_key")
    if action_raw is None or key is None:
        raise ConfigurationError(
            f"Fallback {where!r} needs both 'action' and 'response_key'."
        )
    if not isinstance(key, str) or not key.strip():
        raise ConfigurationError(f"Fallback {where!r} response_key must be non-empty.")
    return FallbackMapping(action=_coerce_action(action_raw), response_key=key)


def _parse_and_validate_fallbacks(
    raw: dict[str, Any], states: dict[ConversationState, StateDefinition]
) -> FallbackConfig:
    """Parse fallback config and verify each action is valid for its state.

    Startup validation (guide requirement): har state-specific fallback action us
    state ke allowed actions mein hona chahiye — warna fail-fast. Isse runtime par
    "fallback khud invalid" wali surprise nahi hoti.

    Args:
        raw: The raw 'fallbacks' config section.
        states: Parsed state definitions, to verify action validity.

    Returns:
        FallbackConfig: Validated fallback configuration.

    Raises:
        ConfigurationError: If a state fallback action isn't allowed in its state,
            or default is missing.
    """
    by_state: dict[ConversationState, FallbackMapping] = {}
    for state_key, body in (raw.get("by_state", {}) or {}).items():
        state = _coerce_state(state_key)
        mapping = _parse_mapping(body or {}, f"by_state.{state_key}")
        # verify the fallback action is structurally allowed in that state.
        state_def = states.get(state)
        if state_def is None:
            raise ConfigurationError(f"Fallback references unknown state {state_key!r}.")
        if mapping.action not in state_def.allowed_actions:
            raise ConfigurationError(
                f"Fallback action {mapping.action.value!r} is not allowed in "
                f"state {state_key!r}."
            )
        by_state[state] = mapping

    by_category: dict[str, FallbackMapping] = {}
    for cat_key, body in (raw.get("by_category", {}) or {}).items():
        by_category[cat_key] = _parse_mapping(body or {}, f"by_category.{cat_key}")

    default_raw = raw.get("default")
    if not default_raw:
        raise ConfigurationError("Fallback config must define a 'default'.")
    default = _parse_mapping(default_raw, "default")

    return FallbackConfig(by_state=by_state, by_category=by_category, default=default)