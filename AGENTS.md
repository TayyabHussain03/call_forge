# Engineering Guardrails

## Core architecture

Preserve this authority chain:

```text
Untrusted model proposal
→ deterministic Scope policy
→ deterministic Authority policy
→ specialized deterministic resolution
→ ActionValidator
→ authoritative StateMachine
→ trusted effects
```

Brain reasons. Policies constrain. Deterministic system decides. Model output is
always untrusted and must never directly control next state, transitions, DNC,
services, contacts, persistence, pricing, discounts, claims, authority, human
approval, or commercial bounds.

## Permanent boundaries

- DNC has highest priority and is distinct from `NOT_INTERESTED`. Only a trusted
  `TrustedPriorityOutcome` may trigger priority execution; never raw text, model
  intent, or enum-like input.
- Keep Scope, Authority, catalog eligibility, claims, state, and persistence
  separate. Scope does not imply authority; `AUTHORITY_APPROVED` permits later
  deterministic validation, not execution.
- Preserve authority-derived bounds. Model output may never widen them.
- Only `AUTHORITY_APPROVED` enters Slice 3. Execution order is specialized
  resolution → `ActionValidator` → `StateMachine`. Never bypass validation, force
  transitions, or regenerate with an LLM after rejection; use deterministic
  recovery.
- Catalog eligibility and claim authorization remain authoritative. Never
  duplicate eligibility, hardcode campaign rules, randomly select, trust a model
  service authorization, or re-offer a recorded service. Track offers only after
  the existing successful transition contract.
- Contact handling remains deterministic. Preserve normalization, provenance,
  explicit confirmation, immutable source contact data, and no persistence of
  unconfirmed details. Confidence or wording never establishes confirmation.
- Business Intelligence is context, not authority. Proposed observations,
  inferences, goals, and memory updates never auto-commit. Keep observation,
  inference, authority, and execution distinct. Future memory must be person-aware
  and must not leak between contacts at one business.
- Confidence is observational only; never use generic confidence to authorize
  scope, authority, transitions, persistence, services, pricing, or confirmation.
- Keep reasoning-call, response-length, turn, and duration budgets separate.
  Budgets control resources, not business success. Pipeline stops are distinct
  from state-machine terminality.
- Mocks are deterministic, scripted, dumb, and transparent. Tests prove system
  behavior, not hidden mock intelligence.

## Development discipline

For each approved sitting: inspect → implement the smallest bounded change → run
focused tests → run full regression → inspect status/diff/stat → commit → push.
Prefer existing contracts/services over parallel implementations. Avoid giant
rewrites, unrelated refactors, speculative abstractions, duplicate truth, and
future-slice work.

Never commit `.env`, secrets, API keys, tokens, credentials, sensitive logs,
caches, or temporary/generated junk. Never print secrets.

After passing tests, stage only approved files, make one concise commit, push the
current branch to its existing remote, and verify final status. Do not force-push,
change remotes, rewrite history, reset, rebase, amend, switch/create branches
without instruction, discard unrelated changes, or guess at conflict resolution.
If tests or safety checks fail, stop without pushing. If authentication fails,
keep the successful local commit, report the branch is ahead, and stop.
