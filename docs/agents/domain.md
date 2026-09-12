# Domain Docs

This is a single-context repository.

## Before exploring or changing behavior

1. Read the repository-root `CONTEXT.md`.
2. Read ADRs under `docs/adr/` that affect the active ticket.
3. Read the active spec and ticket under `.scratch/`.

If a referenced domain document does not exist, continue without creating speculative terminology. Create or update domain documentation only when the work resolves a real term, invariant, or decision.

## File structure

```text
/
├── CONTEXT.md
├── docs/
│   ├── agents/
│   └── adr/
├── .scratch/
├── src/
└── tests/
```

## Vocabulary

Use the terms in `CONTEXT.md` in code, tests, tickets, reports, and experiment labels. If a needed concept is missing, record the gap in the active ticket before adding a durable definition.

## ADR conflicts

If proposed work conflicts with an ADR, name the conflict explicitly. Do not silently overwrite or bypass the decision. Either stay within the ADR or create a superseding ADR with the user's approval.

