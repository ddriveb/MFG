# Issue tracker: Local Markdown

This repository stores issues and specs as Markdown files under `.scratch/`.

## Conventions

- One directory per feature: `.scratch/<feature-slug>/`.
- The feature specification is `.scratch/<feature-slug>/spec.md`.
- Each implementation ticket is a separate file at `.scratch/<feature-slug>/issues/<NN>-<slug>.md`, numbered from `01`.
- Never combine multiple tickets into one ticket file.
- Each ticket contains `Type:`, `Status:`, and `Blocked by:` near the top.
- Allowed statuses are `open`, `claimed`, `blocked`, and `resolved`.
- Claim a ticket before changing code. Resolve it only after its acceptance criteria and verification commands pass.
- Append work-session history under `## Progress log` using `docs/agents/update-format.md`.
- Append discussion that does not belong in the progress record under `## Comments`.

## Publishing and fetching

When a skill says “publish to the issue tracker”, create the appropriate spec or ticket under `.scratch/<feature-slug>/`.

When a skill says “fetch the relevant ticket”, read the referenced ticket file. If no path is supplied, inspect the relevant feature directory and select the first open, unblocked ticket.

## Wayfinding operations

- Map: `.scratch/<effort>/map.md`.
- Child ticket: `.scratch/<effort>/issues/<NN>-<slug>.md`.
- Blocking: `Blocked by: NN, NN`; use `none` when there are no blockers.
- Frontier: the lowest-numbered ticket whose status is `open` and whose blockers are all resolved.
- Claim: set `Status: claimed` before work begins.
- Resolve: append the evidence under `## Answer`, set `Status: resolved`, then update the map if one exists.

