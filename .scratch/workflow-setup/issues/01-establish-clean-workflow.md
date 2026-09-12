# Establish the clean local workflow

Type: task
Status: resolved
Blocked by: none

## Goal

Create the repository rules and supporting documents approved by the user.

## Scope

`AGENTS.md`, `CONTEXT.md`, `docs/agents/`, `docs/adr/`, and this Local Markdown tracker entry.

## Acceptance criteria

- The files listed in the feature spec exist and point to one another correctly.
- Existing Python tests still pass.
- The default configuration self-check still passes.

## Progress log

### Update: 2026-09-03 — Establish clean local workflow

Status: completed

#### Goal

Persist the approved Matt Pocock skills repository setup and the mandatory update format.

#### Changed

Added project-wide Agent rules, Local Markdown tracker rules, single-context domain rules, update format, domain context, ADR-0001, and README workflow pointers.

#### Verification

- `.\.venv\Scripts\python.exe -m unittest discover -s .\tests -v`: passed, 6 tests run.
- `.\.venv\Scripts\python.exe -m mfg_hedge check --config .\configs\v1_minimal.json`: passed with `status: ok`; the expected post-failure base-load audit remains visible.
- Required-file existence scan: all 11 workflow, ADR, spec, ticket, and prompt files are present.

#### Artifacts

`AGENTS.md`, `CONTEXT.md`, `docs/agents/`, `docs/adr/`, `.scratch/workflow-setup/`.

#### Decisions and risks

The project uses Local Markdown rather than a remote issue tracker. Triage labels are omitted because the triage skill is not installed.

#### Next

Implement the first Healthy + No Hedge vertical slice from `.scratch/healthy-no-hedge/`.

## Answer

The workflow structure has been created. Verification evidence is recorded in the progress entry above.

## Comments
