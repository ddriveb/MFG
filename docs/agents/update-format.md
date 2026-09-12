# Required Update Format

Append one entry to the active ticket's `## Progress log` for every work session. Use the same headings in the final user-facing report.

```markdown
### Update: YYYY-MM-DD — <short title>

Status: completed | partial | blocked

#### Goal

What this update set out to accomplish.

#### Changed

Concrete changes made. Use `None` when the session was read-only.

#### Verification

Exact commands or checks executed and their results. Never list an unexecuted check.

#### Artifacts

Paths to source, configuration, reports, or generated outputs. Use `None` when there are none.

#### Decisions and risks

New decisions, retained limitations, known risks, and any ADR impact.

#### Next

The single recommended next action. Use `None` when the ticket is complete and no follow-up is required.
```

Do not use vague entries such as “updated code” or “tests passed”. Name the affected behavior and the exact test command.

