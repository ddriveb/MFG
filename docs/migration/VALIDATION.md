# Migration validation — 2026-09-12

Independent checkout, Windows Python 3.10.11, fresh `.venv`.

- `scripts/bootstrap.ps1 -PythonExe <Python310/python.exe>`: PASS.
- `.venv/Scripts/python.exe -m unittest discover -s tests -v`: **836 tests,
  OK (skipped=1)**, 294.695 s. The skip is the optional old schema-2 artifact
  comparison, not a missing mandatory test input.
- `.venv/Scripts/python.exe -m mfg_hedge check --config configs/v1_minimal.json`:
  `status: ok`. The existing failure-load/headroom observation is unchanged.
- First clean run exposed 16 missing-input errors. After restoring seven
  byte-identical controlled inputs, all 36 affected-module tests passed, then the
  full suite passed. No test was removed or changed to hide the failure.
- All imported Python ASTs match their originals; the simultaneous-routing
  module received explanatory comments only. Original project files are unchanged.

No formal qualification/holdout campaign was started. Unit tests may exercise
bounded synthetic holdout code paths as regression tests; these are not research
holdout evaluations and do not consume or replace the original run ledger.
The new computer itself has not been accessed; repeat bootstrap there.
