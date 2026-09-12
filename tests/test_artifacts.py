import json
from pathlib import Path
import tempfile
import unittest

from mfg_hedge.artifacts import write_summary_json


class WriteSummaryJsonTests(unittest.TestCase):
    def test_writes_payload_inside_a_fresh_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_summary_json(tmp, "run-a", {"value": 1})
            self.assertEqual(path, Path(tmp) / "run-a" / "summary.json")
            with path.open("r", encoding="utf-8") as stream:
                self.assertEqual(json.load(stream), {"value": 1})
            self.assertEqual([item.name for item in path.parent.iterdir()], ["summary.json"])

    def test_refuses_to_overwrite_an_existing_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            write_summary_json(tmp, "run-a", {"value": 1})
            with self.assertRaises(FileExistsError):
                write_summary_json(tmp, "run-a", {"value": 2})

    def test_rejects_run_ids_that_escape_the_artifacts_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "artifacts"
            for bad_id in (
                "../escaped",
                "..",
                "a/b",
                "a\\b",
                "",
                ".",
                " ",
                "run.",
                "CON",
                "con.txt",
                "NUL",
                "COM1",
                "LPT9",
            ):
                with self.assertRaises(ValueError, msg=f"run_id={bad_id!r}"):
                    write_summary_json(root, bad_id, {"value": 1})
            self.assertFalse((Path(tmp) / "escaped").exists())

    def test_accepts_portable_names_that_only_contain_reserved_substrings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for run_id in ("run.a", "console", "compute-COM1-run", "nul-safe"):
                path = write_summary_json(tmp, run_id, {"run_id": run_id})
                self.assertTrue(path.is_file())

    def test_failed_write_leaves_no_run_directory_behind(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(TypeError):
                write_summary_json(tmp, "run-a", {"not_json": object()})
            self.assertFalse((Path(tmp) / "run-a").exists())


if __name__ == "__main__":
    unittest.main()
