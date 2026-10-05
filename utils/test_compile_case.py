"""Compile-wrapper regressions without invoking a compiler or touching real cases.

Run from the repository root: python utils/test_compile_case.py
"""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class CompileCaseTests(unittest.TestCase):
    def setUp(self):
        source = Path(__file__).resolve().with_name("compile_case.sh")
        # /tmp can be mounted noexec; use the same filesystem as the real build.
        self.temp = tempfile.TemporaryDirectory(prefix="compile test ", dir=source.parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "utils").mkdir()
        self.wrapper = self.root / "utils/compile_case.sh"
        shutil.copy2(source, self.wrapper)
        self.case = self.root / "envs/cases/mini_channel"
        self.case.mkdir(parents=True)
        # Deliberately omit executable permission on this case script.
        (self.case / "compile_script").write_text('''#!/bin/bash
solver=drl
if [[ "$1" == --solver ]]; then solver="$2"; shift 2; fi
binary=nek5000
[[ "$solver" == raw ]] && binary=nek5000_solo
if [[ "$1" == --clean ]]; then
    [[ "$COMPILE_TEST_MODE" == clean_fail ]] && exit 17
    [[ "$COMPILE_TEST_MODE" == stale_clean ]] && exit 0
    rm -f "$binary"
else
    # Reproduce v17's removal of the coupled executable during a raw build.
    [[ "$solver" == raw ]] && rm -f nek5000
    [[ "$COMPILE_TEST_MODE" == build_fail ]] && exit 19
    [[ "$COMPILE_TEST_MODE" == masked_fail ]] && exit 0
    printf '#!/bin/bash\\nexit 0\\n' > "$binary"
    chmod +x "$binary"
fi
''')

    def run_wrapper(self, *args, mode="", success=True):
        env = os.environ.copy()
        env["COMPILE_TEST_MODE"] = mode
        result = subprocess.run(
            ["bash", str(self.wrapper), *args], cwd=self.root, env=env,
            capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(result.returncode == 0, success, result.stdout + result.stderr)
        if not success:
            self.assertNotIn("Finished compile", result.stdout)
        return result

    def test_default_case(self):
        self.run_wrapper()
        self.assertTrue(os.access(self.case / "nek5000", os.X_OK))

    def test_path_and_legacy_alias(self):
        for flag in ("--path", "--m"):
            with self.subTest(flag=flag):
                self.run_wrapper(flag, str(self.case))

    def test_invalid_arguments_fail_before_cleaning(self):
        binary = self.case / "nek5000"
        binary.write_text("existing executable")
        for args in (("--path",), ("--solver",), ("--path", "--solver"),
                     ("--solver", "unknown"), ("--version", "v17"),
                     ("--path", "absent"), ("--path", str(self.root))):
            with self.subTest(args=args):
                self.run_wrapper(*args, success=False)
                self.assertEqual(binary.read_text(), "existing executable")

    def test_help_does_not_build(self):
        self.run_wrapper("--help")
        self.assertFalse((self.case / "nek5000").exists())

    def test_preserves_both_backends(self):
        self.run_wrapper("--solver", "drl")
        binary = self.case / "nek5000"
        before = binary.read_bytes(), binary.stat().st_mode
        self.run_wrapper("--solver", "raw")
        self.assertEqual((binary.read_bytes(), binary.stat().st_mode), before)
        self.assertTrue(os.access(self.case / "nek5000_solo", os.X_OK))
        self.assertEqual(list(self.case.glob(".compile_case.*")), [])

    def test_clean_failure_and_stale_binary(self):
        for mode in ("clean_fail", "stale_clean"):
            with self.subTest(mode=mode):
                self.run_wrapper()
                self.run_wrapper(mode=mode, success=False)

    def test_build_failures_restore_other_binary(self):
        for mode in ("build_fail", "masked_fail"):
            with self.subTest(mode=mode):
                self.run_wrapper("--solver", "drl")
                before = (self.case / "nek5000").read_bytes()
                self.run_wrapper("--solver", "raw", mode=mode, success=False)
                self.assertEqual((self.case / "nek5000").read_bytes(), before)
                self.assertFalse((self.case / "nek5000_solo").exists())
                self.assertEqual(list(self.case.glob(".compile_case.*")), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
