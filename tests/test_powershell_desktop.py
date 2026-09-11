"""Windows PowerShell 5.1 wrapper tests, entirely inside temporary fixtures.

The fake package records argv instead of opening images, cameras or models.
A fake pip module fails locally: setup tests never install or download anything.
The real project environment and user execution policy/profile are untouched.
"""

import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
FAKE_MODULE = '''import json, os, sys
from pathlib import Path
Path(os.environ["HELMET_WRAPPER_CAPTURE"]).write_text(json.dumps({
    "argv": sys.argv[1:], "pythonpath": os.environ.get("PYTHONPATH"),
    "pythonioencoding": os.environ.get("PYTHONIOENCODING"), "stdout_encoding": sys.stdout.encoding,
    "role": __name__, "executable": sys.executable,
}), encoding="utf-8")
code = int(os.environ.get("HELMET_WRAPPER_EXIT", "0"))
if code:
    print("Intentional local shim failure", file=sys.stderr)
raise SystemExit(code)
'''


def _ps_literal(value):
    return "'" + str(value).replace("'", "''") + "'"


@unittest.skipUnless(sys.platform == "win32", "Windows PowerShell 5.1 tests run only on Windows.")
class PowerShellDesktopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.powershell = shutil.which("powershell.exe")
        if not cls.powershell:
            raise RuntimeError("Windows PowerShell 5.1 is required for the Windows wrapper tests.")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.project = self.root / "proje şğü 研究 [desktop]"
        self.scripts = self.project / "scripts"
        self.scripts.mkdir(parents=True)
        for script in (PROJECT / "scripts").glob("*.ps1"):
            shutil.copy2(script, self.scripts / script.name)
        (self.project / "models").mkdir()
        (self.project / "models" / "detector.onnx").write_bytes(b"synthetic model placeholder; never loaded")
        package = self.project / "src" / "helmetai_fcw"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("", encoding="utf-8")
        for name in ("__main__.py", "image_check.py", "log_report.py"):
            (package / name).write_text(FAKE_MODULE, encoding="utf-8")
        (package.parent / "pip.py").write_text(FAKE_MODULE, encoding="utf-8")

        # Copy an existing stdlib venv launcher, never the real project venv.
        # No pip bootstrap, dependency installation or network operation occurs.
        temporary_venv = self.project / ".venv"
        launcher = Path(sys.base_prefix) / "Lib" / "venv" / "scripts" / "nt" / "python.exe"
        if launcher.is_file():
            (temporary_venv / "Scripts").mkdir(parents=True)
            shutil.copy2(launcher, temporary_venv / "Scripts" / "python.exe")
            (temporary_venv / "pyvenv.cfg").write_text(
                f"home = {sys.base_prefix}\ninclude-system-site-packages = false\n"
                f"version = {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}\n", encoding="utf-8")
        else:
            # Newer CPython distributions can relocate their bundled launcher.
            # The stdlib knows that layout; this still creates only a temp fixture.
            import venv
            venv.EnvBuilder(with_pip=False).create(temporary_venv)
        self.capture = self.root / "captured arguments.json"
        self.environment = os.environ.copy()
        self.environment["HELMET_WRAPPER_CAPTURE"] = str(self.capture)
        self.environment["HELMET_WRAPPER_EXIT"] = "0"
        self.environment["PYTHONPATH"] = str(self.project / "src")
        self.environment.pop("PYTHONHOME", None)
        self.outside = self.root / "unrelated working directory"
        self.outside.mkdir()

    def _input(self, name="görüntü [1] 研究.jpg"):
        path = self.outside / name
        path.write_bytes(b"synthetic input placeholder; never decoded")
        return path

    def _execute(self, command, *, environment=None):
        encoded = base64.b64encode(command.encode("utf-16-le")).decode("ascii")
        return subprocess.run(
            [self.powershell, "-NoLogo", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-EncodedCommand", encoded],
            cwd=self.outside, env=environment or self.environment, capture_output=True,
            encoding="utf-8", errors="replace", timeout=45,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

    def _invoke(self, name, arguments="", prefix=""):
        return self._execute(prefix + "\n& " + _ps_literal(self.scripts / name) + " " + arguments)

    def _captured(self):
        return json.loads(self.capture.read_text(encoding="utf-8"))

    def _option(self, arguments, name):
        return arguments[arguments.index(name) + 1]

    def test_all_desktop_scripts_parse_in_windows_powershell_51(self):
        command = "$ErrorActionPreference = 'Stop'; "
        command += "if ($PSVersionTable.PSVersion.Major -ne 5) { throw 'Expected Windows PowerShell 5.1' }; "
        command += "foreach ($scriptFile in (Get-ChildItem -LiteralPath " + _ps_literal(self.scripts) + " -Filter *.ps1)) { "
        command += "$tokens = $null; $parseErrors = $null; "
        command += "[System.Management.Automation.Language.Parser]::ParseFile($scriptFile.FullName, [ref]$tokens, [ref]$parseErrors) | Out-Null; "
        command += "if ($parseErrors.Count) { throw ($parseErrors | Out-String) } }"
        result = self._execute(command)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_image_wrapper_preserves_unicode_spaces_brackets_and_option_arguments(self):
        image = self._input()
        output = self.outside / "sonuç 研究 [new]"
        result = self._invoke("run_image_check.ps1", "-Image " + _ps_literal(image) + " -Direction rear -Rotate 90 -CpuThreads 2 -OutputDir " + _ps_literal(output))
        self.assertEqual(result.returncode, 0, result.stderr)
        captured = self._captured()
        arguments = captured["argv"]
        self.assertEqual(Path(self._option(arguments, "--image")), image)
        self.assertEqual(Path(self._option(arguments, "--output-dir")), output)
        self.assertEqual(Path(self._option(arguments, "--model")), self.project / "models" / "detector.onnx")
        self.assertEqual(self._option(arguments, "--direction"), "rear")
        self.assertEqual(self._option(arguments, "--rotate"), "90")
        self.assertEqual(self._option(arguments, "--cpu-threads"), "2")
        self.assertEqual(Path(captured["pythonpath"]), self.project / "src")

    def test_log_wrapper_preserves_optional_paths_and_new_default_output(self):
        runtime = self._input("çalışma [2].log")
        before, after = self._input("önce.txt"), self._input("sonra.txt")
        result = self._invoke("run_log_report.ps1", "-Runtime " + _ps_literal(runtime) + " -Before " + _ps_literal(before) + " -After " + _ps_literal(after))
        self.assertEqual(result.returncode, 0, result.stderr)
        arguments = self._captured()["argv"]
        self.assertEqual(Path(self._option(arguments, "--runtime")), runtime)
        self.assertEqual(Path(self._option(arguments, "--before")), before)
        self.assertEqual(Path(self._option(arguments, "--after")), after)
        output = Path(self._option(arguments, "--output-dir"))
        self.assertEqual(output.parent, self.project / "results")
        self.assertFalse(output.exists(), "Wrapper must leave creation of the NEW result directory to the module.")

    def test_video_decimal_start_is_invariant_in_turkish_culture(self):
        video = self._input("trafik ü [3].mp4")
        output, report = self.outside / "yeni çıktı.mp4", self.outside / "yeni rapor"
        prefix = "[System.Threading.Thread]::CurrentThread.CurrentCulture = [System.Globalization.CultureInfo]::GetCultureInfo('tr-TR')"
        result = self._invoke("run_video_demo.ps1", "-Video " + _ps_literal(video) + " -StartSeconds 1.5 -MaxFrames 12 -NoPreview -Direction rear -Output " + _ps_literal(output) + " -ReportDir " + _ps_literal(report), prefix)
        self.assertEqual(result.returncode, 0, result.stderr)
        arguments = self._captured()["argv"]
        self.assertEqual(arguments[0], "video-run")
        self.assertEqual(self._option(arguments, "--start-seconds"), "1.5")
        self.assertEqual(self._option(arguments, "--max-frames"), "12")
        self.assertIn("--no-preview", arguments)
        self.assertEqual(Path(self._option(arguments, "--video")), video)
        self.assertEqual(Path(self._option(arguments, "--report-dir")), report)
        self.assertTrue(self._option(arguments, "--config").endswith("webcam_rear.example.json"))

    def test_max_frames_zero_is_explicit_for_cli_whole_file_mapping(self):
        result = self._invoke("run_video_demo.ps1", "-Video " + _ps_literal(self._input("clip.mp4")) + " -MaxFrames 0 -NoPreview")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self._option(self._captured()["argv"], "--max-frames"), "0")

    def test_production_log_module_handles_unicode_paths_in_redirected_native_output(self):
        # Unlike the argv-only shim, this runs the dependency-free production
        # log reader on synthetic text, exercising real artifact paths/stdout.
        shutil.copy2(PROJECT / "src" / "helmetai_fcw" / "log_report.py",
                     self.project / "src" / "helmetai_fcw" / "log_report.py")
        runtime = self._input("kayıt 研究 [4].log")
        runtime.write_text(json.dumps({"event": "runtime_status", "frame_count": 1, "detections": []}), encoding="utf-8")
        output = self.outside / "rapor 研究 şğü [new]"
        # Do not depend on developer-shell encoding customizations.
        self.environment.pop("PYTHONUTF8", None)
        self.environment.pop("PYTHONIOENCODING", None)
        result = self._invoke("run_log_report.ps1", "-Runtime " + _ps_literal(runtime) + " -OutputDir " + _ps_literal(output))
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads((output / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(Path(report["runtime_source_path"]), runtime)
        self.assertEqual(report["runs"][0]["status_record_count"], 1)

    def test_native_python_failure_propagates_without_success_message(self):
        self.environment["HELMET_WRAPPER_EXIT"] = "41"
        result = self._invoke("run_log_report.ps1", "-Runtime " + _ps_literal(self._input("runtime.log")))
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(self.capture.exists(), "The failure must come from the native shim, not parameter binding.")
        self.assertIn("exit 41", result.stderr)
        self.assertNotIn("Results:", result.stdout)

    def test_environment_and_console_encoding_are_restored_when_native_python_fails(self):
        self.environment["HELMET_WRAPPER_EXIT"] = "42"
        restored = self.root / "restored environment.txt"
        command = ". " + _ps_literal(self.scripts / "desktop_common.ps1") + "; "
        command += "$env:PYTHONPATH = 'original-sentinel'; $env:PYTHONIOENCODING = 'cp1254'; "
        command += "[Console]::OutputEncoding = [System.Text.Encoding]::GetEncoding(1254); "
        command += "try { Invoke-HelmetPython @('-m','helmetai_fcw.log_report') } catch { }; "
        command += "$probe = @{ pythonpath = $env:PYTHONPATH; pythonioencoding = $env:PYTHONIOENCODING; "
        command += "console_code_page = [Console]::OutputEncoding.CodePage } | ConvertTo-Json -Compress; "
        command += "[System.IO.File]::WriteAllText(" + _ps_literal(restored) + ", $probe)"
        result = self._execute(command)
        self.assertEqual(result.returncode, 0, result.stderr)
        restored_settings = json.loads(restored.read_text(encoding="utf-8"))
        self.assertEqual(restored_settings["pythonpath"], "original-sentinel")
        self.assertEqual(restored_settings["pythonioencoding"], "cp1254")
        self.assertEqual(restored_settings["console_code_page"], 1254)
        captured = self._captured()
        self.assertEqual(Path(captured["pythonpath"]), self.project / "src")
        self.assertEqual(captured["pythonioencoding"], "utf-8")
        self.assertEqual(captured["stdout_encoding"], "utf-8")

    def test_setup_failed_fake_pip_does_not_claim_complete_or_access_network(self):
        self.environment["HELMET_WRAPPER_EXIT"] = "43"
        result = self._invoke("setup_webcam.ps1")
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(self.capture.exists(), "The deliberately failing fake pip module must run.")
        arguments = self._captured()["argv"]
        self.assertEqual(arguments[:2], ["install", "-r"])
        self.assertEqual(Path(arguments[2]), self.project / "requirements-webcam.txt")
        self.assertIn("Dependency installation failed", result.stderr)
        self.assertNotIn("Setup complete.", result.stdout)


if __name__ == "__main__":
    unittest.main()
