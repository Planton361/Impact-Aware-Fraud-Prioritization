"""Packaging boundary contracts."""

from __future__ import annotations

import os
import re
import runpy
import shutil
import subprocess
import sys
import tarfile
import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.contract

ROOT = Path(__file__).resolve().parents[2]


def test_public_cli_import_and_help_work_in_a_fresh_interpreter(
    tmp_path: Path,
) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (
            str(ROOT / "src"),
            environment.get("PYTHONPATH", ""),
        )
        if value
    )
    mpl_config = tmp_path / "mplconfig"
    mpl_config.mkdir()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["MPLCONFIGDIR"] = str(mpl_config)
    help_result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            "import fraud_detection; import fraud_detection.cli as cli; "
            "from fraud_detection.cli import main; "
            "from fraud_detection.cli.app import app; "
            "assert cli.__all__ == ['main']; assert cli.main is main; "
            "from typer.main import get_command; "
            "command=get_command(app); "
            "assert list(command.commands) == "
            "['setup', 'check', 'run', 'build', 'inspect']; "
            "help_paths=(('--help',), ('setup', '--help'), "
            "('check', '--help'), ('run', '--help'), "
            "('build', '--help'), ('inspect', '--help')); "
            "assert all(main(list(path)) == 0 for path in help_paths)",
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert help_result.returncode == 0, help_result.stderr
    for command in ("setup", "check", "run", "build", "inspect"):
        assert command in help_result.stdout
    for hidden_help in (
        "Run fast, read-only diagnostics without downloads",
        "Plan or run the complete frozen experiment",
        "Build all presentation-only artifacts in sequence",
        "Inspect known artifact and manifest paths",
    ):
        assert hidden_help not in help_result.stdout
    assert mpl_config.resolve().is_relative_to(tmp_path.resolve())
    assert not (tmp_path / "generated").exists()
    assert not (tmp_path / "outputs").exists()
    assert not (tmp_path / "thesis_build").exists()
    assert not list(tmp_path.rglob("__pycache__"))


def test_console_entrypoint_targets_package_cli(tmp_path: Path) -> None:
    project = tomllib.loads(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert project["build-system"]["requires"] == [
        "setuptools==83.0.0",
        "wheel==0.47.0",
    ]
    assert project["build-system"]["build-backend"] == "setuptools.build_meta"
    assert project["project"]["license"] == "MIT"
    assert project["project"]["license-files"] == ["LICENSE"]
    assert (
        "License :: OSI Approved :: MIT License"
        not in project["project"]["classifiers"]
    )
    assert project["project"]["version"] == "0.1.0"
    assert project["project"]["requires-python"] == ">=3.12,<3.13"
    assert project["project"]["optional-dependencies"]["dev"] == [
        "pytest==9.1.1",
        "ruff==0.16.1",
    ]

    runtime_requirements = [
        line.strip()
        for line in (
            ROOT / "environment" / "final_experiment_requirements.txt"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert runtime_requirements == [
        "numpy==2.5.1",
        "pandas==3.0.5",
        "scipy==1.18.0",
        "scikit-learn==1.9.0",
        "lightgbm==4.7.0",
        "pyarrow==25.0.0",
        "matplotlib==3.11.1",
        "pillow==12.3.0",
        "typer==0.27.0",
        "rich==15.0.0",
    ]
    assert all("pytest" not in line and "build" not in line for line in runtime_requirements)
    release_requirements = [
        line.strip()
        for line in (
            ROOT / "environment" / "release_requirements.txt"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert release_requirements == ["build==1.5.0"]

    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    for required in (
        "include pyproject.toml",
        "include README.md",
        "include LICENSE",
        "include CITATION.cff",
        "include MANIFEST.in",
        "recursive-include src/fraud_detection *.py",
        "prune .github",
        "prune tests",
        "prune scripts",
        "prune environment",
        "prune docs",
        "prune data",
        "prune reference_results",
        "prune generated",
        "prune outputs",
        "prune thesis_build",
        "exclude AGENTS.md",
        "global-exclude __pycache__",
        "global-exclude *.py[cod]",
    ):
        assert required in manifest

    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    for pattern in ("*.sh", "*.yml", "*.yaml", "*.cff", "*.in"):
        assert f"{pattern} text eol=lf" in attributes
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "/data/creditcard.csv" in ignore
    assert "/data/.creditcard-download-*/" in ignore
    assert "/.tools/" in ignore

    assert (
        project["project"]["scripts"]["fraud-detection"]
        == "fraud_detection.cli:main"
    )

    bootstrap_namespace = runpy.run_path(
        str(ROOT / "scripts" / "bootstrap_local_environment.py")
    )
    bootstrap_roots: list[Path] = []

    def mocked_bootstrap(root: Path) -> int:
        bootstrap_roots.append(root)
        return 0

    bootstrap_entrypoint = bootstrap_namespace["main"]
    bootstrap_entrypoint.__globals__["bootstrap_main"] = mocked_bootstrap
    assert bootstrap_entrypoint() == 0
    assert bootstrap_roots == [ROOT]

    old_workflow_path = ROOT / ".github" / "workflows" / "tests.yml"
    workflow_path = ROOT / ".github" / "workflows" / "ci.yml"
    assert not old_workflow_path.exists()
    assert workflow_path.exists()
    workflow = workflow_path.read_text(encoding="utf-8")
    assert workflow.startswith("name: CI\n")
    assert "workflow_dispatch:" in workflow
    assert "pull_request_target" not in workflow
    assert "workflow_run" not in workflow
    assert "schedule:" not in workflow
    assert "permissions:\n  contents: read" in workflow
    assert "cancel-in-progress: true" in workflow
    for setting in (
        'PIP_DISABLE_PIP_VERSION_CHECK: "1"',
        'PIP_NO_INPUT: "1"',
        'PYTHONDONTWRITEBYTECODE: "1"',
        'PYTHONHASHSEED: "0"',
    ):
        assert setting in workflow
    checkout_sha = "3d3c42e5aac5ba805825da76410c181273ba90b1"
    setup_python_sha = "5fda3b95a4ea91299a34e894583c3862153e4b97"
    assert workflow.count(f"actions/checkout@{checkout_sha}") == 3
    assert workflow.count(f"actions/setup-python@{setup_python_sha}") == 3
    for old_sha in (
        "11d5960a326750d5838078e36cf38b85af677262",
        "a26af69be951a213d495a4c3e4e4022e16d87065",
    ):
        assert old_sha not in workflow
    assert workflow.count("fetch-depth: 1") == 3
    assert workflow.count("fetch-tags: false") == 3
    assert workflow.count("persist-credentials: false") == 3
    assert workflow.count("timeout-minutes:") == 3
    assert "runs-on: ubuntu-24.04" in workflow
    assert "runs-on: ubuntu-latest" not in workflow
    assert "windows-latest" not in workflow
    assert "windows-2025" in workflow
    job_block = workflow.split("jobs:\n", 1)[1]
    assert re.findall(r"^  ([A-Za-z0-9_-]+):$", job_block, re.MULTILINE) == [
        "fast-checks",
        "distributions",
        "platform-smoke",
    ]
    assert "fail-fast: false" in workflow
    assert "          - ubuntu-24.04\n          - windows-2025" in workflow
    assert workflow.count("cache-dependency-path:") == 3
    assert workflow.count("MPLCONFIGDIR: ${{ runner.temp }}/matplotlib") >= 3
    for cache_authority in (
        "environment/final_experiment_requirements.txt",
        "environment/bootstrap_requirements.txt",
        "pyproject.toml",
    ):
        assert workflow.count(cache_authority) >= 3
    for cache_authority in (
        "environment/release_requirements.txt",
        "MANIFEST.in",
    ):
        assert cache_authority in workflow
    assert workflow.count('python-version: "3.12"') == 3
    assert workflow.count('architecture: "x64"') == 3
    assert workflow.count('python -m pip install "pip==26.2"') == 3
    assert workflow.count("python -m build --sdist") == 1
    assert workflow.count("python -m build --wheel") == 1
    assert "tar -xzf" in workflow
    assert "fraud-detection run --profile smoke-synthetic --dry-run" in workflow
    assert all(
        "--dry-run" in line
        for line in workflow.splitlines()
        if "fraud-detection run --profile smoke-synthetic" in line
    )
    assert "creditcard.csv" not in workflow
    assert "upload-artifact" not in workflow
    assert "download-artifact" not in workflow
    assert "ACTIONS_ALLOW_USE_UNSECURE_NODE_VERSION" not in workflow
    action_refs = re.findall(
        r"^\s*- uses: ([^#\s]+)(?:\s+#.*)?$", workflow, re.MULTILINE
    )
    assert len(action_refs) == 6
    assert all("@" in reference and "@v" not in reference for reference in action_refs)
    assert "- name: Public command help" in workflow
    for command in ("setup", "check", "run", "build", "inspect"):
        assert f"fraud-detection {command} --help" in workflow
    for legacy in (
        "scripts/run_amount_gain_ranker.py",
        "fraud-detection doctor",
        "fraud-detection experiment",
        "fraud-detection presentation",
        "fraud-detection artifacts",
    ):
        assert legacy not in workflow

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    windows_happy_path = (
        ".\\scripts\\bootstrap.ps1\n"
        ".\\.venv\\Scripts\\fraud-detection.exe run --profile "
        "smoke-synthetic\n"
        ".\\.venv\\Scripts\\fraud-detection.exe build "
        "generated/runs/smoke-synthetic"
    )
    posix_happy_path = (
        "./scripts/bootstrap.sh\n"
        ".venv/bin/fraud-detection run --profile smoke-synthetic\n"
        ".venv/bin/fraud-detection build generated/runs/smoke-synthetic"
    )
    assert windows_happy_path in readme
    assert posix_happy_path in readme
    assert readme.index(windows_happy_path) < readme.index("fraud-detection check")
    for legacy in (
        "fraud-detection doctor",
        "fraud-detection experiment",
        "fraud-detection presentation",
        "fraud-detection artifacts",
        "bootstrap_local_environment.py",
        "run_amount_gain_ranker.py",
    ):
        assert legacy not in readme

    powershell_wrapper = (ROOT / "scripts" / "bootstrap.ps1").read_text(
        encoding="utf-8"
    )
    posix_wrapper = (ROOT / "scripts" / "bootstrap.sh").read_text(
        encoding="utf-8"
    )
    for wrapper in (powershell_wrapper, posix_wrapper):
        assert wrapper.count("bootstrap_local_environment.py") == 1
        assert "run --profile" not in wrapper
        assert "pip install" not in wrapper
        assert "uv venv" not in wrapper
        assert "uv sync" not in wrapper
        assert "uv pip" not in wrapper
        assert "uv python update-shell" not in wrapper
        assert "uv self update" not in wrapper
        assert "--default" not in wrapper
        assert "UV_PYTHON_INSTALL_DIR" in wrapper
        assert "UV_CACHE_DIR" in wrapper
        assert "UV_MANAGED_PYTHON" in wrapper
        assert "UV_PYTHON_DOWNLOADS" in wrapper
        assert "UV_PYTHON_INSTALL_BIN" in wrapper
        assert "UV_PYTHON_INSTALL_REGISTRY" in wrapper
        assert "UV_NO_CONFIG" in wrapper
        assert "cpython@" in wrapper
        assert "--managed-python" in wrapper
        assert "bootstrap_toolchain.lock" in wrapper
    assert "Set-ExecutionPolicy" not in powershell_wrapper
    assert "Invoke-WebRequest" in powershell_wrapper
    assert "Get-FileHash" in powershell_wrapper
    assert "Expand-Archive" in powershell_wrapper
    assert "$env:PATH" not in powershell_wrapper
    assert "Set-ItemProperty" not in powershell_wrapper
    assert "New-ItemProperty" not in powershell_wrapper
    assert "HKCU:" not in powershell_wrapper
    assert "exit $exitCode" in powershell_wrapper
    assert "curl --fail --location --proto '=https' --tlsv1.2" in posix_wrapper
    assert "wget --https-only --secure-protocol=TLSv1_2" in posix_wrapper
    assert "sha256sum" in posix_wrapper
    assert "shasum -a 256" in posix_wrapper
    assert "tar -xzf" in posix_wrapper
    assert "\nPATH=" not in posix_wrapper
    assert "profile" not in posix_wrapper.lower()
    assert "brew" not in posix_wrapper.lower()
    assert "winget" not in powershell_wrapper.lower()
    assert "choco" not in powershell_wrapper.lower()
    assert "scoop" not in powershell_wrapper.lower()
    assert "| sh" not in posix_wrapper
    assert "| iex" not in powershell_wrapper.lower()

    lock_lines = (
        ROOT / "environment" / "bootstrap_toolchain.lock"
    ).read_text(encoding="utf-8").splitlines()
    assert lock_lines == [
        "schema=1",
        "uv_version=0.12.1",
        "cpython_version=3.12.13",
        "asset=windows|x86_64|uv-x86_64-pc-windows-msvc.zip|8fcb0cb46e1229065e344758980924e569bef5882ef45f46fada8fb24e06b74a",
        "asset=windows|aarch64|uv-aarch64-pc-windows-msvc.zip|9bc7c18e616230fa2dc6fb24bc3afde18a95c2b5c9433de747e9502c66041568",
        "asset=macos|x86_64|uv-x86_64-apple-darwin.tar.gz|69d9f9a00337f25a50dcb13882052da08b8469bac11091c98c5694c3c6721467",
        "asset=macos|aarch64|uv-aarch64-apple-darwin.tar.gz|77d2906988e8074fd43f2f329ec452ebbf9b0c257ba1c66451c71de70a6baf42",
        "asset=linux-gnu|x86_64|uv-x86_64-unknown-linux-gnu.tar.gz|90b2f223fb69d19db49e117da601f64978593417988530aa733d456141b4bcbb",
        "asset=linux-gnu|aarch64|uv-aarch64-unknown-linux-gnu.tar.gz|769d373e146692c639b5fbaae33b331c297a32e03d30448772051902df52bbf4",
    ]

    sh = shutil.which("sh")
    if os.name == "posix" and sh is not None:
        clone = tmp_path / "workspace with spaces"
        scripts = clone / "scripts"
        environment_dir = clone / "environment"
        fake_bin = tmp_path / "native tools"
        scripts.mkdir(parents=True)
        environment_dir.mkdir()
        fake_bin.mkdir()
        shutil.copy2(ROOT / "scripts" / "bootstrap.sh", scripts)
        shutil.copy2(
            ROOT / "environment" / "bootstrap_toolchain.lock",
            environment_dir,
        )

        asset_name = "uv-x86_64-unknown-linux-gnu.tar.gz"
        archive_root = asset_name.removesuffix(".tar.gz")
        uv_source = tmp_path / "uv"
        uv_source.write_text(
            "#!/bin/sh\n"
            "if [ \"${1:-}\" = \"--version\" ]; then\n"
            "  printf '%s\\n' \"$MOCK_UV_VERSION\" >> \"$MOCK_UV_VERSION_RECORD\"\n"
            "  printf '%s\\n' \"$MOCK_UV_VERSION\"\n"
            "  exit \"$MOCK_UV_EXIT\"\n"
            "fi\n"
            "if [ \"${2:-}\" = \"python\" ]; then\n"
            "  printf '%s\\n' \"$UV_PYTHON_INSTALL_DIR|$UV_CACHE_DIR|$UV_MANAGED_PYTHON|$UV_PYTHON_DOWNLOADS|$UV_PYTHON_INSTALL_BIN|$UV_PYTHON_INSTALL_REGISTRY|$UV_NO_CONFIG\" >> \"$MOCK_UV_RECORD\"\n"
            "fi\n"
            "if [ \"${2:-}\" = \"python\" ] && [ \"${3:-}\" = \"install\" ]; then\n"
            "  target=\"$UV_PYTHON_INSTALL_DIR/cpython-3.12.13/bin\"\n"
            "  mkdir -p \"$target\"; cp \"$MOCK_BASE_PYTHON\" \"$target/python\"; chmod +x \"$target/python\"; exit 0\n"
            "fi\n"
            "if [ \"${2:-}\" = \"python\" ] && [ \"${3:-}\" = \"find\" ]; then\n"
            "  printf '%s\\n' \"$UV_PYTHON_INSTALL_DIR/cpython-3.12.13/bin/python\"; exit 0\n"
            "fi\n"
            "exit 0\n",
            encoding="ascii",
        )
        uv_source.chmod(0o755)
        archive = tmp_path / asset_name
        with tarfile.open(archive, "w:gz") as created:
            created.add(uv_source, arcname=f"{archive_root}/uv")

        base_python = tmp_path / "base-python"
        base_python.write_text(
            "#!/bin/sh\n"
            "if [ \"${1:-}\" = \"-c\" ]; then exit 0; fi\n"
            "printf '%s\\n' \"$@\" > \"$MOCK_BOOTSTRAP_RECORD\"\n"
            "exit 0\n",
            encoding="ascii",
        )
        base_python.chmod(0o755)
        (fake_bin / "uname").write_text(
            "#!/bin/sh\n"
            "case \"${1:-}\" in -s) echo Linux ;; -m) echo x86_64 ;; esac\n",
            encoding="ascii",
        )
        (fake_bin / "getconf").write_text("#!/bin/sh\necho 'glibc 2.40'\n", encoding="ascii")
        (fake_bin / "sha256sum").write_text(
            "#!/bin/sh\n"
            "echo '90b2f223fb69d19db49e117da601f64978593417988530aa733d456141b4bcbb  ' \"$1\"\n",
            encoding="ascii",
        )
        (fake_bin / "curl").write_text(
            "#!/bin/sh\n"
            "while [ \"$#\" -gt 0 ]; do\n"
            "  if [ \"$1\" = \"--output\" ]; then output=$2; shift 2; else shift; fi\n"
            "done\n"
            "cp \"$MOCK_UV_ARCHIVE\" \"$output\"\n",
            encoding="ascii",
        )
        for command in ("uname", "getconf", "sha256sum", "curl"):
            (fake_bin / command).chmod(0o755)

        uv_record = tmp_path / "uv-record"
        bootstrap_record = tmp_path / "bootstrap-record"
        version_record = tmp_path / "uv-version-record"
        environment = os.environ.copy()
        environment.update(
            {
                "PATH": str(fake_bin) + os.pathsep + environment["PATH"],
                "MOCK_UV_ARCHIVE": str(archive),
                "MOCK_BASE_PYTHON": str(base_python),
                "MOCK_UV_RECORD": str(uv_record),
                "MOCK_UV_VERSION": "uv 0.12.1 (329541a50 2026-07-31 aarch64-apple-darwin)",
                "MOCK_UV_EXIT": "0",
                "MOCK_UV_VERSION_RECORD": str(version_record),
                "MOCK_BOOTSTRAP_RECORD": str(bootstrap_record),
            }
        )

        def create_posix_fixture(
            name: str,
            version_output: str,
            version_exit: int = 0,
        ) -> tuple[Path, dict[str, str], Path]:
            fixture_clone = tmp_path / name
            fixture_scripts = fixture_clone / "scripts"
            fixture_environment_dir = fixture_clone / "environment"
            fixture_scripts.mkdir(parents=True)
            fixture_environment_dir.mkdir()
            shutil.copy2(ROOT / "scripts" / "bootstrap.sh", fixture_scripts)
            shutil.copy2(
                ROOT / "environment" / "bootstrap_toolchain.lock",
                fixture_environment_dir,
            )
            fixture_platform_dir = (
                fixture_clone / ".tools" / "uv" / "0.12.1" / "linux-gnu-x86_64"
            )
            fixture_platform_dir.mkdir(parents=True)
            shutil.copy2(archive, fixture_platform_dir / asset_name)
            fixture_version_record = tmp_path / f"{name}-uv-version-record"
            fixture_environment = environment.copy()
            fixture_environment.update(
                {
                    "MOCK_UV_VERSION": version_output,
                    "MOCK_UV_EXIT": str(version_exit),
                    "MOCK_UV_VERSION_RECORD": str(fixture_version_record),
                }
            )
            return fixture_clone, fixture_environment, fixture_version_record

        def run_bootstrap(
            fixture_clone: Path,
            fixture_environment: dict[str, str],
        ) -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                [sh, str(fixture_clone / "scripts" / "bootstrap.sh")],
                cwd=fixture_clone,
                env=fixture_environment,
                check=False,
                capture_output=True,
                text=True,
            )

        syntax = subprocess.run(
            [sh, "-n", str(scripts / "bootstrap.sh")],
            check=False,
            capture_output=True,
            text=True,
        )
        assert syntax.returncode == 0, syntax.stderr
        # The first run validates the staged and finally installed binaries.
        result = run_bootstrap(clone, environment)
        assert result.returncode == 0, result.stderr
        assert version_record.read_text(encoding="ascii").splitlines() == [
            environment["MOCK_UV_VERSION"],
            environment["MOCK_UV_VERSION"],
        ]
        assert bootstrap_record.read_text(encoding="utf-8").strip().endswith(
            "scripts/bootstrap_local_environment.py"
        )
        expected_uv_environment = (
            f"{clone / '.tools' / 'python'}|{clone / '.tools' / 'cache'}|1|manual|0|0|1"
        ).replace("\\", "/")
        assert [
            line.replace("\\", "/")
            for line in uv_record.read_text(encoding="utf-8").splitlines()
        ] == [expected_uv_environment, expected_uv_environment]

        for index, version_output in enumerate(
            (
                "uv 0.12.1",
                "uv 0.12.1 (329541a50 2026-07-31 x86_64-pc-windows-msvc)",
                "uv 0.12.1 (329541a50 2026-07-31 x86_64-unknown-linux-gnu)",
            )
        ):
            accepted_clone, accepted_environment, _ = create_posix_fixture(
                f"accepted-version-{index}",
                version_output,
            )
            accepted = run_bootstrap(accepted_clone, accepted_environment)
            assert accepted.returncode == 0, accepted.stderr

        venv_python = clone / ".venv" / "bin" / "python"
        venv_python.parent.mkdir(parents=True)
        shutil.copy2(base_python, venv_python)
        venv_python.chmod(0o755)
        marker = clone / ".venv" / "keep"
        marker.write_text("reuse\n", encoding="ascii")
        # Reuse validates an existing binary and then the final installed binary.
        repeat = run_bootstrap(clone, environment)
        assert repeat.returncode == 0, repeat.stderr
        assert marker.read_text(encoding="ascii") == "reuse\n"

        venv_python.write_text("#!/bin/sh\nexit 1\n", encoding="ascii")
        venv_python.chmod(0o755)
        replacement = run_bootstrap(clone, environment)
        assert replacement.returncode == 0, replacement.stderr
        assert not (clone / ".venv").exists()
        assert version_record.read_text(encoding="ascii").splitlines() == [
            environment["MOCK_UV_VERSION"]
        ] * 6

        environment["MOCK_UV_VERSION"] = "uv 0.12.2"
        existing_invalid = run_bootstrap(clone, environment)
        assert existing_invalid.returncode != 0
        assert "FD-UV-VERSION" in existing_invalid.stderr
        assert version_record.read_text(encoding="ascii").splitlines()[-2:] == [
            "uv 0.12.2",
            "uv 0.12.2",
        ]

        rejected_outputs = (
            "uv 0.12.2",
            "uv 0.12.10",
            "prefix uv 0.12.1",
            "uv 0.12.1 suffix",
            "uv 0.12.1 ()",
            "uv 0.12.1 (metadata) suffix",
            "uv 0.12.1 (outer(inner))",
            "uv 0.12.1 (metadata",
            "uv 0.12.1 metadata)",
            "uv 0.12.1 (one) (two)",
            "uv 0.12.1\nuv 0.12.1",
            "uv 0.12.1\r",
        )
        for index, version_output in enumerate(rejected_outputs):
            rejected_clone, rejected_environment, _ = create_posix_fixture(
                f"rejected-version-{index}",
                version_output,
            )
            rejected = run_bootstrap(rejected_clone, rejected_environment)
            assert rejected.returncode != 0
            assert "FD-UV-VERSION" in rejected.stderr
            assert rejected.stdout == ""

        failed_clone, failed_environment, _ = create_posix_fixture(
            "failed-version-command",
            "uv 0.12.1",
            version_exit=7,
        )
        failed = run_bootstrap(failed_clone, failed_environment)
        assert failed.returncode != 0
        assert "FD-UV-VERSION" in failed.stderr
