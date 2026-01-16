#!/usr/bin/env python3
"""
Apply CoreBench fix packages, run HAL evaluations, and re-grade the new traces.

Usage:
    python scripts/run_corebench_fixes.py \
        --fixes-root fixes/corebench_hard \
        --agent-dir hal-harness/agents/hal_generalist_agent \
        --agent-args agent_args.azure.json \
        --rubric-model azure_openai:o3-mini
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import shlex
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
import re
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "hal-harness"))

DEFAULT_COREBENCH_DATA = REPO_ROOT / "hal-harness" / "hal" / "benchmarks" / "corebench" / "core_test.json"
SMOLAGENTS_SRC = REPO_ROOT / "hal-harness" / "agents" / "open_deep_research" / "src" / "smolagents"

from hal.debugger.fix_loader import (  # type: ignore
    load_fix_package,
    apply_agent_overlay,
    apply_agent_patch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-run HAL eval for CoreBench fixes.")
    parser.add_argument(
        "--fixes-root",
        default=str(REPO_ROOT / "fixes" / "corebench_hard"),
        help="Directory containing per-capsule fix packages (default: fixes/corebench_hard).",
    )
    parser.add_argument(
        "--agent-dir",
        default=str(REPO_ROOT / "hal-harness" / "agents" / "hal_generalist_agent"),
        help="Path to the base agent directory.",
    )
    parser.add_argument(
        "--agent-args",
        default=str(REPO_ROOT / "agent_args.azure.json"),
        help="Path to the JSON file with agent arguments (model_name, etc.).",
    )
    parser.add_argument(
        "--rubric-model",
        required=False,
        help="Model identifier for rubric evaluation (e.g., azure_openai:o3-mini or openai:o3-mini).",
    )
    parser.add_argument(
        "--skip-rubrics",
        action="store_true",
        help="Skip running main.py evaluate (per-task and merged trace).",
    )
    parser.add_argument(
        "--benchmark",
        default="corebench_hard",
        help="Benchmark variant to evaluate (default: corebench_hard).",
    )
    parser.add_argument(
        "--corebench-dataset",
        help=(
            "Path to the CoreBench task dataset JSON (core_test.json). "
            "If omitted, auto-detect under hal-harness/hal/benchmarks."
        ),
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Optional prefix to prepend to per-capsule run_ids and the WANDB project/group (default: empty).",
    )
    parser.add_argument(
        "--merged-trace-output",
        help="Optional output path for the merged trace JSON (must end with .json).",
    )
    parser.add_argument(
        "--merged-run-id",
        help="Optional run_id to store in the merged trace metadata (defaults to auto-generated).",
    )
    parser.add_argument(
        "--docker",
        action="store_true",
        help="Run hal-eval with the --docker flag.",
    )
    parser.add_argument(
        "--vm",
        action="store_true",
        help="Run hal-eval with the --vm flag (use VM execution, e.g. for GPU tasks).",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        help=(
            "Optionally override WANDB_MODE for HAL runs. "
            "Use 'online' to enable Weave trace fetching, 'offline' to avoid network, "
            "or 'disabled' to turn off wandb/weave entirely."
        ),
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Do not delete temporary agent directories (useful for debugging).",
    )
    parser.add_argument(
        "--max-parallel-capsules",
        type=int,
        default=1,
        help="Maximum number of capsules to run concurrently within this rerun (default: %(default)s).",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip installing agent requirements before running hal-eval.",
    )
    parser.add_argument(
        "--task-id",
        dest="task_ids",
        action="append",
        help="Only run the specified capsule ID(s). Can be repeated.",
    )
    parser.add_argument(
        "--isolated-env",
        action="store_true",
        help="Create a fresh isolated conda environment for this run (recommended for clean evaluations).",
    )
    parser.add_argument(
        "--isolated-env-python",
        default="3.12",
        help="Python version to use for isolated conda environment (default: %(default)s).",
    )
    parser.add_argument(
        "--keep-isolated-env",
        action="store_true",
        help="Do not delete the isolated conda environment after the run (useful for debugging).",
    )
    return parser.parse_args()


def load_agent_args(path: Path) -> Dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Agent args JSON must be an object, got {type(data)}")
    return data


def normalize_value(value: object) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    if value is None:
        return "null"
    return str(value)

def sanitize_tls_env(env: Dict[str, str]) -> None:
    """
    Some environments accidentally set invalid CA bundle paths (e.g. "...cacert.pem[/]"),
    which breaks Weave/W&B/requests TLS and can cause long retries/hangs.
    Remove obviously broken overrides and allow defaults to take over.
    """
    for key in ("REQUESTS_CA_BUNDLE", "SSL_CERT_FILE", "CURL_CA_BUNDLE"):
        value = env.get(key)
        if not value:
            continue
        if "[/]" in value or "[" in value or "]" in value:
            env.pop(key, None)

def _split_channels(raw: str | None) -> List[str]:
    if not raw:
        return []
    parts: List[str] = []
    for item in shlex.split(raw):
        parts.extend(x.strip() for x in item.split(",") if x.strip())
    return [p for p in parts if p]


def _split_packages(raw: str | None) -> List[str]:
    if not raw:
        return []
    aliases = {
        # Some older fix packages used this name; conda-forge ships `xorg-server-xvfb`.
        "xorg-xvfb": "xorg-server-xvfb",
    }
    packages: List[str] = []
    for token in shlex.split(raw):
        token = token.strip()
        if not token:
            continue
        packages.append(aliases.get(token, token))
    return packages


def _conda_executable() -> str | None:
    # Prefer mamba over conda for faster dependency resolution
    # Also check in current env's bin directory
    env_bin = Path(sys.prefix) / "bin"
    mamba_in_env = env_bin / "mamba"
    if mamba_in_env.exists():
        return str(mamba_in_env)
    return shutil.which("mamba") or shutil.which("conda") or os.environ.get("CONDA_EXE")


def _conda_spec_name(spec: str) -> str:
    for sep in ("==", ">=", "<=", "=", ">", "<"):
        if sep in spec:
            return spec.split(sep, 1)[0].strip()
    return spec.strip()


def _conda_available_package_names(conda: str, *, channels: List[str], names: List[str]) -> set[str] | None:
    unique = sorted({name for name in names if name})
    if not unique:
        return set()
    cmd = [conda, "search", "--json"]
    for channel in channels:
        cmd += ["-c", channel]
    cmd += unique
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        return None
    try:
        payload = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        return None
    available: set[str] = set()
    # Handle mamba format: {"result": {"pkgs": [{"name": "...", ...}, ...]}}
    if "result" in payload and isinstance(payload.get("result"), dict):
        pkgs = payload["result"].get("pkgs", [])
        if isinstance(pkgs, list):
            for pkg in pkgs:
                if isinstance(pkg, dict) and pkg.get("name") in unique:
                    available.add(pkg["name"])
        return available
    # Handle conda format: {"package_name": [...], ...}
    for name in unique:
        entries = payload.get(name)
        if isinstance(entries, list) and entries:
            available.add(name)
    return available


def _drop_missing_packages_from_conda_output(output: str) -> List[str]:
    if "PackagesNotFoundError" not in output:
        return []
    missing: List[str] = []
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("- "):
            missing.append(line[2:].strip())
    return missing


def _install_conda_packages(
    *,
    env_name: str,
    channels: List[str],
    packages: List[str],
) -> None:
    conda = _conda_executable()
    conda_name = Path(conda).name if conda else "conda"
    if not conda:
        raise FileNotFoundError("conda executable not found (needed for HAL_CONDA_PACKAGES).")
    if not packages:
        print(f"[setup][conda] No packages to install")
        return

    print(f"[setup][conda] Using {conda_name} from: {conda}")
    print(f"[setup][conda] Preparing to install {len(packages)} packages into env '{env_name}'")
    print(f"[setup][conda] Channels: {', '.join(channels)}")
    print(f"[setup][conda] Packages: {', '.join(packages[:10])}{'...' if len(packages) > 10 else ''}")

    lock_dir = REPO_ROOT / "log" / "conda_installs"
    lock_dir.mkdir(parents=True, exist_ok=True)
    cache_key = json.dumps(
        {"env": env_name, "channels": channels, "packages": packages},
        sort_keys=True,
    ).encode("utf-8")
    cache_id = hashlib.sha256(cache_key).hexdigest()[:16]
    sentinel = lock_dir / f"{env_name}_{cache_id}.ok"
    lock_path = lock_dir / f"{env_name}.lock"

    if sentinel.exists():
        print(f"[setup][conda] Cache hit: packages already installed (sentinel: {sentinel.name})")
        return

    print(f"[setup][conda] Cache miss: will install packages (cache_id: {cache_id})")

    # Best-effort filter: skip package specs whose names are not present in the configured channels.
    # Some environments mirror only a subset of conda-forge (e.g. no xorg packages).
    print(f"[setup][conda] Checking package availability in channels...")
    names = [_conda_spec_name(spec) for spec in packages]
    available = _conda_available_package_names(conda, channels=channels, names=names)
    if available is not None:
        print(f"[setup][conda] Found {len(available)}/{len(names)} packages available")
        filtered = [spec for spec in packages if _conda_spec_name(spec) in available]
        dropped = sorted({spec for spec in packages if spec not in filtered})
        if dropped:
            print(f"[setup][conda] Skipping unavailable package specs: {', '.join(dropped)}")
        packages = filtered
        if not packages:
            print(f"[setup][conda] No packages available to install, creating sentinel")
            sentinel.write_text("ok (no available packages)\n", encoding="utf-8")
            return
    else:
        print(f"[setup][conda] Could not check availability, proceeding with all packages")

    print(f"[setup][conda] Acquiring install lock...")
    lock_handle = lock_path.open("a", encoding="utf-8")
    try:
        try:
            import fcntl  # type: ignore

            # Serialize conda installs across concurrent specs without hanging forever on flock().
            wait_count = 0
            while True:
                try:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    wait_count += 1
                    if wait_count == 1:
                        print(f"[setup][conda] Waiting for lock (another install in progress)...")
                    elif wait_count % 20 == 0:  # Every 10 seconds
                        print(f"[setup][conda] Still waiting for lock ({wait_count * 0.5:.0f}s elapsed)...")
                    time.sleep(0.5)
            if wait_count > 0:
                print(f"[setup][conda] Lock acquired after {wait_count * 0.5:.1f}s")
            else:
                print(f"[setup][conda] Lock acquired immediately")
        except Exception:
            print(f"[setup][conda] Lock not available on this platform, proceeding without lock")

        if sentinel.exists():
            print(f"[setup][conda] Another process already installed packages, skipping")
            return

        def run_install(current: List[str]) -> subprocess.CompletedProcess[str]:
            cmd = [conda, "install", "-y", "-n", env_name]
            for channel in channels:
                cmd += ["-c", channel]
            cmd += current
            print(f"[setup][conda] Running: {' '.join(map(str, cmd))}")
            print(f"[setup][conda] Installing {len(current)} packages (this may take several minutes)...")
            start_time = time.time()
            result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
            elapsed = time.time() - start_time
            print(f"[setup][conda] Install command finished in {elapsed:.1f}s (exit code: {result.returncode})")
            return result

        remaining = list(packages)
        for attempt in range(2):
            print(f"[setup][conda] Attempt {attempt + 1}/2: installing {len(remaining)} packages...")
            proc = run_install(remaining)
            if proc.returncode == 0:
                print(f"[setup][conda] SUCCESS: All packages installed successfully")
                print(f"[setup][conda] Creating sentinel file: {sentinel.name}")
                sentinel.write_text("ok\n", encoding="utf-8")
                return
            output = (proc.stdout or "") + "\n" + (proc.stderr or "")
            missing = _drop_missing_packages_from_conda_output(output)
            if not missing:
                print(f"[setup][conda] FAILED: Install failed with no recoverable missing packages")
                print(f"[setup][conda] stdout: {proc.stdout[:500] if proc.stdout else '(empty)'}...")
                print(f"[setup][conda] stderr: {proc.stderr[:500] if proc.stderr else '(empty)'}...")
                raise subprocess.CalledProcessError(proc.returncode, proc.args, output=proc.stdout, stderr=proc.stderr)
            missing_names = {_conda_spec_name(item) for item in missing if item}
            new_remaining = [
                spec for spec in remaining if _conda_spec_name(spec) not in missing_names and spec not in missing
            ]
            if new_remaining == remaining:
                print(f"[setup][conda] FAILED: Could not reduce package list, giving up")
                raise subprocess.CalledProcessError(proc.returncode, proc.args, output=proc.stdout, stderr=proc.stderr)
            print(f"[setup][conda] Dropping {len(missing_names)} missing packages and retrying: {', '.join(sorted(missing_names))}")
            remaining = new_remaining
            if not remaining:
                print(f"[setup][conda] All packages were unavailable, creating sentinel")
                sentinel.write_text("ok (all packages unavailable)\n", encoding="utf-8")
                return

        print(f"[setup][conda] FAILED: Exhausted all retry attempts")
        raise subprocess.CalledProcessError(proc.returncode, proc.args, output=proc.stdout, stderr=proc.stderr)
    finally:
        try:
            lock_handle.close()
        except Exception:
            pass


def _python_can_import(module: str) -> bool:
    proc = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0


def ensure_hal_cli_dependencies(env_name: str, channels: List[str]) -> None:
    # hal.cli requires click at import-time, but it is not consistently present in all envs.
    if _python_can_import("click"):
        return
    print("[setup][hal] Missing python module 'click'; attempting to install via conda.")
    try:
        _install_conda_packages(env_name=env_name, channels=channels or ["conda-forge"], packages=["click"])
        if _python_can_import("click"):
            return
    except Exception as exc:
        print(f"[setup][hal] Conda install for click failed: {exc}")
    print("[setup][hal] Falling back to pip install click.")
    subprocess.run([sys.executable, "-m", "pip", "install", "click"], check=True, cwd=REPO_ROOT)


def build_hal_eval_cmd(
    *,
    benchmark: str,
    agent_name: str,
    agent_dir: Path,
    agent_args: Dict[str, object],
    run_id: str,
    docker: bool,
    vm: bool,
) -> List[str]:
    cmd: List[str] = [
        sys.executable,
        "-m",
        "hal.cli",
        "--benchmark", benchmark,
        "--agent_name", agent_name,
        "--agent_function", "main.run",
        "--agent_dir", str(agent_dir),
        "--run_id", run_id,
        "--max_concurrent", "1",
    ]
    for key, value in agent_args.items():
        cmd += ["-A", f"{key}={normalize_value(value)}"]
    if docker:
        cmd.append("--docker")
    if vm:
        cmd.append("--vm")
    return cmd


def copy_agent_with_fix(base_dir: Path, task_id: str, fix_root: Path, *, benchmark: str) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="corebench_fix_"))
    shutil.copytree(base_dir, temp_dir, dirs_exist_ok=True)
    if SMOLAGENTS_SRC.exists():
        shutil.copytree(SMOLAGENTS_SRC, temp_dir / "smolagents", dirs_exist_ok=True)
    fix = load_fix_package(task_id, fixes_root=fix_root, benchmark=benchmark)
    if not fix:
        return temp_dir
    if fix.agent_overlay:
        apply_agent_overlay(temp_dir, fix.agent_overlay)
    if fix.agent_patch:
        apply_agent_patch(temp_dir, fix.agent_patch)
    return temp_dir


def resolve_corebench_dataset_path(override: str | None) -> Path:
    if override:
        candidate = Path(override)
        if not candidate.is_absolute():
            candidate = (REPO_ROOT / candidate).resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"--corebench-dataset not found: {candidate}")
        return candidate

    if DEFAULT_COREBENCH_DATA.exists():
        return DEFAULT_COREBENCH_DATA

    root = REPO_ROOT / "hal-harness" / "hal" / "benchmarks"
    if root.exists():
        candidates = sorted(root.rglob("core_test.json"))
        if candidates:
            for path in candidates:
                if "corebench" in {part.lower() for part in path.parts}:
                    return path
            return candidates[0]

    raise FileNotFoundError(
        "CoreBench dataset file core_test.json not found. "
        "Ensure hal-harness is present (and any submodules/assets are initialized), "
        "or pass --corebench-dataset /path/to/core_test.json."
    )


def load_corebench_dataset(dataset_path: Path) -> List[dict]:
    return json.loads(dataset_path.read_text(encoding="utf-8"))


def build_single_task_payload(
    *,
    original_tasks: Dict[str, dict],
    capsule_id: str,
    input_override: Dict[str, object] | None,
) -> str:
    if capsule_id not in original_tasks:
        raise ValueError(f"Capsule {capsule_id} not found in core_test.json")
    task = json.loads(json.dumps(original_tasks[capsule_id]))  # deep copy
    if input_override:
        task.update(input_override)
        if "problem_statement" in input_override:
            original_prompt = task.get("task_prompt", "")
            extra = input_override["problem_statement"]
            task["task_prompt"] = f"{original_prompt}\n\n{extra}" if original_prompt else extra
    return json.dumps([task], indent=2)


def write_filtered_dataset(
    *,
    original_tasks: Dict[str, dict],
    capsule_id: str,
    input_override: Dict[str, object] | None,
    dataset_path: Path,
) -> None:
    payload = build_single_task_payload(
        original_tasks=original_tasks,
        capsule_id=capsule_id,
        input_override=input_override,
    )
    tmp_path = dataset_path.with_suffix(dataset_path.suffix + ".tmp")
    tmp_path.write_text(payload, encoding="utf-8")
    tmp_path.replace(dataset_path)


def recover_dataset_if_needed(dataset_path: Path, dataset_backup: Path) -> None:
    """Restore dataset_path from a leftover backup if a prior run was interrupted."""
    if not dataset_backup.exists():
        return
    print(
        f"[WARN] Found leftover dataset backup at {dataset_backup}. "
        f"Restoring {dataset_path} from backup before continuing."
    )
    dataset_path.unlink(missing_ok=True)
    shutil.move(dataset_backup, dataset_path)

def install_agent_requirements(agent_dir: Path) -> None:
    requirements = agent_dir / "requirements.txt"
    if not requirements.exists():
        raise FileNotFoundError(f"requirements.txt not found at {requirements}")
    print(f"[setup][pip] Checking requirements from {requirements}")
    req_fingerprint = hashlib.sha256(
        (sys.executable + "\n" + requirements.read_text(encoding="utf-8")).encode("utf-8")
    ).hexdigest()[:16]
    sentinel = REPO_ROOT / ".agent_requirements_installed" / f"{agent_dir.name}_{req_fingerprint}.ok"
    lock_path = REPO_ROOT / ".agent_requirements_install.lock"
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = lock_path.open("w", encoding="utf-8")
    try:
        print(f"[setup][pip] Acquiring pip install lock...")
        try:
            import fcntl  # type: ignore

            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            print(f"[setup][pip] Lock acquired")
        except Exception:
            # Best-effort: on platforms without fcntl, run without a lock.
            print(f"[setup][pip] Lock not available, proceeding without lock")

        if sentinel.exists():
            print(f"[setup][pip] Cache hit: requirements already installed (sentinel: {sentinel.name})")
            return

        print(f"[setup][pip] Cache miss: installing requirements (fingerprint: {req_fingerprint})")
        env = os.environ.copy()
        env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
        env.setdefault("PIP_NO_INPUT", "1")

        try:
            print(f"[setup][pip] Running: pip install -r {requirements}")
            start_time = time.time()
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(requirements)],
                check=True,
                env=env,
            )
            elapsed = time.time() - start_time
            print(f"[setup][pip] SUCCESS: Requirements installed in {elapsed:.1f}s")
        except subprocess.CalledProcessError as exc:
            # Common in parallel runs if the env is mid-mutation or already partially corrupted.
            print(f"[setup][pip] WARN: pip install failed, attempting minimal repair: {exc}")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--ignore-installed", "rich"],
                check=False,
                env=env,
            )
            print(f"[setup][pip] Retrying pip install after repair...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(requirements)],
                check=True,
                env=env,
            )
            print(f"[setup][pip] SUCCESS: Requirements installed after repair")
        print(f"[setup][pip] Creating sentinel: {sentinel.name}")
        sentinel.write_text("ok\n", encoding="utf-8")
    finally:
        try:
            lock_handle.close()
        except Exception:
            pass


def _slugify(value: str, fallback: str) -> str:
    # Keep readable identifiers for run_ids / filenames.
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    return slug or fallback


def build_task_run_id(benchmark: str, agent_args: Dict[str, object], capsule_id: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    api_model = str(agent_args.get("api_model_id") or agent_args.get("model_name") or "model")
    model_slug = _slugify(api_model.replace("/", "_"), "model")
    effort = str(agent_args.get("reasoning_effort") or "").strip().lower()
    effort_part = f"_{_slugify(effort, '')}" if effort else ""
    capsule_slug = _slugify(capsule_id, "task")
    bench_slug = _slugify(benchmark, "benchmark")
    return f"{model_slug}{effort_part}_{capsule_slug}_{bench_slug}_{timestamp}"


def merge_trace_files(
    trace_paths: List[Path],
    benchmark: str,
    agent_dir: Path,
    *,
    output_path: Path | None = None,
    merged_run_id: str | None = None,
) -> Path:
    first_trace = json.loads(trace_paths[0].read_text(encoding="utf-8"))
    config: Dict[str, Any] = first_trace.get("config", {})
    agent_name = config.get("agent_name") or agent_dir.name
    model_name = (
        config.get("agent_args", {}).get("model_name")
        if isinstance(config.get("agent_args"), dict)
        else None
    ) or "model"

    if output_path is None:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        agent_slug = _slugify(agent_dir.name, "agent")
        model_slug = _slugify(str(model_name), "model")
        run_id = merged_run_id or f"{benchmark}_{agent_slug}{model_slug}_{timestamp}_FIXED"
        output_path = REPO_ROOT / "traces" / f"{run_id}_UPLOAD.json"
    else:
        if output_path.suffix != ".json":
            raise ValueError(f"--merged-trace-output must be a .json file path, got {output_path}")
        run_id = merged_run_id or output_path.stem
        if run_id.endswith("_UPLOAD"):
            run_id = run_id[: -len("_UPLOAD")]

    merge_cmd: List[str] = [
        sys.executable,
        "scripts/merge_traces.py",
    ]
    for path in trace_paths:
        merge_cmd += ["--input", str(path)]
    merge_cmd += [
        "--output",
        str(output_path),
        "--run-id",
        run_id,
        "--agent-name",
        agent_name,
        "--date",
        datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "--force",
    ]
    subprocess.run(merge_cmd, check=True, cwd=REPO_ROOT)
    return output_path


def evaluate_trace(trace_path: Path, rubric_model: str, output_dir: Path) -> None:
    eval_cmd = [
        sys.executable,
        "main.py",
        "evaluate",
        "--trace-file",
        str(trace_path),
        "--rubrics-dir",
        str(REPO_ROOT / "rubrics"),
        "--output-dir",
        str(output_dir),
        "--rubric-model",
        rubric_model,
        "--yes",
    ]
    subprocess.run(eval_cmd, check=True, cwd=REPO_ROOT)


def run_one_capsule(
    *,
    capsule_id: str,
    fixes_base: Path,
    dataset_dir: Path,
    dataset_payload: str,
    agent_dir: Path,
    agent_args: Dict[str, object],
    benchmark: str,
    prefix: str,
    docker: bool,
    vm: bool,
    wandb_mode: str | None,
    env_override: Dict[str, str] | None,
    rubric_model: str,
    rubric_output_dir: Path,
    skip_rubrics: bool,
    keep_temp: bool,
) -> Path:
    dataset_path = dataset_dir / f"core_test_{capsule_id}_{uuid.uuid4().hex}.json"
    dataset_path.write_text(dataset_payload, encoding="utf-8")

    temp_agent_dir = copy_agent_with_fix(agent_dir, capsule_id, fixes_base, benchmark=benchmark)
    try:
        prefix_value = str(prefix or "").strip()
        prefix_part = f"{_slugify(prefix_value, '')}_" if prefix_value else ""
        run_id = prefix_part + build_task_run_id(benchmark, agent_args, capsule_id)
        print(f"[capsule][start] capsule_id={capsule_id} run_id={run_id} benchmark={benchmark}")
        cmd = build_hal_eval_cmd(
            benchmark=benchmark,
            agent_name=f"hal_generalist_agent ({capsule_id})",
            agent_dir=temp_agent_dir,
            agent_args=agent_args,
            run_id=run_id,
            docker=docker,
            vm=vm,
        )
        print(f"[hal-eval] {' '.join(cmd)}")
        hal_env = os.environ.copy()
        sanitize_tls_env(hal_env)
        if env_override:
            for key, value in env_override.items():
                if value is None:
                    continue
                hal_env[str(key)] = str(value)
        extra_path = str(REPO_ROOT / "hal-harness")
        hal_env["PYTHONPATH"] = (
            f"{extra_path}{os.pathsep}{hal_env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
        )
        hal_env["HAL_COREBENCH_DATASET_PATH"] = str(dataset_path)
        hal_env.setdefault("HAL_PRICING_MODEL_NAME", str(agent_args.get("model_name", "")))
        hal_env.setdefault("WANDB_SILENT", "true")
        if prefix_value:
            base_project = hal_env.get("WANDB_PROJECT") or "hal"
            prefix_slug = _slugify(prefix_value, "run")
            if not base_project.startswith(f"{prefix_slug}_"):
                hal_env["WANDB_PROJECT"] = f"{prefix_slug}_{base_project}"
            hal_env.setdefault("WANDB_RUN_GROUP", prefix_slug)
        if wandb_mode == "disabled":
            hal_env["WANDB_MODE"] = "disabled"
        elif wandb_mode:
            hal_env["WANDB_MODE"] = wandb_mode

        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=hal_env,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            output = (proc.stdout or "") + "\n" + (proc.stderr or "")
            output = output.strip()
            if output:
                print(f"[hal-eval][error] capsule_id={capsule_id} run_id={run_id}\n{output}\n")
            raise subprocess.CalledProcessError(proc.returncode, proc.args, output=proc.stdout, stderr=proc.stderr)

        results_dir = REPO_ROOT / "results" / benchmark / run_id
        trace_path = results_dir / f"{run_id}_UPLOAD.json"
        if not trace_path.exists():
            raise FileNotFoundError(f"Expected trace {trace_path} not found.")

        final_trace = REPO_ROOT / "traces" / trace_path.name
        shutil.copyfile(trace_path, final_trace)
        print(f"[trace] Copied {trace_path} -> {final_trace}")
        print(f"[capsule][done] capsule_id={capsule_id} run_id={run_id} trace={final_trace}")

        if not skip_rubrics:
            eval_cmd = [
                sys.executable,
                "main.py",
                "evaluate",
                "--trace-file", str(final_trace),
                "--rubrics-dir", str(REPO_ROOT / "rubrics"),
                "--output-dir", str(rubric_output_dir),
                "--rubric-model", rubric_model,
                "--yes",
            ]
            print(f"[rubric] {' '.join(eval_cmd)}")
            subprocess.run(eval_cmd, check=True, cwd=REPO_ROOT)
            print(f"[capsule][rubric_done] capsule_id={capsule_id} run_id={run_id}")

        return final_trace
    finally:
        dataset_path.unlink(missing_ok=True)
        if not keep_temp:
            shutil.rmtree(temp_agent_dir, ignore_errors=True)


def main() -> None:
    args = parse_args()
    if not args.skip_rubrics and not args.rubric_model:
        raise SystemExit("--rubric-model is required unless --skip-rubrics is set.")
    fixes_root = Path(args.fixes_root)
    agent_dir = Path(args.agent_dir)
    agent_args_path = Path(args.agent_args)
    dataset_source = resolve_corebench_dataset_path(args.corebench_dataset)
    rubric_output_dir = REPO_ROOT / "rubrics_output"
    rubric_output_dir.mkdir(parents=True, exist_ok=True)

    agent_args = load_agent_args(agent_args_path)
    agent_args["benchmark_name"] = args.benchmark

    if not args.skip_install:
        install_agent_requirements(agent_dir)

    temp_root = Path(tempfile.mkdtemp(prefix="corebench_dataset_"))
    tasks_data = load_corebench_dataset(dataset_source)
    capsule_map = {item["capsule_id"]: item for item in tasks_data}

    fix_dirs = sorted([p for p in fixes_root.iterdir() if p.is_dir()])
    if args.task_ids:
        requested = set(args.task_ids)
        fix_dirs = [p for p in fix_dirs if p.name in requested]
        missing = requested - {p.name for p in fix_dirs}
        if missing:
            print(f"[WARN] Requested task(s) not found under {fixes_root}: {', '.join(sorted(missing))}")
    if not fix_dirs:
        print(f"No fix directories found in {fixes_root}")
        return

    try:
        fixes_base = fixes_root.parent
        capsule_jobs: List[tuple[str, str, Dict[str, str] | None]] = []
        conda_env = "hal"
        conda_channels: List[str] = []
        conda_packages: List[str] = []
        for fix_dir in fix_dirs:
            capsule_id = fix_dir.name
            print(f"\n=== Queuing {capsule_id} ===")
            if capsule_id not in capsule_map:
                print(
                    f"[WARN] Capsule {capsule_id} not found in dataset {dataset_source}. "
                    "Skipping this fix folder."
                )
                continue
            try:
                fix = load_fix_package(capsule_id, fixes_root=fixes_base, benchmark=args.benchmark)
            except Exception as exc:
                print(f"[WARN] Failed to load fix for {capsule_id}: {exc}. Skipping.")
                continue
            input_override = fix.input_override if fix else None
            env_override = fix.env_override if fix else None
            if env_override:
                conda_env = str(env_override.get("HAL_FORCE_CONDA_ENV") or conda_env)
                conda_channels.extend(_split_channels(env_override.get("HAL_CONDA_CHANNELS")))
                conda_packages.extend(_split_packages(env_override.get("HAL_CONDA_PACKAGES")))
            payload = build_single_task_payload(
                original_tasks=capsule_map,
                capsule_id=capsule_id,
                input_override=input_override,
            )
            capsule_jobs.append((capsule_id, payload, env_override))

        if not capsule_jobs:
            print("No runnable capsules found after filtering.")
            return

        # Best-effort: ensure conda packages requested by fix packages are installed once up-front.
        conda_packages = sorted({p for p in conda_packages if p})
        conda_channels = sorted({c for c in conda_channels if c}) or ["conda-forge"]
        ensure_hal_cli_dependencies(conda_env, conda_channels)
        if conda_packages:
            _install_conda_packages(env_name=conda_env, channels=conda_channels, packages=conda_packages)

        generated_traces: List[Path] = []
        max_workers = max(1, int(args.max_parallel_capsules))
        if max_workers <= 1 or len(capsule_jobs) <= 1:
            for capsule_id, payload, env_override in capsule_jobs:
                print(f"\n=== Processing {capsule_id} ===")
                generated_traces.append(
                    run_one_capsule(
                        capsule_id=capsule_id,
                        fixes_base=fixes_base,
                        dataset_dir=temp_root,
                        dataset_payload=payload,
                        agent_dir=agent_dir,
                        agent_args=agent_args,
                        benchmark=args.benchmark,
                        prefix=args.prefix,
                        docker=args.docker,
                        vm=args.vm,
                        wandb_mode=args.wandb_mode,
                        env_override=env_override,
                        rubric_model=args.rubric_model,
                        rubric_output_dir=rubric_output_dir,
                        skip_rubrics=args.skip_rubrics,
                        keep_temp=args.keep_temp,
                    )
                )
        else:
            print(f"\n=== Running {len(capsule_jobs)} capsules with max_parallel_capsules={max_workers} ===")
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(
                        run_one_capsule,
                        capsule_id=capsule_id,
                        fixes_base=fixes_base,
                        dataset_dir=temp_root,
                        dataset_payload=payload,
                        agent_dir=agent_dir,
                        agent_args=agent_args,
                        benchmark=args.benchmark,
                        prefix=args.prefix,
                        docker=args.docker,
                        vm=args.vm,
                        wandb_mode=args.wandb_mode,
                        env_override=env_override,
                        rubric_model=args.rubric_model,
                        rubric_output_dir=rubric_output_dir,
                        skip_rubrics=args.skip_rubrics,
                        keep_temp=args.keep_temp,
                    ): capsule_id
                    for capsule_id, payload, env_override in capsule_jobs
                }
                for future in as_completed(futures):
                    try:
                        generated_traces.append(future.result())
                    except Exception:
                        for pending in futures:
                            pending.cancel()
                        raise

        if generated_traces:
            generated_traces = sorted(generated_traces, key=lambda p: p.name)
            merged_trace = merge_trace_files(
                generated_traces,
                benchmark=args.benchmark,
                agent_dir=agent_dir,
                output_path=Path(args.merged_trace_output).resolve() if args.merged_trace_output else None,
                merged_run_id=args.merged_run_id,
            )
            print(f"[merge] Saved merged trace -> {merged_trace}")
            if not args.skip_rubrics:
                evaluate_trace(merged_trace, args.rubric_model, rubric_output_dir)
                print(f"[merge] Completed rubric evaluation for {merged_trace}")

    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
