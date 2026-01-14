#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set

REPO_ROOT = Path(__file__).resolve().parents[1]


def _conda_executable() -> str | None:
    return shutil.which("conda") or shutil.which("mamba") or shutil.which("micromamba") or None


def _conda_spec_name(spec: str) -> str:
    for sep in ("==", ">=", "<=", "=", ">", "<"):
        if sep in spec:
            return spec.split(sep, 1)[0].strip()
    return spec.strip()


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
    return [p for p in shlex.split(raw) if p]


def _load_json(path: Path) -> Dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def discover_fix_envs(fixes_root: Path, benchmark: str) -> List[Dict[str, object]]:
    benchmark_root = fixes_root / benchmark
    envs: List[Dict[str, object]] = []
    for capsule_dir in sorted(benchmark_root.glob("capsule-*")):
        if not capsule_dir.is_dir():
            continue
        env_override = _load_json(capsule_dir / "env_override.json") or {}
        envs.append({"capsule": capsule_dir.name, "env_override": env_override})
    return envs


def conda_search_available(conda: str, *, channels: List[str], names: List[str]) -> Set[str] | None:
    names = sorted({n for n in names if n})
    if not names:
        return set()
    cmd = [conda, "search", "--json"]
    for channel in channels:
        cmd += ["-c", channel]
    cmd += names
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        return None
    try:
        payload = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        return None
    available: Set[str] = set()
    for name in names:
        entries = payload.get(name)
        if isinstance(entries, list) and entries:
            available.add(name)
    return available


def main() -> None:
    parser = argparse.ArgumentParser(description="Check conda package availability for fix env overrides.")
    parser.add_argument("--fixes-root", default="fixes", help="Fixes root directory (default: fixes).")
    parser.add_argument("--benchmark", default="corebench_hard", help="Benchmark folder under fixes/ (default: corebench_hard).")
    args = parser.parse_args()

    conda = _conda_executable()
    if not conda:
        raise SystemExit("Could not find conda/mamba/micromamba on PATH.")

    fixes_root = (REPO_ROOT / args.fixes_root).resolve()
    envs = discover_fix_envs(fixes_root, args.benchmark)

    channels: Set[str] = set()
    requested_specs: Dict[str, Set[str]] = {}
    requested_names: Set[str] = set()

    for item in envs:
        capsule = str(item["capsule"])
        env_override = item.get("env_override") or {}
        if not isinstance(env_override, dict):
            continue
        channels.update(_split_channels(env_override.get("HAL_CONDA_CHANNELS")))
        specs = _split_packages(env_override.get("HAL_CONDA_PACKAGES"))
        if specs:
            requested_specs[capsule] = set(specs)
        for spec in specs:
            requested_names.add(_conda_spec_name(spec))

    channels_list = sorted(channels) or ["conda-forge"]
    available = conda_search_available(conda, channels=channels_list, names=sorted(requested_names))
    if available is None:
        print(f"[warn] conda search failed; cannot determine availability (channels={channels_list})")
        sys.exit(2)

    missing = sorted(requested_names - available)
    print(f"Conda executable: {conda}")
    print(f"Channels: {channels_list}")
    print(f"Unique requested package names: {len(requested_names)}")
    print(f"Missing package names: {len(missing)}")
    for name in missing:
        print(f"- {name}")

    # Exit non-zero if anything is missing (useful for CI / preflight).
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()

