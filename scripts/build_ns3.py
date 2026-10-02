#!/usr/bin/env python3
"""Prepare and build the pinned ns-3.35 source used by the main experiment.

Run with Python 3.10. Build products and extracted upstream sources stay in
work/, outside the versioned source snapshots. This command never runs ns-3.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "code/vendor/ns-allinone-3.35.tar.bz2"
ARCHIVE_SHA256 = "25e07a95349847b3e453d3af29a94545a4f869b1c6b4d860900cb7718fb1a618"
PREFIX = "ns-allinone-3.35/ns-3.35/"
MODULES = "core,network,mobility,config-store,wifi,internet,applications"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: list[str], cwd: Path, env: dict[str, str]) -> None:
    print("Executing: " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def find_compiler(name: str, alternatives: list[str]) -> str:
    if os.environ.get(name):
        return os.environ[name]
    for candidate in alternatives:
        found = shutil.which(candidate)
        if found:
            return found
        sibling = Path(sys.executable).resolve().parent / candidate
        if sibling.is_file():
            return str(sibling)
    raise SystemExit(f"No {name} compiler found. Install build-essential or set {name}.")


def prepare(ns3: Path) -> None:
    if digest(ARCHIVE) != ARCHIVE_SHA256:
        raise SystemExit("The ns-3.35 release archive does not match its pinned SHA256.")
    if not (ns3 / "waf").is_file():
        ns3.mkdir(parents=True, exist_ok=True)
        # Only extract the ns-3 component; network visualizers and other bundled
        # tools are unnecessary. Avoid tar extractall and reject path traversal.
        with tarfile.open(ARCHIVE, "r:bz2") as archive:
            for member in archive:
                if not member.name.startswith(PREFIX) or not member.isfile():
                    continue
                relative = Path(member.name[len(PREFIX):])
                if relative.is_absolute() or ".." in relative.parts:
                    raise SystemExit("Unsafe archive member: " + member.name)
                destination = ns3 / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                stream = archive.extractfile(member)
                assert stream is not None
                destination.write_bytes(stream.read())
                destination.chmod(member.mode)
        stamp = ns3 / ".main-results-patch-applied"
        patch = ROOT / "code/patches/ns3-35-block-ack-stability.patch"
        subprocess.run(["patch", "-p1", "--batch", "-i", str(patch)],
                       cwd=ns3, check=True)
        stamp.write_text(digest(patch) + "\n")
    else:
        expected = digest(ROOT / "code/patches/ns3-35-block-ack-stability.patch")
        stamp = ns3 / ".main-results-patch-applied"
        if not stamp.exists() or stamp.read_text().strip() != expected:
            raise SystemExit("Existing work tree has no matching patch stamp. "
                             "Use a fresh --ns3-dir to prepare pinned sources.")

    source = ROOT / "code/nsTest"
    destination = ns3 / "scratch/nsTest"
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.rglob("*"):
        if path.is_file():
            target = destination / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    (destination / "data").mkdir(exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--ns3-dir", type=Path,
                        default=ROOT / "work/ns-allinone-3.35/ns-3.35")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--configure-only", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if sys.version_info[:2] != (3, 10):
        raise SystemExit("Use Python 3.10 for the bundled Waf 2.0.21. "
                         "Python 3.11+ removed file modes used by this upstream Waf.")
    ns3 = args.ns3_dir.expanduser().resolve()
    prepare(ns3)
    if args.prepare_only:
        print(f"Pinned source prepared: {ns3}")
        return 0
    env = os.environ.copy()
    env["CC"] = find_compiler("CC", ["gcc", "x86_64-conda-linux-gnu-gcc", "clang"])
    env["CXX"] = find_compiler("CXX", ["g++", "x86_64-conda-linux-gnu-g++", "clang++"])
    env["AR"] = find_compiler("AR", ["ar", "x86_64-conda-linux-gnu-ar", "llvm-ar"])
    env["PATH"] = str(Path(env["CC"]).parent) + os.pathsep + env.get("PATH", "")
    # An optimized release build with the same C++17 language mode as the
    # archived runs. Supplying flags keeps the build portable across CPUs.
    env.setdefault("CXXFLAGS", "-O3 -g -Wall -std=c++17")
    run([sys.executable, str(ns3 / "waf"), "configure", "--build-profile=optimized",
         "--disable-python", "--disable-tests", "--disable-examples",
         f"--enable-modules={MODULES}"], ns3, env)
    if not args.configure_only:
        # Build the enabled module set as a whole: Waf's --targets filter
        # omits generated public-header tasks on a completely fresh tree.
        run([sys.executable, str(ns3 / "waf"), "build",
             "-j", str(args.jobs)], ns3, env)
    report = {"archive_sha256": ARCHIVE_SHA256, "ns3_version": "3.35",
              "python": sys.version, "CC": env["CC"], "CXX": env["CXX"],
              "CXXFLAGS": env["CXXFLAGS"], "modules": MODULES.split(","),
              "compiled": not args.configure_only,
              "binary": str(ns3 / "build/scratch/nsTest/nsTest"),
              "simulation_run": False}
    (ns3 / "main_results_build.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
