"""Run a local CPU benchmark in an isolated Python process."""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import statistics
import time
from importlib import metadata
from pathlib import Path

from .model import DecisionModel, sha256


def _peak_rss_mib() -> float:
    """Read the process high-water mark on Linux; fall back to ru_maxrss."""
    status = Path("/proc/self/status")
    if status.is_file():
        for line in status.read_text(encoding="ascii").splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _cpu_model() -> str | None:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    return None


def _installed_dependency_sizes(artifact_kind: str) -> dict[str, int | None]:
    packages = ["cryptography"]
    if artifact_kind == "encoder_onnx":
        packages.extend(["numpy", "onnx", "onnxruntime", "tokenizers"])
    sizes = {}
    for package in packages:
        try:
            distribution = metadata.distribution(package)
            files = distribution.files or []
            sizes[package] = sum(path.stat().st_size for item in files
                                 if (path := distribution.locate_file(item)).is_file() and not path.suffix == ".pyc")
        except (metadata.PackageNotFoundError, OSError):
            sizes[package] = None
    return sizes


def benchmark(bundle: Path, texts: list[str], *, warmup: int = 100, samples: int = 1000,
              trusted_public_key: bytes | None = None) -> dict:
    if not texts or warmup < 0 or samples < 1:
        raise ValueError("benchmark needs representative texts and positive sample count")
    started = time.perf_counter()
    model = DecisionModel.load(bundle, allow_unsigned=trusted_public_key is None,
                               trusted_public_key=trusted_public_key)
    cold_load_ms = (time.perf_counter() - started) * 1000
    for index in range(warmup):
        model.predict(texts[index % len(texts)], evaluation=True)
    durations = []
    for index in range(samples):
        started = time.perf_counter()
        model.predict(texts[index % len(texts)], evaluation=True)
        durations.append((time.perf_counter() - started) * 1000)
    ordered = sorted(durations)
    peak_rss_mib = _peak_rss_mib()
    return {
        "machine": platform.platform(),
        "architecture": platform.machine(),
        "os_release": platform.release(),
        "processor": platform.processor(),
        "cpu_model": _cpu_model(),
        "logical_cpus": os.cpu_count(),
        "affinity_cpus": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "artifact_kind": model.manifest["artifact_kind"],
        "manifest_sha256": sha256((bundle / "manifest.json").read_bytes()),
        "bundle_bytes": sum(path.stat().st_size for path in bundle.rglob("*") if path.is_file()),
        "installed_inference_dependency_bytes": _installed_dependency_sizes(model.manifest["artifact_kind"]),
        "python": platform.python_version(),
        "cold_load_ms": cold_load_ms,
        "warm_p50_ms": statistics.median(durations),
        "warm_p95_ms": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "peak_rss_mib": peak_rss_mib,
        "peak_rss_method": "linux_proc_vmhwm" if Path("/proc/self/status").is_file() else "rusage_maxrss",
        "warmup_calls": warmup,
        "timed_calls": samples,
        "concurrency": 1,
        "raw_timing_ms": durations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a local decision bundle")
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--texts", type=Path, required=True, help="JSON array of representative texts")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trusted-public-key", type=Path, help="external PEM public key for a signed bundle")
    args = parser.parse_args()
    texts = json.loads(args.texts.read_text(encoding="utf-8"))
    result = benchmark(args.bundle, texts,
                       trusted_public_key=args.trusted_public_key.read_bytes() if args.trusted_public_key else None)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
