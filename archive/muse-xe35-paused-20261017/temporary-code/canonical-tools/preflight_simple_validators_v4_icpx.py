import ast
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def digest(path):
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def main():
    names = [
        "canonical_tools_loader_v1.py",
        "prepare_simple_canonical_comparator_v1.py",
        "check_simple_canonical_package_v1.py",
        "benchmark_simple_canonical_v1.py",
        "validate_simple_canonical_v1.py",
        "analyze_simple_canonical_v1.py",
    ]
    files = {}
    for name in names:
        path = ROOT / name
        ast.parse(path.read_text(), filename=name)
        files[name] = {"sha256": digest(path), "ast_pass": True}
    validator = [sys.executable, "-S", str(ROOT / "validate_simple_canonical_v1.py"), "position"]
    missing_grant = ROOT / "nonexistent-cpu-preflight-grant-v4.json"
    assert not missing_grant.exists()
    accuracy = validator + ["--phase", "accuracy", "--grant", str(missing_grant)]
    performance = validator + ["--phase", "performance", "--grant", str(missing_grant)]
    tests = [
        (validator + ["--phase", "accuracy"], {}, "--grant"),
        (accuracy, {"ZE_AFFINITY_MASK": "0"}, "Wrong allocated GPU"),
        (performance, {"ZE_AFFINITY_MASK": "8"}, "Wrong allocated GPU"),
        (accuracy + ["--build-version", "canonical-v2"], {}, "invalid choice"),
        ([sys.executable, "-S", str(ROOT / "prepare_simple_canonical_comparator_v1.py"), "position"], {}, "--source-native-manifest"),
    ]
    aliases = ["BATCHES", "BATCH", "SERVING_BATCH", "BATCH_SIZE", "SGLANG_BATCH_SIZE", "SGLANG_MUSE_BATCH_SIZE", "MUSE_BATCH_SIZE", "SERVING_BATCH_SIZE", "MUSE_BS", "BS"]
    for name in aliases:
        tests.append((accuracy, {"ZE_AFFINITY_MASK": "11", name: "16"}, "B8-only"))
    for name, value in {"ISL": "2048", "OSL": "32", "TP": "2", "EP": "2"}.items():
        tests.append((accuracy, {"ZE_AFFINITY_MASK": "11", name: value}, "Production scope requires"))
    results = []
    for command, overrides, expected in tests:
        environment = dict(os.environ)
        for name in aliases + ["ISL", "OSL", "TP", "EP", "ZE_AFFINITY_MASK"]:
            environment.pop(name, None)
        environment.update(overrides)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        process = subprocess.run(command, env=environment, capture_output=True, text=True)
        assert process.returncode != 0 and expected in process.stderr, (command, process.returncode, process.stderr)
        assert "No module named 'torch'" not in process.stderr
        results.append({"command": command, "environment_overrides": overrides, "actual_exit": process.returncode, "stderr": process.stderr, "before_torch_import": True})
    report = {"version": "simple-canonical-validator-cpu-preflight-v4-icpx", "epoch": time.time(), "files": files, "negative_entry_guards": results, "GPU_started": False, "fresh_source_manifest_and_coordinator_grant_required": True, "verified": True}
    destination = ROOT / "simple-canonical-validator-cpu-preflight-v4-icpx.json"
    assert not destination.exists()
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"verified": True, "guard_cases": len(results), "files": len(files), "report": str(destination)}))


if __name__ == "__main__":
    main()
