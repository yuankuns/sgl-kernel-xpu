import argparse
import hashlib
import inspect
import json
import math
import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
COMPONENTS = ("memory", "position", "logits-vocab")


def digest(path):
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def scope():
    for name in ("BATCHES", "BATCH", "SERVING_BATCH", "BATCH_SIZE", "SGLANG_BATCH_SIZE", "SGLANG_MUSE_BATCH_SIZE", "MUSE_BATCH_SIZE", "SERVING_BATCH_SIZE", "MUSE_BS", "BS"):
        if os.environ.get(name, "8") != "8":
            raise SystemExit(f"B8-only: {name}")
    for name, expected in {"ISL": "4096", "OSL": "16", "TP": "1", "EP": "1"}.items():
        if os.environ.get(name, expected) != expected:
            raise SystemExit(f"Wrong production scope: {name}")


def provider(component, role, gpu):
    scope()
    evidence = ROOT / component / "source-compiler-accuracy-main-v1"
    prepared = read(evidence / "prepared.json")
    stage = evidence / role / "site"
    import sgl_kernel
    import torch

    from benchmark_simple import native_operations

    assert Path(sgl_kernel.__file__).resolve() == stage / "sgl_kernel/__init__.py"
    assert Path(torch.__file__).resolve().is_relative_to(Path("/opt/gfx-deps/venv"))
    mapped = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines() if "/common_ops" in line or "libsgl-ops-sycl-" in line})
    actual = {}
    for filename in mapped:
        path = Path(filename).resolve()
        relative = path.relative_to(stage).as_posix()
        assert digest(path) == prepared["roles"][role]["files"][relative]
        actual[relative] = digest(path)
    for name in prepared["tested_native"]:
        assert f"sgl_kernel/{name}" in actual
    operations = native_operations(torch, argparse.Namespace(component=component, api="source-native"))
    schemas = {name: str(operation.default._schema) for name, operation in operations.items()}
    if gpu:
        assert os.environ.get("ZE_AFFINITY_MASK") == "11"
        assert torch.xpu.is_available()
        assert "674f" in torch.xpu.get_device_name(0).lower()
    else:
        assert not torch.xpu.is_initialized()
    return torch, {"component": component, "role": role, "package": sgl_kernel.__file__, "torch": torch.__file__, "mapped_native": actual, "schemas": schemas, "gpu_checked": gpu, "normal_original_registration": True, "AFTER": False}


def collect_checked_tensors(torch, function, result, prefix="check", seen=None):
    seen = set() if seen is None else seen
    if id(function) in seen:
        return
    seen.add(id(function))
    if torch.is_tensor(function):
        if function.device.type == "xpu":
            result[prefix] = function.detach().cpu()
    elif inspect.isfunction(function):
        for name, cell in zip(function.__code__.co_freevars, function.__closure__ or ()):
            collect_checked_tensors(torch, cell.cell_contents, result, prefix + "/" + name, seen)
    elif isinstance(function, (list, tuple)):
        for index, value in enumerate(function):
            collect_checked_tensors(torch, value, result, prefix + f"/{index}", seen)


def worker(component, role):
    torch, report = provider(component, role, True)
    from benchmark_simple import logits_cases, memory_cases, native_operations
    from benchmark_simple import position_cases

    torch.manual_seed(42)
    arguments = argparse.Namespace(component=component, api="source-native")
    operations = native_operations(torch, arguments)
    cases = {"memory": memory_cases, "position": position_cases, "logits-vocab": logits_cases}[component]
    checked_outputs = {}
    case_results = []
    for index, (name, launch, check, reset) in enumerate(cases(torch, operations)):
        if index:
            torch.xpu.synchronize()
            time.sleep(2)
        if reset:
            reset()
        launch()
        torch.xpu.synchronize()
        check()
        captured = {}
        collect_checked_tensors(torch, check, captured)
        assert captured, name
        checked_outputs[name] = captured
        case_results.append({"case": name, "independent_original_full_reference": "PASS", "captured_full_output_tensors": {key: {"shape": list(value.shape), "dtype": str(value.dtype), "elements": value.numel()} for key, value in captured.items()}})
        print(json.dumps(case_results[-1]), flush=True)
    evidence = ROOT / component / "source-compiler-accuracy-main-v1"
    torch.save(checked_outputs, evidence / role / "checked-full-outputs.pt")
    report.update({"cases": case_results, "precision": "original references; softcap FP32 atol=rtol3e-5; masks/embedding/integers exact", "timing": False, "events": False, "graphs": False, "warmup": False})
    (evidence / role / "production-reference.json").write_text(json.dumps(report, indent=2) + "\n")


def execute(command, environment, prefix):
    prefix.with_suffix(".command").write_text(subprocess.list2cmdline(command) + "\n")
    with prefix.with_suffix(".log").open("x") as output:
        process = subprocess.run(command, env=environment, stdout=output, stderr=subprocess.STDOUT)
    prefix.with_suffix(".exit").write_text(str(process.returncode) + "\n")
    assert process.returncode == 0, (str(prefix), process.returncode)


def analyze(component):
    scope()
    import torch

    assert not torch.xpu.is_initialized()
    evidence = ROOT / component / "source-compiler-accuracy-main-v1"
    historical = torch.load(evidence / "historical/checked-full-outputs.pt", map_location="cpu", weights_only=True)
    fresh = torch.load(evidence / "fresh-icpx/checked-full-outputs.pt", map_location="cpu", weights_only=True)
    assert historical.keys() == fresh.keys()
    results = {}
    for case, values in historical.items():
        assert values.keys() == fresh[case].keys()
        comparisons = {}
        for name, expected in values.items():
            actual = fresh[case][name]
            assert expected.shape == actual.shape and expected.dtype == actual.dtype
            differences = int((expected.contiguous().view(torch.uint8) != actual.contiguous().view(torch.uint8)).sum().item())
            comparisons[name] = {"elements": expected.numel(), "dtype": str(expected.dtype), "byte_differences": differences, "entire_tensor_bitwise": differences == 0}
        results[case] = comparisons
    all_equal = all(result["entire_tensor_bitwise"] for comparisons in results.values() for result in comparisons.values())
    (evidence / "full-output-equivalence.json").write_text(json.dumps({"component": component, "cases": results, "all_full_outputs_bitwise": all_equal, "comparison": "actual entire CPU tensors as uint8, not hash-only", "AFTER": False, "performance": False}, indent=2) + "\n")
    assert all_equal, component
    print(json.dumps({"component": component, "all_full_outputs_bitwise": True, "case_count": len(results)}))


def supervisor(grant_path):
    scope()
    assert os.environ.get("ZE_AFFINITY_MASK") == "11"
    assert os.sched_getaffinity(0) == set(range(112, 128))
    grant = read(grant_path)
    assert grant["issuer"] == "main" and grant["scope"] == "simple-SOURCE-only-compiler-accuracy-v1"
    assert grant["gpu"] == 11 and grant["performance_authorized"] is False
    assert grant["accuracy_authorized"] is True
    assert grant["cpu_affinity"] == list(range(112, 128))
    for name, expected in {"serving_batch": 8, "isl": 4096, "osl": 16, "tp": 1, "ep": 1}.items():
        assert type(grant[name]) is int and grant[name] == expected
    assert math.isfinite(grant["start_epoch"]) and grant["start_epoch"] > 0
    assert grant["components"] == list(COMPONENTS)
    assert grant["driver_sha256"] == digest(Path(__file__))
    for component in COMPONENTS:
        evidence = ROOT / component / "source-compiler-accuracy-main-v1"
        assert digest(evidence / "prepared.json") == grant["prepared_sha256"][component]
        prepared = read(evidence / "prepared.json")
        assert digest(ROOT / "benchmark_simple.py") == prepared["original_references_driver_sha256"]
        assert digest(Path(prepared["fresh_source_manifest"])) == prepared["fresh_source_manifest_sha256"]
        for name, sha256 in prepared["original_tests_sha256"].items():
            assert digest(ROOT / component / "native-diagnostic/tests" / name) == sha256
        for role in ("historical", "fresh-icpx"):
            assert (evidence / role / "CPU-normal-provider-final.exit").read_text().strip() == "0"
            for relative, expected in prepared["roles"][role]["files"].items():
                assert digest(evidence / role / "site" / relative) == expected
    time.sleep(max(0, grant["start_epoch"] - time.time()))
    for component in COMPONENTS:
        evidence = ROOT / component / "source-compiler-accuracy-main-v1"
        tests = {"memory": ("test_paged_alloc.py", "test_write_req_to_token_pool.py"), "position": ("test_compute_position.py",), "logits-vocab": ("test_softcap_logits.py", "test_vocab_mask.py")}[component]
        for role in ("historical", "fresh-icpx"):
            environment = dict(os.environ, PYTHONPATH=f'{evidence / role / "site"}:{ROOT / "logits-vocab/framework-original/site"}:{ROOT}', PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8", LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:" + os.environ.get("LD_LIBRARY_PATH", ""))
            execute(["/opt/gfx-deps/venv/bin/python", "-B", str(Path(__file__)), "--mode", "worker", "--component", component, "--role", role], environment, evidence / role / "production-reference")
            execute(["/opt/gfx-deps/venv/bin/python", "-m", "pytest", "-p", "acceptance_cooldown", "-p", "no:cacheprovider", "-q", *(str(ROOT / component / "native-diagnostic/tests" / name) for name in tests)], environment, evidence / role / "original-tests")
            time.sleep(2)
        execute(["/opt/gfx-deps/venv/bin/python", "-B", str(Path(__file__)), "--mode", "analyze", "--component", component], environment, evidence / "full-output-analysis")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("cpu-provider", "worker", "supervisor", "analyze"), required=True)
    parser.add_argument("--component", choices=COMPONENTS)
    parser.add_argument("--role", choices=("historical", "fresh-icpx"))
    parser.add_argument("--grant", type=Path)
    args = parser.parse_args()
    if args.mode == "supervisor":
        assert args.grant is not None
        supervisor(args.grant)
    elif args.mode == "worker":
        worker(args.component, args.role)
    elif args.mode == "analyze":
        analyze(args.component)
    else:
        _, report = provider(args.component, args.role, False)
        print(json.dumps(report))


if __name__ == "__main__":
    main()
