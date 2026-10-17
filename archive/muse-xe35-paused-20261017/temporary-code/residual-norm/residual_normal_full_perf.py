import argparse
import ast
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import after_pre_wave_accuracy_v4 as after
import history24_accuracy as history
import single_case_forensic_v2 as helpers
import source_compiler_accuracy as source


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "history24-accuracy-v1"
OUTPUT = ROOT / "normal-full-perf-gpu4-v1"
BARRIER = ROOT / "gpu4-normal-full-perf-v1-grant.json"
PHASE = "residual-normal-full-gpu4-performance-v1"


def write(name, record):
    path = OUTPUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


def timer():
    tree = ast.parse(after.BENCH.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_case_xe35")
    block = next(node for node in function.body if isinstance(node, ast.If) and ast.unparse(node.test) == "not args.accuracy_only")
    arguments = ast.arguments(posonlyargs=[], args=[ast.arg(arg=name) for name in ("torch", "invoke_xe35", "args")], kwonlyargs=[], kw_defaults=[], defaults=[])
    body = [ast.parse("result = {}").body[0], *block.body, ast.parse("return result").body[0]]
    measured = ast.FunctionDef(name="measure_graph", args=arguments, body=body, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[measured], type_ignores=[]))
    namespace = dict(time=time, statistics=statistics)
    exec(compile(module, str(after.BENCH), "exec"), namespace)
    return namespace["measure_graph"], hashlib.sha256(ast.dump(block, include_attributes=False).encode()).hexdigest()


def prerequisites():
    source.scope()
    assert set(os.sched_getaffinity(0)) <= set(range(48, 64))
    audit = json.loads((ROOT / "NORMAL_PRECISION_PYTEST_FINAL_AUDIT_v2.json").read_text())
    independent = json.loads((ROOT / "RESIDUAL_NORMAL_PRECISION_INDEPENDENT_AUDIT_v2.json").read_text())
    assert audit["terminal_actual_exit"] == 0 and audit["GPU9_released"]
    assert independent["independent_audit_actual_exit"] == 0
    assert (ROOT / "precision-terminal-independent-v2.exit").read_text().strip() == "0"
    assert (DATA / "full-comparison.exit").read_text().strip() == "0"
    assert json.loads((DATA / "final-audit.json").read_text())["bitwise_paired_elements"] == 8055029760
    _, _, binding = after.provenance()
    binding.update(precision69_73_audit_SHA256=source.digest(ROOT / "NORMAL_PRECISION_PYTEST_FINAL_AUDIT_v2.json"),
        independent_audit_SHA256=source.digest(ROOT / "RESIDUAL_NORMAL_PRECISION_INDEPENDENT_AUDIT_v2.json"),
        history24_fulltensor_manifest_SHA256=source.digest(DATA / "fulltensor-evidence.sha256"),
        history24_comparison_SHA256=source.digest(DATA / "full-comparison.json"),
        timer_AST_SHA256=timer()[1], original_benchmark_SHA256=source.digest(after.BENCH),
        runner_SHA256=source.digest(Path(__file__)), wrapper_SHA256=source.digest(ROOT / "residual_normal_full_perf_container.sh"),
        supervisor_SHA256=source.digest(ROOT / "residual_normal_full_perf_stage.py"),
        driver_proof=after.driver_proof(), plan_SHA256=source.digest(ROOT / "NORMAL_FULL_PERF_GPU4_PLAN_v1.md"))
    return binding


def dataset():
    records = {}
    comparison = json.loads((DATA / "full-comparison.json").read_text())
    cases = history.cases()
    assert len(cases) == 24 and cases[17]["case"] == "32768_weight_0_1"
    for case in cases:
        key = case["case"]
        assert all(value["mismatch"] == 0 for value in comparison[key].values())
        stems = (f"fixtures/{key}-inputs", f"fixtures/{key}-oracle", f"SOURCE/{key}-actual", f"AFTER/{key}-actual")
        entries = {}
        for stem in stems:
            proof = json.loads((DATA / (stem + ".json")).read_text())
            assert source.digest(DATA / (stem + ".pt")) == proof["file_sha256"]
            entries[stem] = dict(file_SHA256=proof["file_sha256"], metadata_SHA256=source.digest(DATA / (stem + ".json")), tensors=proof["tensors"])
        records[key] = dict(case=case, full_input_oracle_SOURCE_AFTER=entries)
    return records


def load(torch, stem):
    record = json.loads((DATA / (stem + ".json")).read_text())
    assert source.digest(DATA / (stem + ".pt")) == record["file_sha256"]
    tensors = torch.load(DATA / (stem + ".pt"), mmap=True, map_location="cpu", weights_only=True)
    assert tensors.keys() == record["tensors"].keys()
    for name, tensor in tensors.items():
        assert helpers.tensor_digest(torch, tensor) == record["tensors"][name]["sha256"]
    return tensors


def cpu():
    assert os.environ["ONEAPI_DEVICE_SELECTOR"] == "level_zero:cpu"
    if not OUTPUT.exists():
        OUTPUT.mkdir()
    binding = prerequisites()
    records = dataset()
    stat = os.statvfs(OUTPUT)
    write("CPU-dataset.json", dict(binding=binding, records=records, cases=24, previous_NaN_case18_included=True,
        CPU_only=True, no_GPU_or_compilation=True, available_disk_bytes=stat.f_bavail * stat.f_frsize,
        planned_success_snapshots_GiB=576, artifact_cap_GiB=640, disk_headroom_GiB=100,
        worker_cap_GiB=96, descendant_tree_cap_GiB=128, performance_authorized=False,
        acceptance_scope="original operator contract; not actual framework/model path"))


def provider_cpu(role):
    assert os.environ["ONEAPI_DEVICE_SELECTOR"] == "level_zero:cpu"
    assert (OUTPUT / "CPU-dataset.exit").read_text().strip() == "0"
    torch, _, proof = helpers.provider(role)
    assert not torch.xpu.is_initialized()
    write(f"CPU-{role}.json", dict(binding=prerequisites(), maps=proof, GPU_initialized=False))


def grant():
    released = json.loads(BARRIER.read_text())
    assert released["phase"] == PHASE and released["issuer"] == "main coordinator"
    assert released["gpu"] == 4 and released["CPU"] == "48-63"
    assert released["coordinator_release"] and released["performance_granted"]
    assert released["cases"] == 24 and released["BAAB_rounds"] == 3
    assert released["graph_calls"] == [128, 256] and released["samples"] >= 31
    assert released["acceptance_scope"] == "original operator contract"
    assert released["native_compilers_drained"] and released["stable_window_granted"]
    assert time.time() >= released["start_epoch"] > released["issued_epoch"]
    assert os.environ["ZE_AFFINITY_MASK"] == "4"
    assert os.environ["LD_LIBRARY_PATH"].split(":")[0] == "/opt/gfx-deps/profiling_tools/lib"
    for key, value in prerequisites().items():
        assert released["binding"][key] == value
    for role in ("SOURCE", "AFTER"):
        assert (OUTPUT / f"CPU-{role}.exit").read_text().strip() == "0"
        assert released["CPU_provider_SHA256"][role] == source.digest(OUTPUT / f"CPU-{role}.json")
    assert released["CPU_dataset_SHA256"] == source.digest(OUTPUT / "CPU-dataset.json")
    return released


def resources():
    stat = os.statvfs(OUTPUT)
    available = stat.f_bavail * stat.f_frsize
    used = sum(path.stat().st_size for path in OUTPUT.rglob("*") if path.is_file())
    assert available >= 100 * 2**30 and used <= 640 * 2**30


def snapshot(torch, tensors, stem, expected, baseline):
    path = OUTPUT / (stem + ".pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensors, path)
    result = dict(file_SHA256=source.digest(path), tensors=helpers.metadata(torch, tensors), checks={})
    failed = False
    for name, tensor in tensors.items():
        norm = name == "norm"
        target = expected["norm" if norm else "stream"] if name != "widen" else tensors["norm"].float()
        tolerance = 5e-3 if norm else 1e-5
        nan_count = inf_count = mismatch_count = bit_count = 0
        for start in range(0, tensor.shape[0], 512):
            actual = tensor[start:start + 512]
            nan_count += torch.isnan(actual).sum().item()
            inf_count += torch.isinf(actual).sum().item()
            mismatch_count += (~torch.isclose(actual.float(), target[start:start + 512].float(), rtol=tolerance, atol=tolerance, equal_nan=False)).sum().item()
            bits = torch.int16 if tensor.dtype == torch.bfloat16 else torch.int32
            bit_count += (actual.contiguous().view(bits) != baseline[name][start:start + 512].contiguous().view(bits)).sum().item()
        widened_exact = name != "widen" or torch.equal(tensor.view(torch.int32), tensors["norm"].float().view(torch.int32))
        result["checks"][name] = dict(elements=tensor.numel(), nan=nan_count, inf=inf_count,
            strict_mismatch=mismatch_count, full_SOURCE_bit_mismatch=bit_count, widened_exact=widened_exact)
        failed |= bool(nan_count or inf_count or mismatch_count or bit_count or not widened_exact)
        if failed:
            torch.save(dict(nan=torch.isnan(tensor), inf=torch.isinf(tensor)), OUTPUT / (stem + "-" + name + "-finite.pt"))
    write(stem + ".json", result)
    assert not failed, "full actual outputs saved before strict failure; stop, no retry"


def run(role, round_number, slot):
    release = grant()
    assert (role, slot) in (("AFTER", 0), ("SOURCE", 1), ("SOURCE", 2), ("AFTER", 3))
    assert round_number in (1, 2, 3)
    prefix = f"round{round_number}/slot{slot}-{role}"
    assert not (OUTPUT / prefix).exists(), "once-only; no replay/retry of a provider slot"
    torch, invoke, maps = helpers.provider(role)
    assert maps == json.loads((OUTPUT / f"CPU-{role}.json").read_text())["maps"]
    assert torch.xpu.is_available()
    from sgl_kernel.utils import is_xe3_arch
    assert is_xe3_arch()
    libraries = {str(Path(line.split()[-1]).resolve()) for line in Path("/proc/self/maps").read_text().splitlines()
                 if "libze_intel_gpu" in line or "libsycl.so" in line}
    driver = release["binding"]["driver_proof"]
    mapped = {name: source.digest(Path(name)) for name in libraries}
    assert any("libze_intel_gpu" in name for name in mapped)
    assert all(value == driver["GPU_driver_candidate_libraries_sha256"][name] for name, value in mapped.items() if "libze_intel_gpu" in name)
    assert any(value == driver["SYCL_runtime"]["sha256"] for name, value in mapped.items() if "libsycl" in name)
    write(prefix + "/live-provider.json", dict(pid=os.getpid(),
        starttick=Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19], maps=maps,
        driver_libraries_SHA256=mapped, loaded_stack=history.stack(),
        gpu=4, source_or_model_bridge_changed=False, operator_contract_only=True))
    measured, _ = timer()
    for case in history.cases():
        resources()
        key = case["case"]
        inputs = load(torch, f"fixtures/{key}-inputs")
        expected = load(torch, f"fixtures/{key}-oracle")
        baseline = load(torch, f"SOURCE/{key}-actual")
        device = {name: value.to("xpu") for name, value in inputs.items()}
        norm = torch.empty_like(device["branch"])
        updated = torch.empty_like(device["stream"])
        widened = torch.empty_like(updated) if case["widen"] else None

        def invoke_op():
            invoke(*(device[name] for name in ("branch", "stream", "post", "pre", "alpha", "beta")),
                norm, updated, 1e-8, 1e-5, case["post_mode"] == "input_scale", case["round_pre"], widened)

        def outputs():
            result = dict(norm=norm.cpu(), stream=updated.cpu())
            if widened is not None:
                result["widen"] = widened.cpu()
            return result

        invoke_op()
        torch.xpu.synchronize()
        snapshot(torch, outputs(), prefix + "/" + key + "-pre", expected, baseline)
        torch.xpu.synchronize()
        time.sleep(2)
        for calls in release["graph_calls"]:
            args = SimpleNamespace(warmup=15, iterations=calls, samples=release["samples"], cpu_submit_trace=True)
            result = measured(torch, invoke_op, args)
            result.update(case=case, role=role, round=round_number, slot=slot,
                original_operator_contract=True, model_path_passed=False,
                cpu_spans_not_subtracted_from_device_intervals=True, exclusive_kernel_time_claim=False)
            write(prefix + f"/{key}-calls{calls}-perf.json", result)
            torch.xpu.synchronize()
            time.sleep(2)
        snapshot(torch, outputs(), prefix + "/" + key + "-post", expected, baseline)
        for name, value in device.items():
            copied = value.cpu()
            assert helpers.tensor_digest(torch, copied) == helpers.tensor_digest(torch, inputs[name]), "input mutated; immutable source input already saved"
        del inputs, expected, baseline, device, norm, updated, widened, invoke_op, outputs
        torch.xpu.synchronize()
        gc.collect()
        torch.xpu.empty_cache()
        time.sleep(2)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--mode", choices=("cpu", "provider", "run"), required=True)
    parser.add_argument("--role", choices=("SOURCE", "AFTER"))
    parser.add_argument("--round", type=int, choices=(1, 2, 3))
    parser.add_argument("--slot", type=int, choices=(0, 1, 2, 3))
    args = parser.parse_args()
    source.scope()
    if args.mode == "cpu":
        assert args.role is args.round is args.slot is None
        cpu()
    elif args.mode == "provider":
        assert args.round is args.slot is None and args.role is not None
        provider_cpu(args.role)
    else:
        assert args.role is not None and args.round is not None and args.slot is not None
        run(args.role, args.round, args.slot)


if __name__ == "__main__":
    main()
