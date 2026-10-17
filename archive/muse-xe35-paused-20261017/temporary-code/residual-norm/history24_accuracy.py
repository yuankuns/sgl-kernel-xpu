import argparse
import ast
import gc
import json
import os
from pathlib import Path
import platform
import time

import after_pre_wave_accuracy_v4 as after
import single_case_forensic_v2 as helpers
import source_compiler_accuracy as source


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "history24-accuracy-v1"
BARRIER = ROOT / "gpu7-history24-accuracy-v1-grant.json"
PHASE = "residual-history24-accuracy-v1"
SCRIPTS = ("history24_accuracy.py", "history24_stage.py", "history24_container.sh", "run_history24_accuracy.py",
           "single_case_forensic_v2.py", "source_compiler_accuracy.py", "after_pre_wave_accuracy_v4.py", "check_canonical_package_v3.py")


def write(name, payload):
    path = OUTPUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def cases():
    audit = json.loads((ROOT / "RESIDUAL_HISTORY24_CPU_AUDIT_v1.json").read_text())
    own_tree = ast.parse((ROOT / "history24_accuracy.py").read_text())
    generator = next(node for node in own_tree.body if isinstance(node, ast.FunctionDef) and node.name == "generate_case")
    recipe = []
    for node in generator.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "torch.manual_seed":
            recipe.append(node)
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ("branch", "stream", "post", "pre", "alpha", "beta"):
            recipe.append(node)
    import hashlib
    recipe_SHA = hashlib.sha256(ast.dump(ast.Module(body=recipe, type_ignores=[]), include_attributes=False).encode()).hexdigest()
    assert recipe_SHA == audit["SOURCE_actual_matrix_recipe"]["recipe_AST_sha256"]
    assert audit["actual_matrix_recipe_AST_exact_equal"] and all(audit["reference_math_AST_equal_after_suffix_normalization"].values())
    assert len(audit["case_order"]) == 24
    for role in ("SOURCE", "AFTER"):
        record = audit[f"{role}_actual_matrix_recipe"]
        assert source.digest(ROOT / Path(record["path"]).name) == record["file_sha256"]
    return audit["case_order"]


def stack():
    loaded = [line for line in Path("/proc/modules").read_text().splitlines() if line.startswith("xe ")]
    fields = {}
    for name in ("version", "srcversion", "taint"):
        path = Path("/sys/module/xe") / name
        fields[name] = path.read_text().strip() if path.exists() else None
    assert loaded and platform.release().startswith("7.0.0-38")
    return dict(running_kernel=platform.release(), loaded_xe=loaded, xe_sysfs=fields,
                external_DKMS_may_compile_other_kernel=True, loaded_xe_binary_SHA_not_available=True)


def resources(label, required_bytes=0):
    usage = sum(path.stat().st_size for path in OUTPUT.rglob("*") if path.is_file())
    disk = os.statvfs(OUTPUT)
    available = disk.f_bavail * disk.f_frsize
    record = dict(label=label, epoch=time.time(), artifact_bytes=usage, available_bytes=available,
                  cap_bytes=256 * 2**30, headroom_bytes=100 * 2**30, required_bytes=required_bytes)
    with (OUTPUT / "disk-snapshots.jsonl").open("a") as logfile:
        logfile.write(json.dumps(record) + "\n")
    assert usage + required_bytes <= record["cap_bytes"] and available - required_bytes >= record["headroom_bytes"]


def cpu_smoke():
    import torch
    assert not torch.xpu.is_initialized()
    records = {}
    for filename, expected in (("shared-new-inputs.pt", helpers.INPUT_SHA), ("full-original-CPU-oracle.pt", helpers.ORACLE_SHA)):
        path = helpers.DATA / filename
        assert source.digest(path) == expected
        tensors = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        records[filename] = dict(file_sha256=expected, tensors=helpers.metadata(torch, tensors))
        del tensors
    audit = json.loads((ROOT / "RESIDUAL_HISTORY24_CPU_AUDIT_v1.json").read_text())
    scopes = {role: audit[f"{role}_precision_tests"] for role in ("SOURCE", "AFTER")}
    for role, record in scopes.items():
        path = Path(record["path"])
        if str(path).startswith("/home/syk/"):
            path = Path("/root") / path.relative_to("/home/syk")
        assert source.digest(path) == record["sha256"]
    write("CPU-pytest-collection-plan.json", dict(scopes=scopes, static_AST_collection_only=True,
          actual_pytest_collection_run=False, reason="normal pytest collection calls XPU architecture guard; GPU pytest needs separate grant"))
    resources("CPU-start")
    disk = os.statvfs(OUTPUT)
    assert disk.f_bavail * disk.f_frsize >= 356 * 2**30
    write("CPU-smoke.json", dict(full_local_explicit_torch_metadata_digest0=True, records=records,
          GPU_initialized=False, case_order=cases(), stack=stack(), generator_recipe_AST_sha256=audit["SOURCE_actual_matrix_recipe"]["recipe_AST_sha256"],
          plan_sha256=source.digest(ROOT / "RESIDUAL_HISTORY24_CPU_PLAN_v1.md")))
    assert not torch.xpu.is_initialized()


def cpu_provider(role):
    assert (OUTPUT / "CPU-smoke.exit").read_text().strip() == "0"
    _, _, record = after.provenance()
    torch, _, proof = helpers.provider(role)
    assert not torch.xpu.is_initialized()
    write(f"CPU-{role}-maps.json", proof)
    write(f"CPU-{role}-driver.json", after.driver_proof())
    record.update(provider_maps_sha256=source.digest(OUTPUT / f"CPU-{role}-maps.json"),
                  driver_sha256=source.digest(OUTPUT / f"CPU-{role}-driver.json"), GPU_initialized=False)
    write(f"CPU-{role}-gate.json", record)


def binding():
    assert (OUTPUT / "CPU-smoke.exit").read_text().strip() == "0"
    _, _, record = after.provenance()
    maps = {}
    for role in ("SOURCE", "AFTER"):
        assert (OUTPUT / f"CPU-{role}.exit").read_text().strip() == "0"
        previous = json.loads((OUTPUT / f"CPU-{role}-gate.json").read_text())
        assert previous["GPU_initialized"] is False
        for key, value in record.items():
            assert previous[key] == value
        assert source.digest(OUTPUT / f"CPU-{role}-maps.json") == previous["provider_maps_sha256"]
        driver = json.loads((OUTPUT / f"CPU-{role}-driver.json").read_text())
        assert driver == after.driver_proof()
        assert source.digest(OUTPUT / f"CPU-{role}-driver.json") == previous["driver_sha256"]
        maps[role] = previous["provider_maps_sha256"]
    current_stack = stack()
    initial = json.loads((OUTPUT / "CPU-smoke.json").read_text())["stack"]
    assert current_stack["running_kernel"] == initial["running_kernel"] and current_stack["xe_sysfs"] == initial["xe_sysfs"]
    cases()
    record.update(CPU_provider_maps_sha256=maps, driver_sha256=source.digest(OUTPUT / "CPU-AFTER-driver.json"),
                  CPU_smoke_sha256=source.digest(OUTPUT / "CPU-smoke.json"),
                  CPU_pytest_scope_sha256=source.digest(OUTPUT / "CPU-pytest-collection-plan.json"),
                  plan_sha256=source.digest(ROOT / "RESIDUAL_HISTORY24_CPU_PLAN_v1.md"),
                  recipe_audit_sha256=source.digest(ROOT / "RESIDUAL_HISTORY24_CPU_AUDIT_v1.json"),
                  benchmark_sha256=source.digest(after.BENCH),
                  script_sha256={name: source.digest(ROOT / name) for name in SCRIPTS})
    return record


def grant():
    released = json.loads(BARRIER.read_text())
    assert released["phase"] == PHASE and released["issuer"] == "main coordinator user authorization"
    assert released["gpu"] == 7 and released["accuracy_only"] and released["coordinator_release"]
    assert released["scope"] == "B8/ISL4096/OSL16/TP1/EP1" and released["cases"] == 24
    assert released["no_timing_events_graphs_warmup"] and not released["formal_perf_barrier_consumed"]
    assert time.time() >= released["start_epoch"]
    for key, value in binding().items():
        assert released[key] == value


def initialize(role, label=None):
    label = label or role
    grant()
    torch, invoke, proof = helpers.provider(role)
    assert os.environ.get("ZE_AFFINITY_MASK") == "7" and torch.xpu.is_available()
    from sgl_kernel.utils import is_xe3_arch
    assert is_xe3_arch(), "Xe35 capability 3.5 only"
    torch.xpu.init()
    write(f"{label}-normal-provider.json", proof)
    stat = Path("/proc/self/stat").read_text()
    maps = Path("/proc/self/maps").read_text()
    (OUTPUT / f"{label}-runtime.maps").write_text(maps)
    libraries = {}
    for line in maps.splitlines():
        filename = line.split()[-1]
        if filename.startswith("/") and any(name in filename for name in ("libsycl", "libze_", "libigc", "libigdgmm", "libur_")):
            path = Path(filename).resolve()
            if path.is_file():
                libraries[str(path)] = source.digest(path)
    driver = json.loads((OUTPUT / "CPU-AFTER-driver.json").read_text())
    matched = {path: value for path, value in libraries.items() if path in driver["GPU_driver_candidate_libraries_sha256"]}
    write(f"{label}-runtime-driver.json", dict(pid=os.getpid(), starttick=int(stat[stat.rfind(")") + 2:].split()[19]),
          live_worker_internal_collection=True, libraries_sha256=libraries, CPU_candidate_matches=matched, stack=stack()))
    assert any("libze_intel_gpu" in path for path in matched)
    assert all(value == driver["GPU_driver_candidate_libraries_sha256"][path] for path, value in matched.items())
    assert any(value == driver["SYCL_runtime"]["sha256"] for path, value in libraries.items() if "libsycl" in path)
    return torch, invoke


def memory(torch, label):
    with (OUTPUT / "allocator-history.jsonl").open("a") as logfile:
        logfile.write(json.dumps(dict(label=label, pid=os.getpid(), allocated_bytes=torch.xpu.memory_allocated(),
                  reserved_bytes=torch.xpu.memory_reserved(), peak_allocated_bytes=torch.xpu.max_memory_allocated(),
                  exact_original_allocator_replay=False)) + "\n")


def save(torch, tensors, stem):
    path = OUTPUT / (stem + ".pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensors, path)
    record = dict(file_sha256=source.digest(path), tensors=helpers.metadata(torch, tensors))
    write(stem + ".json", record)
    return record


def finite(torch, tensors, stem):
    result = {}
    for name, tensor in tensors.items():
        nan = torch.empty(tensor.shape, dtype=torch.bool)
        inf = torch.empty(tensor.shape, dtype=torch.bool)
        first = []
        for start in range(0, tensor.shape[0], 512):
            nan[start:start + 512] = torch.isnan(tensor[start:start + 512])
            inf[start:start + 512] = torch.isinf(tensor[start:start + 512])
            if len(first) < 32:
                positions = torch.nonzero(nan[start:start + 512], as_tuple=False)[:32 - len(first)]
                first.extend([[start + row] + rest for row, *rest in positions.tolist()])
        path = OUTPUT / f"{stem}-{name}-finite.pt"
        torch.save(dict(nan=nan, inf=inf), path)
        result[name] = dict(nan_count=nan.sum().item(), inf_count=inf.sum().item(), first_global_NaN_positions=first,
                            full_global_masks=path.relative_to(OUTPUT).as_posix(), file_sha256=source.digest(path))
        del nan, inf
    write(stem + "-finite.json", result)
    return result


def load(torch, stem):
    record = json.loads((OUTPUT / (stem + ".json")).read_text())
    path = OUTPUT / (stem + ".pt")
    assert source.digest(path) == record["file_sha256"]
    tensors = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    for name, tensor in tensors.items():
        assert helpers.tensor_digest(torch, tensor) == record["tensors"][name]["sha256"]
    return tensors, record


def generate_case(torch, case, reference):
    rows, mode, rounded = case["rows"], case["post_mode"], case["round_pre"]
    key = case["case"]
    save(torch, dict(CPU=torch.get_rng_state(), XPU=torch.xpu.get_rng_state()), f"fixtures/{key}-RNG-before")
    torch.manual_seed(42)
    save(torch, dict(CPU=torch.get_rng_state(), XPU=torch.xpu.get_rng_state()), f"fixtures/{key}-RNG-seeded")
    branch = torch.randn(rows, 8192, device="xpu", dtype=torch.bfloat16)
    stream = torch.randn(rows, 8192, device="xpu", dtype=torch.float32)
    post = (torch.empty(0, device="xpu", dtype=torch.bfloat16) if mode == "none" else torch.randn(8192, device="xpu", dtype=torch.bfloat16))
    pre = torch.randn(8192, device="xpu", dtype=torch.bfloat16)
    alpha = torch.rand(8192, device="xpu", dtype=torch.float32)
    beta = torch.rand(8192, device="xpu", dtype=torch.float32)
    torch.xpu.synchronize()
    inputs = {name: value.cpu() for name, value in dict(branch=branch, stream=stream, post=post, pre=pre, alpha=alpha, beta=beta).items()}
    save(torch, inputs, f"fixtures/{key}-inputs")
    save(torch, dict(CPU=torch.get_rng_state(), XPU=torch.xpu.get_rng_state()), f"fixtures/{key}-RNG-drawn")
    expected = dict(norm=torch.empty((rows, 8192), dtype=torch.float32), stream=torch.empty((rows, 8192), dtype=torch.float32),
                    post=torch.empty((rows, 8192), dtype=torch.bfloat16))
    for start in range(0, rows, 512):
        result = reference(inputs["branch"][start:start + 512], inputs["stream"][start:start + 512], *(inputs[name] for name in ("post", "pre", "alpha", "beta")), mode == "input_scale", rounded)
        for name, value in zip(("norm", "stream", "post"), result, strict=True):
            expected[name][start:start + 512] = value
    save(torch, expected, f"fixtures/{key}-oracle")
    inputs_finite = finite(torch, inputs, f"fixtures/{key}-inputs")
    oracle_finite = finite(torch, expected, f"fixtures/{key}-oracle")
    write(f"fixtures/{key}-status.json", dict(case=case, new_fixture_not_missing_failed_raw=True,
          reference_invocations_per_chunk_once=True, input_finite=inputs_finite, oracle_finite=oracle_finite))
    memory(torch, "generator/" + key + "/before-cleanup")
    assert all(record["nan_count"] == record["inf_count"] == 0 for record in list(inputs_finite.values()) + list(oracle_finite.values()))


def generate():
    torch, _ = initialize("SOURCE", "generator")
    tree = ast.parse(after.BENCH.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ("round_bfloat16_xe35", "reference_xe35")]
    assert len(functions) == 2
    namespace = dict(torch=torch, POST_EPS=1e-8, PRE_EPS=1e-5)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(after.BENCH), "exec"), namespace)
    completed = []
    for case in cases():
        resources("generator/" + case["case"], case["input_bytes"] + case["oracle_bytes"] + case["full_finite_masks_bytes"])
        generate_case(torch, case, namespace["reference_xe35"])
        torch.xpu.synchronize()
        gc.collect()
        torch.xpu.empty_cache()
        time.sleep(2)
        memory(torch, "generator/" + case["case"] + "/after-cleanup")
        resources("generator/" + case["case"] + "/saved")
        completed.append(case["case"])
        write("generator-completed.json", completed)
    assert len(completed) == 24


def fixture_gate():
    import torch
    assert not torch.xpu.is_initialized()
    assert (OUTPUT / "generator.exit").read_text().strip() == "0"
    records = {}
    for case in cases():
        key = case["case"]
        records[key] = {}
        for label in ("inputs", "oracle"):
            tensors, record = load(torch, f"fixtures/{key}-{label}")
            masks = json.loads((OUTPUT / f"fixtures/{key}-{label}-finite.json").read_text())
            for value in masks.values():
                assert value["nan_count"] == value["inf_count"] == 0
                assert source.digest(OUTPUT / value["full_global_masks"]) == value["file_sha256"]
            records[key][label] = record["file_sha256"]
            del tensors
        resources("fixture-gate/" + key)
    write("fixture-gate.json", dict(all24_complete_full_inputs_oracle_SHA_finite0=True, records=records, GPU_initialized=False))


def role_case(torch, invoke, role, case):
    key = case["case"]
    save(torch, dict(CPU=torch.get_rng_state(), XPU=torch.xpu.get_rng_state()), f"{role}/{key}-RNG-before")
    inputs, input_record = load(torch, f"fixtures/{key}-inputs")
    device = {}
    layout = {}
    for name, value in inputs.items():
        uploaded = value.to("xpu")
        copied = uploaded.cpu()
        assert helpers.tensor_digest(torch, copied) == input_record["tensors"][name]["sha256"]
        device[name] = uploaded
        layout[name] = dict(shape=list(uploaded.shape), dtype=str(uploaded.dtype), stride=list(uploaded.stride()),
                            data_ptr=uploaded.data_ptr(), storage_offset=uploaded.storage_offset(), full_copy_SHA_verified=True)
        del copied
    norm = torch.empty_like(device["branch"])
    updated = torch.empty_like(device["stream"])
    widened = torch.empty_like(updated) if case["widen"] else None
    pointers = [tensor.data_ptr() for tensor in device.values() if tensor.numel()] + [norm.data_ptr(), updated.data_ptr()] + ([widened.data_ptr()] if widened is not None else [])
    assert len(set(pointers)) == len(pointers)
    write(f"{role}/{key}-layout.json", dict(case=case, tensors=layout, nonalias=True, stream=str(torch.xpu.current_stream()), post_eps=1e-8, pre_eps=1e-5))
    write(f"{role}/{key}-invoke.json", dict(started=1, completed=0))
    invoke(*(device[name] for name in ("branch", "stream", "post", "pre", "alpha", "beta")), norm, updated, 1e-8, 1e-5,
           case["post_mode"] == "input_scale", case["round_pre"], widened)
    torch.xpu.synchronize()
    actual = dict(norm=norm.cpu(), stream=updated.cpu())
    if widened is not None:
        actual["widen"] = widened.cpu()
    save(torch, actual, f"{role}/{key}-actual")
    finite(torch, actual, f"{role}/{key}-actual")
    readonly = {}
    for name, tensor in device.items():
        copied = tensor.cpu()
        readonly[name] = helpers.tensor_digest(torch, copied) == input_record["tensors"][name]["sha256"]
        if not readonly[name]:
            save(torch, {name: copied}, f"{role}/{key}-mutated-{name}")
        del copied
    write(f"{role}/{key}-invoke.json", dict(started=1, completed=1, actual_saved_before_strict=True, readonly=readonly))
    save(torch, dict(CPU=torch.get_rng_state(), XPU=torch.xpu.get_rng_state()), f"{role}/{key}-RNG-after")
    memory(torch, role + "/" + key + "/before-cleanup")
    assert all(readonly.values()), "Input mutation; raw preserved"


def role_run(role):
    assert (OUTPUT / "fixture-gate.exit").read_text().strip() == "0"
    gate = json.loads((OUTPUT / "fixture-gate.json").read_text())
    assert gate["all24_complete_full_inputs_oracle_SHA_finite0"]
    if role == "AFTER":
        assert (OUTPUT / "SOURCE.exit").read_text().strip() == "0"
    torch, invoke = initialize(role)
    completed = []
    for case in cases():
        key = case["case"]
        for label in ("inputs", "oracle"):
            assert source.digest(OUTPUT / f"fixtures/{key}-{label}.pt") == gate["records"][key][label]
        resources(role + "/" + key, case["actual_two_roles_bytes"] // 2 + case["full_finite_masks_bytes"])
        role_case(torch, invoke, role, case)
        torch.xpu.synchronize()
        gc.collect()
        torch.xpu.empty_cache()
        time.sleep(2)
        memory(torch, role + "/" + key + "/after-cleanup")
        resources(role + "/" + key + "/saved")
        completed.append(key)
        write(role + "-completed.json", completed)
    assert len(completed) == 24


def compare():
    import torch
    assert not torch.xpu.is_initialized()
    assert all((OUTPUT / f"{role}.exit").read_text().strip() == "0" for role in ("SOURCE", "AFTER"))
    all_results = {}
    for case in cases():
        key = case["case"]
        resources("comparison/" + key, case["full_difference_masks_bytes"])
        expected, _ = load(torch, f"fixtures/{key}-oracle")
        outputs = {role: load(torch, f"{role}/{key}-actual")[0] for role in ("SOURCE", "AFTER")}
        checks = []
        for role, actual in outputs.items():
            checks.extend([(role + "-norm", actual["norm"], expected["norm"], 5e-3, False),
                           (role + "-stream", actual["stream"], expected["stream"], 1e-5, False)])
            if case["widen"]:
                checks.extend([(role + "-widen", actual["widen"], actual["norm"], 1e-5, False),
                               (role + "-widen-exact", actual["widen"], actual["norm"], 0, True)])
        for name in outputs["SOURCE"]:
            checks.append(("SOURCE-AFTER-" + name, outputs["SOURCE"][name], outputs["AFTER"][name], 0, True))
        results = {}
        for label, left, right, tolerance, bitwise in checks:
            mask = torch.empty(left.shape, dtype=torch.bool)
            positions = []
            for start in range(0, case["rows"], 512):
                left_chunk, right_chunk = left[start:start + 512], right[start:start + 512]
                if bitwise:
                    bits = torch.int16 if left.dtype == right.dtype == torch.bfloat16 else torch.int32
                    mismatch = left_chunk.contiguous().view(bits) != (right_chunk if left.dtype == right.dtype else right_chunk.float()).contiguous().view(bits)
                else:
                    mismatch = ~torch.isclose(left_chunk.float(), right_chunk.float(), rtol=tolerance, atol=tolerance, equal_nan=False)
                mask[start:start + 512] = mismatch
                if len(positions) < 32:
                    found = torch.nonzero(mismatch, as_tuple=False)[:32 - len(positions)]
                    positions.extend([[start + row, column] for row, column in found.tolist()])
            path = OUTPUT / f"comparison/{key}-{label}.pt"
            path.parent.mkdir(exist_ok=True)
            torch.save(mask, path)
            results[label] = dict(mismatch=mask.sum().item(), elements=left.numel(), first_global_positions=positions,
                                  full_global_mask=path.relative_to(OUTPUT).as_posix(), sha256=source.digest(path), tolerance=tolerance, bitwise=bitwise)
            del mask
        all_results[key] = results
        write("full-comparison.json", all_results)
        del expected, outputs, checks, left, right, left_chunk, right_chunk, mismatch
        gc.collect()
    passed = all(record["mismatch"] == 0 for result in all_results.values() for record in result.values())
    write("classification.json", dict(all24_new_shared_fixture_pass=passed, old_failure_actual_exit=1, old_failure_explained=False,
          exact_missing_raw_or_allocator_replay=False, performance_approved=False, precision_pytest_run=False, model_adapter_gap_unfixed=True))
    if not passed:
        raise AssertionError("History24 strict failure; all24 actual and global masks preserved")


def historical():
    import torch
    assert not torch.xpu.is_initialized()
    old = json.loads((ROOT / "after-pre-wave-accuracy-v4/AFTER-cases.json").read_text())
    assert len(old) == 17
    results = {}
    for record in old:
        key = record["case"]
        path = ROOT / "after-pre-wave-accuracy-v4/AFTER-outputs" / (key + ".pt")
        assert source.digest(path) == record["file_sha256"]
        tensors = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        results[key] = dict(historical_input_identity_proven=False, output_SHA_matches={})
        for name, tensor in tensors.items():
            old_sha = helpers.tensor_digest(torch, tensor)
            assert old_sha == record["tensor_sha256"][name]
            results[key]["output_SHA_matches"][name] = {role: old_sha == json.loads((OUTPUT / f"{role}/{key}-actual.json").read_text())["tensors"][name]["sha256"] for role in ("SOURCE", "AFTER")}
        del tensors
    write("historical17-output-comparison.json", dict(results=results, historical_inputs_not_available=True,
          output_hash_equality_not_input_identity=True, failure18_raw_still_missing=True))


def main():
    source.scope()
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--mode", required=True, choices=("CPU-smoke", "CPU-SOURCE", "CPU-AFTER", "binding", "generator", "fixture-gate", "SOURCE", "AFTER", "compare", "historical", "post"))
    args = parser.parse_args()
    assert set(os.sched_getaffinity(0)) <= set(range(48, 64))
    if args.mode == "CPU-smoke":
        cpu_smoke()
    elif args.mode.startswith("CPU-"):
        cpu_provider(args.mode.removeprefix("CPU-"))
    elif args.mode in ("binding", "post"):
        write("binding.json" if args.mode == "binding" else "post-protected-source.json", binding())
    elif args.mode == "generator":
        generate()
    elif args.mode == "fixture-gate":
        fixture_gate()
    elif args.mode in ("SOURCE", "AFTER"):
        role_run(args.mode)
    elif args.mode == "compare":
        compare()
    else:
        historical()


if __name__ == "__main__":
    main()
