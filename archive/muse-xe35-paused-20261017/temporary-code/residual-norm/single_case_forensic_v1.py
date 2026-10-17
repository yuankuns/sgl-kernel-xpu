import argparse
import ast
import gc
import json
import os
from pathlib import Path
import time

import after_pre_wave_accuracy_v4 as after
import source_compiler_accuracy as source


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "single-case-forensic-v1"
BARRIER = ROOT / "gpu7-single-case-forensic-v1-grant.json"
PHASE = "residual-single-case-forensic-v1"
SCRIPTS = ("single_case_forensic_v1.py", "single_case_forensic_v1_container.sh",
           "run_single_case_forensic_v1.py", "after_pre_wave_accuracy_v4.py",
           "source_compiler_accuracy.py", "check_canonical_package_v3.py")


def write(name, payload):
    (OUTPUT / name).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def provider(name):
    if name == "SOURCE":
        torch, _, invoke, proof = source.normal_import("fresh")
    else:
        manifest, expected, _ = after.provenance()
        torch, invoke, proof = after.normal_provider(manifest, expected)
    return torch, invoke, proof


def cpu(name):
    _, _, record = after.provenance()
    torch, _, proof = provider(name)
    assert not torch.xpu.is_initialized()
    driver = after.driver_proof()
    write(f"CPU-{name}-maps.json", proof)
    write(f"CPU-{name}-driver.json", driver)
    record.update(provider=name, GPU_initialized=False,
                  provider_maps_sha256=source.digest(OUTPUT / f"CPU-{name}-maps.json"),
                  driver_sha256=source.digest(OUTPUT / f"CPU-{name}-driver.json"))
    write(f"CPU-{name}-gate.json", record)


def binding():
    _, _, current = after.provenance()
    before = json.loads((OUTPUT / "CPU-SOURCE-gate.json").read_text())
    port = json.loads((OUTPUT / "CPU-AFTER-gate.json").read_text())
    for key, value in current.items():
        assert before[key] == port[key] == value, key
    assert before["driver_sha256"] == port["driver_sha256"]
    for name, gate in (("SOURCE", before), ("AFTER", port)):
        assert source.digest(OUTPUT / f"CPU-{name}-maps.json") == gate["provider_maps_sha256"]
        assert source.digest(OUTPUT / f"CPU-{name}-driver.json") == gate["driver_sha256"]
    assert after.driver_proof() == json.loads((OUTPUT / "CPU-AFTER-driver.json").read_text())
    scripts = {name: source.digest(ROOT / name) for name in SCRIPTS}
    return dict(current, driver_sha256=port["driver_sha256"],
                SOURCE_CPU_maps_sha256=before["provider_maps_sha256"],
                AFTER_CPU_maps_sha256=port["provider_maps_sha256"], script_sha256=scripts,
                oracle_benchmark_sha256=source.digest(after.BENCH),
                approved_plan_sha256=source.digest(ROOT / "RESIDUAL_SINGLE_CASE_FORENSIC_PLAN_v1.md"))


def grant():
    released = json.loads(BARRIER.read_text())
    assert released["phase"] == PHASE and released["gpu"] == 7
    assert released["issuer"] == "main coordinator user authorization"
    assert released["accuracy_only"] and released["no_timing_events_graphs_warmup"]
    assert released["case"] == "32768_weight_0_1"
    assert released["scope"] == "B8/ISL4096/OSL16/TP1/EP1"
    assert released["coordinator_release"] and not released["formal_perf_barrier_consumed"]
    assert released["invocations_per_provider"] == 1
    assert released["tolerances"] == {"norm": 5e-3, "residual": 1e-5, "widen": 1e-5, "widen_exact": True}
    for key, value in binding().items():
        assert released[key] == value, key
    assert time.time() >= released["start_epoch"]
    return released


def metadata(torch, tensors):
    result = {}
    for name, tensor in tensors.items():
        result[name] = dict(dtype=str(tensor.dtype), shape=list(tensor.shape),
                            stride=list(tensor.stride()), storage_offset=tensor.storage_offset(),
                            data_ptr=tensor.data_ptr(), sha256=source.tensor_digest(tensor))
    return result


def save_finite(torch, tensors, label):
    summary = {}
    for name, tensor in tensors.items():
        nan_mask = torch.empty(tensor.shape, dtype=torch.bool)
        inf_mask = torch.empty(tensor.shape, dtype=torch.bool)
        for start in range(0, tensor.shape[0], 512):
            nan_mask[start:start + 512] = torch.isnan(tensor[start:start + 512])
            inf_mask[start:start + 512] = torch.isinf(tensor[start:start + 512])
        path = OUTPUT / f"{label}-{name}-finite-masks.pt"
        torch.save(dict(nan=nan_mask, inf=inf_mask), path)
        summary[name] = dict(nan_count=nan_mask.sum().item(), inf_count=inf_mask.sum().item(),
                             global_positions_full_masks=path.name, file_sha256=source.digest(path))
        del nan_mask, inf_mask
    write(f"{label}-finite.json", summary)
    return summary


def driver_maps(name):
    pid = os.getpid()
    stat = Path("/proc/self/stat").read_text()
    starttick = int(stat[stat.rfind(")") + 2:].split()[19])
    maps = Path("/proc/self/maps").read_text()
    (OUTPUT / f"{name}-live-runtime.maps").write_text(maps)
    libraries = {}
    for line in maps.splitlines():
        filename = line.split()[-1]
        if filename.startswith("/") and any(token in filename for token in
                ("libsycl", "libze_", "libigc", "libigdgmm", "libur_")):
            path = Path(filename).resolve()
            if path.is_file():
                libraries[str(path)] = source.digest(path)
    expected_driver = json.loads((OUTPUT / "CPU-AFTER-driver.json").read_text())
    matched = {path: value for path, value in libraries.items()
               if path in expected_driver["GPU_driver_candidate_libraries_sha256"]}
    write(f"{name}-live-runtime-driver.json", dict(pid=pid, starttick=starttick,
          collected_inside_live_worker=True, mapped_libraries_sha256=libraries,
          CPU_driver_candidate_matches=matched,
          maps_sha256=source.digest(OUTPUT / f"{name}-live-runtime.maps")))
    assert any("libsycl" in path for path in libraries)
    assert any("libze_intel_gpu" in path for path in libraries)
    assert any("libze_intel_gpu" in path for path in matched)
    assert all(value == expected_driver["GPU_driver_candidate_libraries_sha256"][path]
               for path, value in matched.items())


def run(name):
    grant()
    torch, invoke, proof = provider(name)
    assert os.environ.get("ZE_AFFINITY_MASK") == "7"
    assert torch.xpu.is_available(), "XPU unavailable"
    from sgl_kernel.utils import is_xe3_arch
    assert is_xe3_arch(), "Only Xe35 capability 3.5"
    torch.xpu.init()
    write(f"{name}-normal-provider.json", proof)
    driver_maps(name)
    if name == "SOURCE":
        torch.manual_seed(42)
        device_inputs = dict(
            branch=torch.randn(32768, 8192, device="xpu", dtype=torch.bfloat16),
            stream=torch.randn(32768, 8192, device="xpu", dtype=torch.float32),
            post=torch.randn(8192, device="xpu", dtype=torch.bfloat16),
            pre=torch.randn(8192, device="xpu", dtype=torch.bfloat16),
            alpha=torch.rand(8192, device="xpu", dtype=torch.float32),
            beta=torch.rand(8192, device="xpu", dtype=torch.float32))
        torch.xpu.synchronize()
        cpu_inputs = {key: value.cpu() for key, value in device_inputs.items()}
        torch.save(cpu_inputs, OUTPUT / "shared-new-inputs.pt")
        write("shared-new-inputs.json", dict(new_dataset_not_missing_failed_raw_replay=True,
              seed=42, recipe="original GPU randn/rand recipe", case="32768_weight_0_1",
              post_eps=1e-8, pre_eps=1e-5, post_norm_input_scale=False,
              round_pre_norm_input=False, output_fp32=True, tensors=metadata(torch, cpu_inputs),
              file_sha256=source.digest(OUTPUT / "shared-new-inputs.pt")))
        save_finite(torch, cpu_inputs, "inputs")
        del device_inputs, cpu_inputs
        gc.collect()
        torch.xpu.empty_cache()
    input_record = json.loads((OUTPUT / "shared-new-inputs.json").read_text())
    assert source.digest(OUTPUT / "shared-new-inputs.pt") == input_record["file_sha256"]
    inputs = torch.load(OUTPUT / "shared-new-inputs.pt", map_location="cpu", mmap=True, weights_only=True)
    device_inputs = {}
    layout = {}
    for key, value in inputs.items():
        assert source.tensor_digest(value) == input_record["tensors"][key]["sha256"]
        device_value = value.to("xpu")
        copied = device_value.cpu()
        assert source.tensor_digest(copied) == input_record["tensors"][key]["sha256"]
        device_inputs[key] = device_value
        layout[key] = dict(dtype=str(device_value.dtype), shape=list(device_value.shape),
                           stride=list(device_value.stride()), storage_offset=device_value.storage_offset(),
                           data_ptr=device_value.data_ptr(), full_copy_sha_verified=True)
        del copied
    norm = torch.empty_like(device_inputs["branch"])
    updated = torch.empty_like(device_inputs["stream"])
    widened = torch.empty_like(updated)
    pointers = [tensor.data_ptr() for tensor in device_inputs.values()] + [norm.data_ptr(), updated.data_ptr(), widened.data_ptr()]
    assert len(set(pointers)) == len(pointers)
    write(f"{name}-input-layout.json", dict(tensors=layout, nonalias=True, stream=str(torch.xpu.current_stream()),
          post_eps=1e-8, pre_eps=1e-5, post_norm_input_scale=False, round_pre_norm_input=False, output_fp32=True))
    write(f"{name}-invoke-state.json", dict(invocations_started=1, invocations_completed=0))
    invoke(*(device_inputs[key] for key in ("branch", "stream", "post", "pre", "alpha", "beta")),
           norm, updated, 1e-8, 1e-5, False, False, widened)
    torch.xpu.synchronize()
    actual = dict(norm=norm.cpu(), stream=updated.cpu(), widen=widened.cpu())
    path = OUTPUT / f"{name}-actual.pt"
    torch.save(actual, path)
    write(f"{name}-actual.json", dict(tensors=metadata(torch, actual), file_sha256=source.digest(path),
          invocations=1, outputs_saved_before_precision_assertions=True))
    save_finite(torch, actual, name)
    readonly = {}
    for key, value in device_inputs.items():
        current = value.cpu()
        readonly[key] = source.tensor_digest(current) == input_record["tensors"][key]["sha256"]
        if not readonly[key]:
            torch.save(current, OUTPUT / f"{name}-mutated-{key}.pt")
        del current
    write(f"{name}-input-readonly.json", readonly)
    write(f"{name}-invoke-state.json", dict(invocations_started=1, invocations_completed=1))
    torch.xpu.synchronize()
    time.sleep(2)
    assert all(readonly.values()), "Input mutation; actual data preserved"


def oracle():
    import torch
    assert not torch.xpu.is_initialized()
    inputs = torch.load(OUTPUT / "shared-new-inputs.pt", map_location="cpu", mmap=True, weights_only=True)
    tree = ast.parse(after.BENCH.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ("round_bfloat16_xe35", "reference_xe35")]
    assert len(functions) == 2
    namespace = dict(torch=torch, POST_EPS=1e-8, PRE_EPS=1e-5)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(after.BENCH), "exec"), namespace)
    expected = dict(norm=torch.empty((32768, 8192), dtype=torch.float32),
                    stream=torch.empty((32768, 8192), dtype=torch.float32),
                    post=torch.empty((32768, 8192), dtype=torch.bfloat16))
    for start in range(0, 32768, 512):
        result = namespace["reference_xe35"](inputs["branch"][start:start + 512],
                 inputs["stream"][start:start + 512], *(inputs[key] for key in ("post", "pre", "alpha", "beta")), False, False)
        for key, value in zip(("norm", "stream", "post"), result, strict=True):
            expected[key][start:start + 512] = value
    torch.save(expected, OUTPUT / "full-original-CPU-oracle.pt")
    write("full-original-CPU-oracle.json", dict(tensors=metadata(torch, expected),
          file_sha256=source.digest(OUTPUT / "full-original-CPU-oracle.pt"), all_elements=True,
          original_FP64_math_BF16_boundaries=True, benchmark_sha256=source.digest(after.BENCH)))
    save_finite(torch, expected, "oracle")
    assert not torch.xpu.is_initialized()


def compare():
    import torch
    assert not torch.xpu.is_initialized()
    for name in ("SOURCE-actual", "AFTER-actual", "full-original-CPU-oracle", "shared-new-inputs"):
        record = json.loads((OUTPUT / f"{name}.json").read_text())
        filename = "shared-new-inputs.pt" if name == "shared-new-inputs" else f"{name}.pt"
        assert source.digest(OUTPUT / filename) == record["file_sha256"]
    expected = torch.load(OUTPUT / "full-original-CPU-oracle.pt", map_location="cpu", mmap=True, weights_only=True)
    outputs = {name: torch.load(OUTPUT / f"{name}-actual.pt", map_location="cpu", mmap=True, weights_only=True)
               for name in ("SOURCE", "AFTER")}
    results = {}
    checks = []
    for name, actual in outputs.items():
        checks.extend([(f"{name}-norm-oracle", actual["norm"], expected["norm"], 5e-3, False),
                       (f"{name}-residual-oracle", actual["stream"], expected["stream"], 1e-5, False),
                       (f"{name}-widen-norm", actual["widen"], actual["norm"], 1e-5, False),
                       (f"{name}-widen-exact", actual["widen"], actual["norm"], 0, True)])
    for key in ("norm", "stream", "widen"):
        checks.append((f"SOURCE-AFTER-{key}-bitwise", outputs["SOURCE"][key], outputs["AFTER"][key], 0, True))
    for label, left, right, tolerance, bitwise in checks:
        mask = torch.empty((32768, 8192), dtype=torch.bool)
        first_positions = []
        count = 0
        for start in range(0, 32768, 512):
            left_chunk, right_chunk = left[start:start + 512], right[start:start + 512]
            if bitwise:
                bits = torch.int16 if left.dtype == right.dtype == torch.bfloat16 else torch.int32
                mismatch = left_chunk.contiguous().view(bits) != (right_chunk if right.dtype == left.dtype else right_chunk.float()).contiguous().view(bits)
            else:
                mismatch = ~torch.isclose(left_chunk.float(), right_chunk.float(), rtol=tolerance, atol=tolerance, equal_nan=False)
            mask[start:start + 512] = mismatch
            count += mismatch.sum().item()
            if len(first_positions) < 32:
                positions = torch.nonzero(mismatch, as_tuple=False)[:32 - len(first_positions)]
                first_positions.extend([[start + row, column] for row, column in positions.tolist()])
        path = OUTPUT / f"{label}-mismatch-mask.pt"
        torch.save(mask, path)
        results[label] = dict(mismatching_elements=count, total_elements=32768 * 8192,
                             first_global_positions=first_positions, full_global_mask=path.name,
                             file_sha256=source.digest(path), bitwise=bitwise, rtol=tolerance, atol=tolerance)
        del mask
        write("full-comparison.json", results)
    passed = all(record["mismatching_elements"] == 0 for record in results.values())
    write("classification.json", dict(current_same_case_pass=passed, old_failure_actual_exit=1,
          original_missing_failed_inputs_replayed=False, old_failure_explained=False,
          precision_acceptance_24cases=False, performance_approved=False,
          kernel_rootcause_proven=False, all_raw_outputs_oracle_saved=True))
    assert not torch.xpu.is_initialized()
    if not passed:
        raise AssertionError("Same-case strict accuracy/bitwise failure; all data saved")


def main():
    source.scope()
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--mode", required=True, choices=("CPU-SOURCE", "CPU-AFTER", "binding", "SOURCE", "AFTER", "oracle", "compare", "post"))
    args = parser.parse_args()
    assert set(os.sched_getaffinity(0)) <= set(range(48, 64))
    if args.mode.startswith("CPU-"):
        cpu(args.mode.removeprefix("CPU-"))
    elif args.mode in ("binding", "post"):
        write("binding.json" if args.mode == "binding" else "post-protected-source.json", binding())
    elif args.mode in ("SOURCE", "AFTER"):
        run(args.mode)
    elif args.mode == "oracle":
        oracle()
    else:
        compare()


if __name__ == "__main__":
    main()
