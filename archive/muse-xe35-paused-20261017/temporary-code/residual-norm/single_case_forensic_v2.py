import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import after_pre_wave_accuracy_v4 as after
import source_compiler_accuracy as source


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "single-case-forensic-v2"
BARRIER = ROOT / "gpu7-single-case-forensic-v2-grant.json"
PHASE = "residual-single-case-forensic-v2"
DATA = ROOT / "single-case-forensic-v1"
INPUT_SHA = "613cc5b0137f7d9f54ecd7f46ab8da6c2b7cccef8729f3c45712f05612b31c47"
ORACLE_SHA = "3d73796f44bc761b1ee2360e767fc0c517d86ca94f1602b8bc73d4d7c6e9c3db"
SCRIPTS = ("single_case_forensic_v2.py", "single_case_forensic_v2_container.sh",
           "run_single_case_forensic_v2.py", "after_pre_wave_accuracy_v4.py",
           "source_compiler_accuracy.py", "check_canonical_package_v3.py")


def write(name, payload):
    (OUTPUT / name).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def tensor_digest(torch, tensor):
    result = hashlib.sha256()
    for start in range(0, tensor.shape[0], 512):
        result.update(tensor[start:start + 512].contiguous().view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def metadata(torch, tensors):
    return {name: dict(dtype=str(tensor.dtype), shape=list(tensor.shape), stride=list(tensor.stride()),
                       storage_offset=tensor.storage_offset(), data_ptr=tensor.data_ptr(),
                       sha256=tensor_digest(torch, tensor)) for name, tensor in tensors.items()}


def verify_raw():
    assert source.digest(DATA / "shared-new-inputs.pt") == INPUT_SHA
    assert source.digest(DATA / "full-original-CPU-oracle.pt") == ORACLE_SHA
    assert (OUTPUT / "CPU-data.exit").read_text().strip() == "0"
    proof = json.loads((OUTPUT / "CPU-data.json").read_text())
    assert proof["input_file_sha256"] == INPUT_SHA and proof["oracle_file_sha256"] == ORACLE_SHA
    return proof


def cpu_data():
    import torch
    assert not torch.xpu.is_initialized()
    assert source.digest(DATA / "shared-new-inputs.pt") == INPUT_SHA
    assert source.digest(DATA / "full-original-CPU-oracle.pt") == ORACLE_SHA
    old = json.loads((DATA / "CPU-harness-failure-audit.json").read_text())
    assert old["GPU_initialized"] is False and old["actual_v1_exit"] == 1
    records = {}
    for label, filename in (("new_inputs", "shared-new-inputs.pt"), ("original_CPU_oracle", "full-original-CPU-oracle.pt")):
        tensors = torch.load(DATA / filename, map_location="cpu", mmap=True, weights_only=True)
        current = metadata(torch, tensors)
        assert current.keys() == old["files"][label]["tensors"].keys()
        for name, tensor in tensors.items():
            expected = old["files"][label]["tensors"][name]
            for key in ("dtype", "shape", "stride", "sha256"):
                assert current[name][key] == expected[key]
            nan_count = inf_count = 0
            for start in range(0, tensor.shape[0], 512):
                chunk = tensor[start:start + 512]
                nan_count += torch.isnan(chunk).sum().item()
                inf_count += torch.isinf(chunk).sum().item()
            assert nan_count == inf_count == expected["nan_count"] == expected["inf_count"] == 0
            assert source.digest(DATA / expected["full_global_position_masks"]) == expected["mask_sha256"]
            current[name].update(nan_count=nan_count, inf_count=inf_count,
                                 full_global_masks=str(DATA / expected["full_global_position_masks"]),
                                 masks_sha256=expected["mask_sha256"])
        records[label] = current
        del tensors
    assert not torch.xpu.is_initialized()
    write("CPU-data.json", dict(input_file_sha256=INPUT_SHA, oracle_file_sha256=ORACLE_SHA,
          reused_v1_new_dataset_not_original_failed_raw=True, oracle_recomputed=False,
          full_file_and_tensor_metadata_digest_finite_gates0=True, GPU_initialized=False, records=records,
          v1_CPU_supplement_sha256=source.digest(DATA / "CPU-harness-failure-audit.json")))


def provider(name):
    if name == "SOURCE":
        torch, _, invoke, proof = source.normal_import("fresh")
    else:
        manifest, expected, _ = after.provenance()
        torch, invoke, proof = after.normal_provider(manifest, expected)
    return torch, invoke, proof


def cpu(name):
    verify_raw()
    _, _, record = after.provenance()
    torch, _, proof = provider(name)
    assert not torch.xpu.is_initialized()
    write(f"CPU-{name}-maps.json", proof)
    write(f"CPU-{name}-driver.json", after.driver_proof())
    record.update(GPU_initialized=False, provider=name,
                  provider_maps_sha256=source.digest(OUTPUT / f"CPU-{name}-maps.json"),
                  driver_sha256=source.digest(OUTPUT / f"CPU-{name}-driver.json"))
    write(f"CPU-{name}-gate.json", record)


def binding():
    verify_raw()
    _, _, current = after.provenance()
    gates = {name: json.loads((OUTPUT / f"CPU-{name}-gate.json").read_text()) for name in ("SOURCE", "AFTER")}
    for name, gate in gates.items():
        assert (OUTPUT / f"CPU-{name}.exit").read_text().strip() == "0"
        assert gate["GPU_initialized"] is False
        for key, value in current.items():
            assert gate[key] == value
        assert source.digest(OUTPUT / f"CPU-{name}-maps.json") == gate["provider_maps_sha256"]
        assert source.digest(OUTPUT / f"CPU-{name}-driver.json") == gate["driver_sha256"]
    assert gates["SOURCE"]["driver_sha256"] == gates["AFTER"]["driver_sha256"]
    assert after.driver_proof() == json.loads((OUTPUT / "CPU-AFTER-driver.json").read_text())
    return dict(current, input_file_sha256=INPUT_SHA, oracle_file_sha256=ORACLE_SHA,
                CPU_data_gate_sha256=source.digest(OUTPUT / "CPU-data.json"),
                SOURCE_CPU_maps_sha256=gates["SOURCE"]["provider_maps_sha256"],
                AFTER_CPU_maps_sha256=gates["AFTER"]["provider_maps_sha256"],
                driver_sha256=gates["AFTER"]["driver_sha256"],
                script_sha256={name: source.digest(ROOT / name) for name in SCRIPTS},
                oracle_benchmark_sha256=source.digest(after.BENCH),
                plan_sha256=source.digest(ROOT / "SINGLE_CASE_FORENSIC_v2_PLAN.md"))


def grant():
    released = json.loads(BARRIER.read_text())
    assert released["phase"] == PHASE and released["gpu"] == 7
    assert released["issuer"] == "main coordinator user authorization"
    assert released["accuracy_only"] and released["no_timing_events_graphs_warmup"]
    assert released["case"] == "32768_weight_0_1" and released["invocations_per_provider"] == 1
    assert released["scope"] == "B8/ISL4096/OSL16/TP1/EP1"
    assert released["coordinator_release"] and not released["formal_perf_barrier_consumed"]
    assert released["tolerances"] == {"norm": 5e-3, "residual": 1e-5, "widen": 1e-5, "widen_exact": True}
    for key, value in binding().items():
        assert released[key] == value
    assert time.time() >= released["start_epoch"]


def driver_maps(name):
    stat = Path("/proc/self/stat").read_text()
    maps = Path("/proc/self/maps").read_text()
    (OUTPUT / f"{name}-live-runtime.maps").write_text(maps)
    libraries = {}
    for line in maps.splitlines():
        filename = line.split()[-1]
        if filename.startswith("/") and any(token in filename for token in ("libsycl", "libze_", "libigc", "libigdgmm", "libur_")):
            path = Path(filename).resolve()
            if path.is_file():
                libraries[str(path)] = source.digest(path)
    expected = json.loads((OUTPUT / "CPU-AFTER-driver.json").read_text())
    matched = {path: value for path, value in libraries.items() if path in expected["GPU_driver_candidate_libraries_sha256"]}
    write(f"{name}-live-runtime-driver.json", dict(pid=os.getpid(),
          starttick=int(stat[stat.rfind(")") + 2:].split()[19]), collected_inside_live_worker=True,
          mapped_libraries_sha256=libraries, CPU_driver_candidate_matches=matched,
          maps_sha256=source.digest(OUTPUT / f"{name}-live-runtime.maps")))
    assert any("libze_intel_gpu" in path for path in matched)
    assert any(value == expected["SYCL_runtime"]["sha256"] for path, value in libraries.items() if "libsycl" in path)
    assert all(value == expected["GPU_driver_candidate_libraries_sha256"][path] for path, value in matched.items())


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
                             full_global_masks=path.name, file_sha256=source.digest(path))
        del nan_mask, inf_mask
    write(f"{label}-finite.json", summary)


def run(name):
    grant()
    if name == "AFTER":
        assert (OUTPUT / "SOURCE.exit").read_text().strip() == "0"
        record = json.loads((OUTPUT / "SOURCE-actual.json").read_text())
        assert record["invocations"] == 1 and source.digest(OUTPUT / "SOURCE-actual.pt") == record["file_sha256"]
    torch, invoke, proof = provider(name)
    assert os.environ.get("ZE_AFFINITY_MASK") == "7"
    assert torch.xpu.is_available(), "XPU unavailable"
    from sgl_kernel.utils import is_xe3_arch
    assert is_xe3_arch(), "Only Xe35 capability 3.5"
    torch.xpu.init()
    write(f"{name}-normal-provider.json", proof)
    driver_maps(name)
    inputs = torch.load(DATA / "shared-new-inputs.pt", map_location="cpu", mmap=True, weights_only=True)
    input_meta = verify_raw()["records"]["new_inputs"]
    device_inputs = {}
    layout = {}
    for key, value in inputs.items():
        assert tensor_digest(torch, value) == input_meta[key]["sha256"]
        device_value = value.to("xpu")
        copied = device_value.cpu()
        assert tensor_digest(torch, copied) == input_meta[key]["sha256"]
        layout[key] = dict(dtype=str(device_value.dtype), shape=list(device_value.shape), stride=list(device_value.stride()),
                           storage_offset=device_value.storage_offset(), data_ptr=device_value.data_ptr(), full_copy_sha_verified=True)
        device_inputs[key] = device_value
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
        readonly[key] = tensor_digest(torch, current) == input_meta[key]["sha256"]
        if not readonly[key]:
            torch.save(current, OUTPUT / f"{name}-mutated-{key}.pt")
        del current
    write(f"{name}-input-readonly.json", readonly)
    write(f"{name}-invoke-state.json", dict(invocations_started=1, invocations_completed=1))
    torch.xpu.synchronize()
    time.sleep(2)
    assert all(readonly.values()), "Input mutation; actual data saved"


def compare():
    import torch
    assert not torch.xpu.is_initialized()
    verify_raw()
    for name in ("SOURCE", "AFTER"):
        assert (OUTPUT / f"{name}.exit").read_text().strip() == "0"
        record = json.loads((OUTPUT / f"{name}-actual.json").read_text())
        assert record["invocations"] == 1 and source.digest(OUTPUT / f"{name}-actual.pt") == record["file_sha256"]
    expected = torch.load(DATA / "full-original-CPU-oracle.pt", map_location="cpu", mmap=True, weights_only=True)
    outputs = {name: torch.load(OUTPUT / f"{name}-actual.pt", map_location="cpu", mmap=True, weights_only=True) for name in ("SOURCE", "AFTER")}
    checks = []
    for name, actual in outputs.items():
        checks.extend([(f"{name}-norm-oracle", actual["norm"], expected["norm"], 5e-3, False),
                       (f"{name}-residual-oracle", actual["stream"], expected["stream"], 1e-5, False),
                       (f"{name}-widen-norm", actual["widen"], actual["norm"], 1e-5, False),
                       (f"{name}-widen-exact", actual["widen"], actual["norm"], 0, True)])
    for key in ("norm", "stream", "widen"):
        checks.append((f"SOURCE-AFTER-{key}-bitwise", outputs["SOURCE"][key], outputs["AFTER"][key], 0, True))
    results = {}
    for label, left, right, tolerance, bitwise in checks:
        mask = torch.empty((32768, 8192), dtype=torch.bool)
        count = 0
        positions = []
        for start in range(0, 32768, 512):
            left_chunk, right_chunk = left[start:start + 512], right[start:start + 512]
            if bitwise:
                bits = torch.int16 if left.dtype == right.dtype == torch.bfloat16 else torch.int32
                mismatch = left_chunk.contiguous().view(bits) != (right_chunk if right.dtype == left.dtype else right_chunk.float()).contiguous().view(bits)
            else:
                mismatch = ~torch.isclose(left_chunk.float(), right_chunk.float(), rtol=tolerance, atol=tolerance, equal_nan=False)
            mask[start:start + 512] = mismatch
            count += mismatch.sum().item()
            if len(positions) < 32:
                found = torch.nonzero(mismatch, as_tuple=False)[:32 - len(positions)]
                positions.extend([[start + row, column] for row, column in found.tolist()])
        path = OUTPUT / f"{label}-mismatch-mask.pt"
        torch.save(mask, path)
        results[label] = dict(mismatching_elements=count, total_elements=32768 * 8192,
                             first_global_positions=positions, full_global_mask=path.name,
                             file_sha256=source.digest(path), bitwise=bitwise, rtol=tolerance, atol=tolerance)
        del mask
        write("full-comparison.json", results)
    passed = all(record["mismatching_elements"] == 0 for record in results.values())
    write("classification.json", dict(current_same_case_pass=passed, old_failure_actual_exit=1,
          original_failed_raw_replayed=False, old_failure_explained=False, precision_acceptance_24cases=False,
          performance_approved=False, oracle_recomputed=False, all_actual_outputs_saved_before_strict=True))
    assert not torch.xpu.is_initialized()
    if not passed:
        raise AssertionError("Same-case strict failure; all raw data saved")


def main():
    source.scope()
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--mode", required=True, choices=("CPU-data", "CPU-SOURCE", "CPU-AFTER", "binding", "SOURCE", "AFTER", "compare", "post"))
    args = parser.parse_args()
    assert set(os.sched_getaffinity(0)) <= set(range(48, 64))
    if args.mode == "CPU-data":
        cpu_data()
    elif args.mode.startswith("CPU-"):
        cpu(args.mode.removeprefix("CPU-"))
    elif args.mode in ("binding", "post"):
        write("binding.json" if args.mode == "binding" else "post-protected-source.json", binding())
    elif args.mode in ("SOURCE", "AFTER"):
        run(args.mode)
    else:
        compare()


if __name__ == "__main__":
    main()
