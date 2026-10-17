import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

import after_pre_wave_accuracy_v4 as after
import history24_accuracy as history
import single_case_forensic_v2 as helpers
import source_compiler_accuracy as source


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "normal-precision-pytest-v2"
BARRIER = ROOT / "gpu9-normal-precision-pytest-v2-grant.json"
PHASE = "residual-normal-precision-pytest-v2"
SCRIPTS = ("precision_pytest_v2.py", "precision_pytest_stage_v2.py", "precision_pytest_container_v2.sh", "run_precision_pytest_v2.py", "precision_maps_preflight_v2.py")


def write(name, value):
    (OUTPUT / name).write_text(json.dumps(value, indent=2) + "\n")


def test_path(role):
    audit = json.loads((ROOT / "RESIDUAL_HISTORY24_CPU_AUDIT_v1.json").read_text())
    record = audit[role + "_precision_tests"]
    path = Path(record["path"])
    if not path.exists() and str(path).startswith("/home/syk/"):
        path = Path("/root") / path.relative_to("/home/syk")
    assert source.digest(path) == record["sha256"]
    return path


def scope():
    source.scope()
    assert set(os.sched_getaffinity(0)) <= set(range(48, 64))
    assert os.environ.get("PYTEST_ADDOPTS", "") == ""
    assert os.environ.get("PYTEST_PLUGINS", "") == ""
    assert os.environ["LD_LIBRARY_PATH"].split(":")[0] == "/opt/gfx-deps/profiling_tools/lib"


def binding():
    _, _, record = after.provenance()
    final = json.loads((ROOT / "history24-accuracy-v1/final-audit.json").read_text())
    assert all(value == 0 for value in final["actual_phase_exits"].values())
    assert final["GPU7_released"] and not final["old_failure_explained"]
    record["history24_final_audit_sha256"] = source.digest(ROOT / "history24-accuracy-v1/final-audit.json")
    record["driver_proof"] = after.driver_proof()
    record["stack"] = history.stack()
    record["stack"].pop("loaded_xe")
    record["script_sha256"] = {name: source.digest(ROOT / name) for name in SCRIPTS}
    record["CPU_collection"] = {}
    normalized = {}
    for role, count in (("SOURCE", 69), ("AFTER", 73)):
        assert (OUTPUT / f"CPU-{role}.exit").read_text().strip() == "0"
        proof = json.loads((OUTPUT / f"CPU-{role}.json").read_text())
        assert proof["count"] == count and not proof["GPU_initialized"]
        assert proof["test_sha256"] == source.digest(test_path(role))
        assert proof["provenance"] == after.provenance()[2]
        assert proof["driver_proof"] == record["driver_proof"]
        record["CPU_collection"][role] = source.digest(OUTPUT / f"CPU-{role}.json")
        normalized[role] = [node.split("::")[-1].replace("_xe35", "") for node in proof["nodeids"]]
    assert len(set(normalized["SOURCE"])) == 69
    assert set(normalized["SOURCE"]) <= set(normalized["AFTER"])
    record["mechanical_suffix_all69_preserved"] = True
    record["counts"] = {"SOURCE": 69, "AFTER": 73}
    import precision_maps_preflight_v2 as metadata
    record["CPU_metadata_preflight"] = metadata.binding()
    return record


def grant():
    release = json.loads(BARRIER.read_text())
    assert release["phase"] == PHASE and release["issuer"] == "main coordinator user authorization"
    assert release["gpu"] == 9 and release["accuracy_only"] and release["coordinator_release"]
    assert release["no_timing_events_graphs_warmup"] and not release["performance_granted"]
    assert time.time() >= release["start_epoch"]
    for key, value in binding().items():
        assert release[key] == value
    return release


def resource(label):
    stat = os.statvfs(OUTPUT)
    available = stat.f_bavail * stat.f_frsize
    with (OUTPUT / "disk.jsonl").open("a") as logfile:
        logfile.write(json.dumps(dict(label=label, available_bytes=available, headroom_bytes=100 * 2**30)) + "\n")
    assert available >= 100 * 2**30


class PrecisionPlugin:
    def __init__(self, torch, role, collect):
        self.torch = torch
        self.role = role
        self.collect = collect
        self.nodeids = []
        self.results = []
        self.failed = False
        self.skipped = False

    def pytest_collection_finish(self, session):
        self.nodeids = [item.nodeid for item in session.items]
        expected = 69 if self.role == "SOURCE" else 73
        assert len(self.nodeids) == expected
        if not self.collect:
            original = json.loads((OUTPUT / f"CPU-{self.role}.json").read_text())
            assert self.nodeids == original["nodeids"]
        write(self.role + ("-CPU-nodeids.json" if self.collect else "-runtime-nodeids.json"), self.nodeids)

    def pytest_runtest_logreport(self, report):
        result = dict(nodeid=report.nodeid, when=report.when, outcome=report.outcome)
        self.results.append(result)
        with (OUTPUT / f"{self.role}-results.jsonl").open("a") as logfile:
            logfile.write(json.dumps(result) + "\n")
        self.skipped |= report.skipped
        self.failed |= report.failed

    def pytest_runtest_teardown(self, item, nextitem):
        if self.role == "SOURCE":
            self.torch.xpu.synchronize()
            gc.collect()
            self.torch.xpu.empty_cache()
            time.sleep(2)

    def pytest_runtest_setup(self, item):
        resource(self.role + "/" + item.nodeid)

    def pytest_exception_interact(self, node, call, report):
        if not report.failed:
            return
        torch = self.torch
        directory = OUTPUT / (self.role + "-first-failure")
        directory.mkdir(exist_ok=True)
        (directory / "exception.txt").write_text(str(call.excinfo.value) + "\n")
        records = []
        seen = set()

        def save(value, label, depth=0):
            if isinstance(value, torch.Tensor):
                if id(value) in seen:
                    return
                seen.add(id(value))
                resource("failure/" + label)
                tensor = value.detach().cpu()
                filename = f"tensor-{len(records):04d}.pt"
                torch.save(tensor, directory / filename)
                masks = dict(nan=torch.isnan(tensor), inf=torch.isinf(tensor))
                maskname = f"tensor-{len(records):04d}-finite.pt"
                torch.save(masks, directory / maskname)
                records.append(dict(label=label, original_device=str(value.device), filename=filename,
                    file_sha256=source.digest(directory / filename), dtype=str(tensor.dtype), shape=list(tensor.shape),
                    tensor_sha256=helpers.tensor_digest(torch, tensor), finite_masks=maskname,
                    mask_file_sha256=source.digest(directory / maskname), nan_count=masks["nan"].sum().item(),
                    inf_count=masks["inf"].sum().item()))
                del tensor, masks
            elif depth < 3 and isinstance(value, dict):
                for key, item in value.items():
                    save(item, label + "/" + str(key), depth + 1)
            elif depth < 3 and isinstance(value, (tuple, list)):
                for index, item in enumerate(value):
                    save(item, label + "/" + str(index), depth + 1)

        frames = []
        try:
            torch.xpu.synchronize()
            for index, entry in enumerate(call.excinfo.traceback):
                frame = entry.frame
                frames.append(dict(file=str(frame.code.path), function=frame.code.name, lineno=entry.lineno + 1))
                for name, value in frame.f_locals.items():
                    save(value, f"frame{index}/{name}")
        finally:
            (directory / "evidence.json").write_text(json.dumps(dict(nodeid=node.nodeid,
                full_tensor_records=records, frames=frames, tolerance_changed=False, original_failure_preserved=True), indent=2) + "\n")


def run(role, collect):
    scope()
    if not collect:
        grant()
    _, _, provenance = after.provenance()
    torch, _, maps = helpers.provider(role)
    assert not torch.xpu.is_initialized()
    assert torch.ones(4, device="cpu").sum().item() == 4
    if not collect:
        assert maps == json.loads((OUTPUT / f"CPU-{role}.json").read_text())["provider_maps"]
    else:
        toy = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        torch.save(dict(value=toy), OUTPUT / f"{role}-CPU-metadata-probe.pt")
        loaded = torch.load(OUTPUT / f"{role}-CPU-metadata-probe.pt", weights_only=True)
        metadata = helpers.metadata(torch, loaded)
        assert metadata["value"]["sha256"] == helpers.tensor_digest(torch, toy)
        write(role + "-CPU-metadata-probe.json", dict(metadata=metadata,
              file_sha256=source.digest(OUTPUT / f"{role}-CPU-metadata-probe.pt"), GPU_initialized=False))
    path = test_path(role)
    extra = ["-k", "xe35"] if role == "AFTER" else []
    sys.path.insert(0, str(path.parent))
    import pytest
    assert not torch.xpu.is_initialized()
    if collect:
        assert not torch.xpu.is_available()
    else:
        assert torch.xpu.is_available()
        from sgl_kernel.utils import is_xe3_arch
        assert is_xe3_arch()
        driver = {line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                  if "libze_intel_gpu" in line or "libsycl.so" in line}
        driver_proof = json.loads((OUTPUT / f"CPU-{role}.json").read_text())["driver_proof"]
        mapped = {str(Path(name).resolve()): source.digest(Path(name)) for name in driver}
        candidates = driver_proof["GPU_driver_candidate_libraries_sha256"]
        assert any("libze_intel_gpu" in name for name in mapped)
        assert all(value == candidates[name] for name, value in mapped.items() if "libze_intel_gpu" in name)
        assert any(value == driver_proof["SYCL_runtime"]["sha256"] for name, value in mapped.items() if "libsycl" in name)
        write(role + "-live-runtime.json", dict(pid=os.getpid(),
              starttick=Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19],
              libraries_sha256=mapped, stack=history.stack(),
              native_maps=maps, actual_provider=role, GPU=9, accuracy_only=True))
    plugin = PrecisionPlugin(torch, role, collect)
    argv = [str(path), "-q", "-x", "--maxfail=1", "--tb=short", "--junitxml=" + str(OUTPUT / f"{role}-{'collect' if collect else 'accuracy'}.xml")] + extra
    if collect:
        argv += ["--collect-only"]
    write(role + ("-CPU-pytest-command.json" if collect else "-pytest-command.json"), argv)
    status = int(pytest.main(argv, plugins=[plugin]))
    if collect:
        write(f"CPU-{role}.json", dict(count=len(plugin.nodeids), nodeids=plugin.nodeids,
              actual_pytest_collection_exit=status, GPU_initialized=torch.xpu.is_initialized(),
              test_sha256=source.digest(path), test_file=str(path), provider_maps=maps,
              driver_proof=after.driver_proof(), provenance=provenance,
              CPU_only_real_device_visibility=os.environ["ONEAPI_DEVICE_SELECTOR"],
              original_tests_unmodified=True, collection_not_accuracy_pass=True))
        assert not torch.xpu.is_initialized()
    else:
        passed = [item["nodeid"] for item in plugin.results if item["when"] == "call" and item["outcome"] == "passed"]
        write(role + "-terminal.json", dict(actual_pytest_exit=status, passed_count=len(passed),
              failed=plugin.failed, skipped=plugin.skipped, unchanged_test_sha256=source.digest(path),
              no_events_graphs_warmup_or_perf=True, original_tolerances_unchanged=True))
        if status == 0:
            assert len(passed) == len(plugin.nodeids) and not plugin.skipped and not plugin.failed
    raise SystemExit(status)


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--mode", choices=("CPU-SOURCE", "CPU-AFTER", "binding", "SOURCE", "AFTER", "post"), required=True)
    args = parser.parse_args()
    if args.mode == "binding":
        write("binding.json", binding())
    elif args.mode == "post":
        write("post-protected-source.json", after.provenance()[2])
    else:
        run(args.mode.removeprefix("CPU-"), args.mode.startswith("CPU-"))


if __name__ == "__main__":
    main()
