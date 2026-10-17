import argparse
import json
from pathlib import Path

from canonical_tools_loader_v1 import digest_file


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("component", choices=("memory", "position", "logits-vocab"))
    parser.add_argument("label", choices=("before", "after"))
    args = parser.parse_args()
    artifact = ROOT / args.component
    official_root = artifact / "canonical-v4-icpx"
    official = json.loads((official_root / "source-binary-manifest.json").read_text())
    assert official["build_evidence"]["exit"] == 0 and official["install_actual_exit"] == 0
    if args.label == "after":
        stage = official_root / "site"
        expected = official["installed_files"] | official["python_snapshot_files"]
    else:
        comparator = json.loads((artifact / "canonical-validation-v4-icpx/source-comparator-manifest.json").read_text())
        assert digest_file(Path(comparator["source_native_manifest"])) == comparator["source_native_manifest_sha256"]
        assert digest_file(official_root / "source-binary-manifest.json") == comparator["after_manifest_sha256"]
        stage = Path(comparator["stage"])
        expected = comparator["files"]
    import sgl_kernel
    import torch

    assert Path(torch.__file__).resolve().is_relative_to(Path("/opt/gfx-deps/venv"))
    assert Path(sgl_kernel.__file__).resolve() == stage / "sgl_kernel/__init__.py"
    assert digest_file(Path(sgl_kernel.__file__)) == expected["sgl_kernel/__init__.py"]["sha256"]
    names = {"memory": ("paged_alloc_extend", "paged_alloc_extend_and_write", "paged_alloc_decode", "write_req_to_token_pool"), "position": ("compute_position", "compute_position_with_hint"), "logits-vocab": ("softcap_logits", "vocab_mask", "vocab_parallel_embedding")}[args.component]
    schemas = {name: str(getattr(torch.ops.sgl_kernel, name + "_xe35").default._schema) for name in names}
    mapped = {line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines() if "libsgl-ops-sycl-" in line or "/common_ops" in line}
    actual = {}
    for filename in mapped:
        path = Path(filename).resolve()
        relative = path.relative_to(stage).as_posix()
        assert digest_file(path) == expected[relative]["sha256"]
        actual[relative] = digest_file(path)
    required = {official["common_ops"]} | {record["path"] for name, record in official["binaries"].items() if name in official["common_ops_dt_needed"]}
    assert required.issubset(actual)
    assert not torch.xpu.is_initialized()
    print(json.dumps({"component": args.component, "label": args.label, "package": sgl_kernel.__file__, "torch": torch.__file__, "schemas": schemas, "mapped_native": actual, "GPU_initialized": False, "normal_current_registration": True, "actual_cpu_preflight": "PASS"}))


if __name__ == "__main__":
    main()
