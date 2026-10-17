import hashlib
import json
from pathlib import Path

art = Path(__file__).resolve().parent
root = art.parent.parent / "pr/muse-fmha-xe35"
source = Path("/root/worktrees/sgl-kernel-xpu/review-validation-artifacts/fmha-before-after-20261017/site-after/sgl_kernel")
site = art / "site-port-diagnostic/sgl_kernel"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


replaced = "libsgl-ops-sycl-flash_attentionXe35.so"
unchanged = {}
for path in source.rglob("*"):
    if path.is_file() and "__pycache__" not in path.parts and path.name != replaced:
        relative = str(path.relative_to(source))
        assert digest(path) == digest(site / relative), relative
        unchanged[relative] = digest(path)
providers = {}
for kind in ("decode", "prefill"):
    name = f"libsgl-ops-sycl-fmha_{kind}_64_bf16.so"
    fresh = root / "build-xe35/src" / name
    assert digest(fresh) == digest(site / name), name
    providers[name] = digest(fresh)
report = {"acceptance": "diagnostic_not_official_full_package_pass", "source_package": str(source),
          "normal_package": str(site), "unchanged_source_package_files": unchanged,
          "newly_built_actual_port_providers": providers, "bridge_sha256": digest(site / replaced),
          "replaced_source_root_provider_sha256": digest(source / replaced),
          "public_cutlass_pin": "7b56b936b87f799f80dde60c2fd8d11c1128cebb",
          "common_ops_sha256": digest(site / "common_ops.abi3.so"),
          "bridge_source_sha256": digest(art / "diagnostic/xe35/fmha/abi_forward.cpp"),
          "no_fake_registration_or_load_library": True}
(art / "diagnostic-provenance.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({key: value for key, value in report.items() if key != "unchanged_source_package_files"}, indent=2))
