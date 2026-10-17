import hashlib
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "single-case-forensic-v1"


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def tensor_digest(tensor):
    result = hashlib.sha256()
    for start in range(0, tensor.shape[0], 512):
        result.update(tensor[start:start + 512].contiguous().view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def main():
    assert not torch.xpu.is_initialized()
    assert int((OUTPUT / "supervisor.exit").read_text()) == 1
    smoke = []
    for dtype in (torch.bfloat16, torch.float32):
        for shape in ((8192,), (2, 8192)):
            sample = torch.arange(8192, dtype=torch.float32).to(dtype)
            if len(shape) == 2:
                sample = sample.repeat(2, 1)
            assert tensor_digest(sample) == hashlib.sha256(sample.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
            smoke.append(dict(dtype=str(dtype), shape=list(shape), digest_smoke_actual0=True))
    result = dict(scope="CPU-only post-failure audit; no GPU retry", actual_v1_exit=1,
                  rootcause="new runner reused module-global torch digest without initializing that module global",
                  original_AFTER_NaN_failure_explained=False,
                  SOURCE_kernel_invocations=0, AFTER_kernel_invocations=0,
                  current_kernel_accuracy_measured=False, local_digest_CPU_smoke=smoke, files={})
    for label, filename in (("new_inputs", "shared-new-inputs.pt"), ("original_CPU_oracle", "full-original-CPU-oracle.pt")):
        path = OUTPUT / filename
        tensors = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        records = {}
        for name, tensor in tensors.items():
            nan_mask = torch.empty(tensor.shape, dtype=torch.bool)
            inf_mask = torch.empty(tensor.shape, dtype=torch.bool)
            for start in range(0, tensor.shape[0], 512):
                nan_mask[start:start + 512] = torch.isnan(tensor[start:start + 512])
                inf_mask[start:start + 512] = torch.isinf(tensor[start:start + 512])
            masks_path = OUTPUT / f"CPU-supplement-{label}-{name}-finite.pt"
            torch.save(dict(nan=nan_mask, inf=inf_mask), masks_path)
            records[name] = dict(dtype=str(tensor.dtype), shape=list(tensor.shape), stride=list(tensor.stride()),
                                 elements=tensor.numel(), sha256=tensor_digest(tensor),
                                 nan_count=nan_mask.sum().item(), inf_count=inf_mask.sum().item(),
                                 full_global_position_masks=masks_path.name, mask_sha256=digest(masks_path))
            del nan_mask, inf_mask
        result["files"][label] = dict(path=filename, file_sha256=digest(path), tensors=records)
        del tensors
    assert not (OUTPUT / "SOURCE-invoke-state.json").exists()
    assert not (OUTPUT / "AFTER-invoke-state.json").exists()
    assert not (OUTPUT / "SOURCE-actual.pt").exists()
    assert not (OUTPUT / "AFTER-actual.pt").exists()
    assert not torch.xpu.is_initialized()
    result["GPU_initialized"] = False
    (OUTPUT / "CPU-harness-failure-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(CPU_audit_actual0=True, kernel_invocations=0, GPU_initialized=False)), flush=True)


if __name__ == "__main__":
    main()
