import json
import os
import time
from pathlib import Path
from checks import ROOT, ROUND, digest, package_gate, runtime_provenance, write_json


def main():
    if os.environ.get('ONEAPI_DEVICE_SELECTOR') != 'level_zero:cpu' or os.sched_getaffinity(0) != set(range(16)):
        raise RuntimeError('CPU0–15/CPU-only provider required')
    manifest, manifest_file, site, protected = package_gate()
    import torch
    import sgl_kernel
    source_site = ROOT / 'fresh-source-icpx-v1/site/sgl_kernel'
    assert Path(sgl_kernel.__file__).resolve() == source_site / '__init__.py'
    mapped = runtime_provenance(ROOT, 'source', source_site, manifest, manifest_file)
    assert len(mapped['loaded_libraries']) == 156
    assert not torch.xpu.is_available()
    source_manifest = ROOT / 'fresh-source-icpx-v1/source-native-binary-manifest.json'
    assert digest(source_manifest) == '8db500c8958c2afa15aedb50b06b8e6f3dcc3ec3b94eec3dd4f58f2422cdd112'
    write_json(ROUND / 'CPU_SOURCE_PROVIDER.json', {'epoch': time.time(), 'actual_exit': 0, 'affinity': sorted(os.sched_getaffinity(0)), 'SOURCE_manifest_sha256': digest(source_manifest), 'mapped': mapped, 'GPU_started': False})
    print(json.dumps({'CPU_SOURCE_provider_exit': 0, 'mapped_native_count': 156, 'GPU_started': False}), flush=True)


if __name__ == '__main__':
    main()
