import argparse
import hashlib
import inspect
import json
import os
import sys
import time
from pathlib import Path


FOLDER = Path(__file__).resolve().parent
ROOT = FOLDER.parent
sys.path.insert(0, str(ROOT / 'official-v4-icpx-tools'))
from official_package_check import canonical_preflight, digest, runtime_provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('label', choices=['source', 'port'])
    args = parser.parse_args()
    if os.environ.get('ONEAPI_DEVICE_SELECTOR') != 'level_zero:cpu' or os.environ.get('BATCHES', '8') != '8':
        raise RuntimeError('CPU-only B8 preparation required before imports')
    if sorted(os.sched_getaffinity(0)) != list(range(96, 112)):
        raise RuntimeError('CPU96–111 required')
    manifest, manifest_file, after_site = canonical_preflight(ROOT, 'canonical-v4-icpx')
    site = after_site if args.label == 'port' else ROOT / 'fresh-source-icpx-v1/site/sgl_kernel'
    import torch
    import sgl_kernel

    assert not torch.xpu.is_available()
    assert Path(sgl_kernel.__file__).resolve() == site / '__init__.py'
    mapped = runtime_provenance(ROOT, args.label, site, manifest, manifest_file)
    operation = 'sgl_kernel::mxfp4_small_permutation' + ('_xe35' if args.label == 'port' else '')
    assert torch._C._dispatch_has_kernel_for_dispatch_key(operation, 'XPU')
    api = inspect.getsource(torch.xpu.XPUGraph)
    methods = ['capture_begin', 'capture_end', 'instantiate', 'enable_debug_mode', 'debug_dump', 'replay', 'reset']
    assert all(hasattr(torch.xpu.XPUGraph, name) for name in methods)
    assert 'keep_graph' in inspect.signature(torch.xpu.XPUGraph.__new__).parameters
    header = Path(torch.__file__).resolve().parent / 'include/ATen/xpu/XPUGraph.h'
    graph_header = Path('/opt/gfx-deps/dpcpp/include/sycl/ext/oneapi/experimental/graph/modifiable_graph.hpp')
    assert 'std::vector<node> get_nodes() const' in graph_header.read_text()
    assert 'void print_graph(' in graph_header.read_text()
    record = {'epoch': time.time(), 'label': args.label, 'normal_package': sgl_kernel.__file__,
              'actual_normal_provider_maps_verified': True, 'mapped_DSO_count': len(mapped['loaded_libraries']),
              'actual_CPU_affinity': sorted(os.sched_getaffinity(0)), 'torch_version': torch.__version__,
              'actual_registered_XPU_op': operation, 'real_XPUGraph_class_API_verified': methods,
              'real_XPUGraph_source': api, 'real_XPUGraph_source_sha256': hashlib.sha256(api.encode()).hexdigest(),
              'installed_XPUGraph_header_sha256': digest(header), 'installed_SYCL_graph_header_sha256': digest(graph_header),
              'source_digest': manifest['checker_proofs']['source_digest'], 'AFTER_manifest_sha256': digest(manifest_file),
              'SOURCE_precision_and_compiler_proofs': manifest['checker_proofs']['tested_SOURCE'],
              'GPU_started': False, 'XPUGraph_instantiated_or_captured': False,
              'actual_native_1000_nodes_verified': False,
              'future_runtime_requirement': 'Real 1000-node DOT preflight before any timing under a new main GPU4 grant'}
    record.update(mapped)
    (FOLDER / ('CPU-normal-' + args.label + '.json')).write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'label': args.label, 'actual_normal_maps': len(mapped['loaded_libraries']),
                      'actual_XPUGraph_API_verified_CPU_only': True, 'actual_1000_node_capture': False}), flush=True)


if __name__ == '__main__':
    main()
