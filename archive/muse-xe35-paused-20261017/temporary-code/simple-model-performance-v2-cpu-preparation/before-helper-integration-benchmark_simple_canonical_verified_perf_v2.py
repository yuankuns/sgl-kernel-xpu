import argparse
import json
import math
import os
import time
from pathlib import Path

from canonical_tools_loader_v1 import digest_file


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('component', choices=('memory', 'position', 'logits-vocab'))
    parser.add_argument('--label', choices=('before', 'after'), required=True)
    parser.add_argument('--phase', choices=('performance',), required=True)
    parser.add_argument('--serving-batch', type=int, choices=(8,), default=8)
    parser.add_argument('--samples', type=int, default=101)
    parser.add_argument('--calls', type=int, default=128)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--grant', type=Path, required=True)
    args = parser.parse_args()
    for name in ('BATCHES', 'BATCH', 'SERVING_BATCH', 'BATCH_SIZE', 'SGLANG_BATCH_SIZE', 'SGLANG_MUSE_BATCH_SIZE', 'MUSE_BATCH_SIZE', 'SERVING_BATCH_SIZE', 'MUSE_BS', 'BS'):
        if os.environ.get(name, '8') != '8':
            parser.error(f'B8-only: {name} must be 8')
    for name, expected in {'ISL': '4096', 'OSL': '16', 'TP': '1', 'EP': '1'}.items():
        if os.environ.get(name, expected) != expected:
            parser.error(f'Production scope requires {name}={expected}')
    gpu = os.environ.get('ZE_AFFINITY_MASK')
    if args.phase == 'performance' and gpu not in {'4', '5', '6', '7'}:
        parser.error('Performance requires stable GPU4–7; rejected before Torch import')
    if gpu not in {'4', '5', '6', '7', '8', '9', '10', '11'}:
        parser.error('Coordinator GPU allocation required')
    if args.samples <= 0 or args.calls <= 0:
        parser.error('samples and calls must be positive')
    args.authorization = json.loads(args.grant.read_text())
    expected = {'scope': 'simple-representative-verified-performance-v2', 'module': args.component,
                'phase': 'performance', 'gpu': 7, 'coordinator_release': True,
                'serving_batch': 8, 'isl': 4096, 'osl': 16, 'tp': 1, 'ep': 1}
    for name, value in expected.items():
        if type(args.authorization.get(name)) is not type(value) or args.authorization[name] != value:
            parser.error('Wrong coordinator grant: ' + name)
    if gpu != '7' or args.samples != 101 or args.calls != 128:
        parser.error('Main GPU7 and fixed 101 samples/128 calls required')
    for key, filename in (('driver_sha256', Path(__file__).name),
                          ('graph_helper_sha256', 'simple_verified_graph_v1.py'),
                          ('case_helper_sha256', 'simple_representative_performance_cases_v1.py')):
        if args.authorization.get(key) != digest_file(Path(__file__).with_name(filename)):
            parser.error('Changed frozen performance dependency: ' + key)
    if args.output.exists():
        parser.error('Refusing to overwrite evidence')
    deadline = args.authorization['start_epoch']
    if not math.isfinite(deadline) or deadline <= 0:
        parser.error('Invalid future grant epoch')
    time.sleep(max(0, deadline - time.time()))
    args.api = 'xe35'
    return args


def main():
    args = arguments()
    from benchmark_simple import native_operations
    from simple_representative_performance_cases_v1 import logits_cases, memory_cases, position_cases
    from simple_verified_graph_v1 import time_graph
    import sgl_kernel
    import torch

    if not torch.xpu.is_available() or not sgl_kernel.is_xe3_arch():
        raise SystemExit('Xe35 compute capability 3.5 required')
    torch.manual_seed(42)
    cases = {'memory': memory_cases, 'position': position_cases, 'logits-vocab': logits_cases}[args.component]
    operations = native_operations(torch, args)
    results = []
    for index, (name, launch, check, reset) in enumerate(cases(torch, operations)):
        if index:
            torch.xpu.synchronize()
            time.sleep(2)
        if reset:
            reset()
        launch()
        torch.xpu.synchronize()
        check()
        result = {'case': name, 'accuracy': 'PASS'}
        if args.phase == 'performance':
            result.update(time_graph(torch, launch, args, reset))
        results.append(result)
        print(json.dumps(result), flush=True)
    torch.xpu.synchronize()
    names = {'memory': ('PagedAlloc', 'WriteReqToTokenPool'), 'position': ('ComputePosition',), 'logits-vocab': ('SoftcapLogits', 'VocabMask')}[args.component]
    mapped = {line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if any('libsgl-ops-sycl-' + name + '.so' in line for name in names)}
    if len(mapped) != len(names):
        raise RuntimeError(f'Missing actual tested native libraries: {mapped}')
    report = {
        'component': args.component, 'label': args.label, 'phase': args.phase, 'api': args.api,
        'registration': 'normal current official package initialization and real _xe35 registered operators',
        'serving_batch': 8, 'isl': 4096, 'osl': 16, 'tp': 1, 'ep': 1,
        'gpu_affinity': os.environ['ZE_AFFINITY_MASK'], 'device': torch.xpu.get_device_name(0),
        'torch': torch.__version__, 'package': str(Path(sgl_kernel.__file__).resolve()),
        'samples': args.samples if args.phase == 'performance' else 0,
        'calls': args.calls if args.phase == 'performance' else 0,
        'timing': 'XPU graph events; per graph node; input reset excluded' if args.phase == 'performance' else 'none: accuracy-only',
        'libraries': {library: digest_file(Path(library)) for library in sorted(mapped)}, 'results': results,
        'driver_sha256': digest_file(Path(__file__)), 'reference_driver_sha256': digest_file(Path(__file__).with_name('benchmark_simple.py')),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise RuntimeError('Refusing to overwrite evidence')
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
