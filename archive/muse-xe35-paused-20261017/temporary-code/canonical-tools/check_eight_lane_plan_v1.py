import argparse
import hashlib
import json
from pathlib import Path


MODULES = frozenset({
    'bf16-gemm', 'fmha', 'qk-norm-rope', 'mxfp4-w4a4', 'residual-norm',
    'nvfp4-woq', 'moe-permutation', 'routing-quant', 'memory', 'position',
    'logits-vocab',
})
WORKLOAD = {'serving_batch': 8, 'isl': 4096, 'osl': 16, 'tp': 1, 'ep': 1,
            'production_precision_preserved': True}
ARTIFACTS = frozenset({'runner', 'normal_package_manifest', 'provider_cpu_gate',
                       'input_reference_manifest'})


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            checksum.update(block)
    return checksum.hexdigest()


def check_plan(plan, root):
    require(plan.get('workload') == WORKLOAD, 'Only the production B8 workload is eligible')
    lanes = plan.get('lanes', [])
    require(len(lanes) == 8, 'Exactly eight lanes are required')
    require(plan.get('compilation_cpu_ceiling') == 128, 'Compilation CPU ceiling must remain 128')
    require(plan.get('native_compilation_during_runtime') is False, 'No native compilation during the round')
    gpus, cpus, modules = set(), set(), set()
    verified = []
    for lane in lanes:
        gpu = lane.get('gpu')
        require(type(gpu) is int and 0 <= gpu < 12, 'GPU must be an allowed physical index')
        require(gpu not in gpus, 'Two lanes cannot use the same physical GPU')
        gpus.add(gpu)
        mode = lane.get('mode')
        require(mode in ('accuracy', 'performance'), 'Explicit accuracy/performance mode required')
        require(gpu < 8 or mode == 'accuracy', 'GPUs8–11 are accuracy-only')
        affinity = lane.get('runtime_cpus', [])
        require(bool(affinity), 'Runtime CPU affinity is required')
        require(all(type(core) is int and 0 <= core < 128 for core in affinity), 'CPU outside0–127')
        require(len(set(affinity)) == len(affinity), 'Duplicate core within one affinity')
        require(not cpus.intersection(affinity), 'Runtime lanes must not overlap CPU affinities')
        cpus.update(affinity)
        entries = lane.get('modules', [])
        require(bool(entries), 'Every lane must perform useful module validation')
        for entry in entries:
            module = entry.get('name')
            require(module in MODULES and module not in modules, 'Unknown or repeated module assignment')
            modules.add(module)
            artifacts = entry.get('artifacts', {})
            require(set(artifacts) == ARTIFACTS, 'Bind runner, normal package, CPU gate and inputs/references')
            for kind, record in artifacts.items():
                require(set(record) == {'path', 'sha256'}, 'Artifact needs a path and SHA256')
                path = Path(record['path'])
                path = path if path.is_absolute() else root / path
                require(path.is_file(), f'Missing {module} {kind}: {path}')
                require(digest(path) == record['sha256'], f'Changed {module} {kind}: {path}')
                verified.append({'module': module, 'kind': kind, 'path': str(path),
                                 'sha256': record['sha256']})
    require(modules == MODULES, 'All eleven Muse modules must remain in scope')
    return {'actual_plan_check_exit': 0, 'lane_count': 8, 'module_count': 11,
            'physical_gpus': sorted(gpus), 'runtime_cpu_count': len(cpus),
            'verified_artifacts': verified, 'GPU_work_started': False,
            'runtime_grant_issued': False, 'eight_card_overlap_proven': False,
            'precision_or_performance_accepted': False,
            'scope': 'Planning integrity only; not CPU-gate semantics, fresh drain/resource admission, or runtime proof'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Refuse to replace previous evidence')
    result = check_plan(json.loads(args.plan.read_text()), args.plan.resolve().parent)
    result['plan_sha256'] = digest(args.plan)
    with args.output.open('x') as output:
        output.write(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
