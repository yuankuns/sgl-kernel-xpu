import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('component', choices=('memory', 'position'))
    parser.add_argument('--label', choices=('before', 'after'), required=True)
    parser.add_argument('--phase', choices=('performance',), default='performance')
    parser.add_argument('--samples', type=int, default=101)
    parser.add_argument('--calls', type=int, default=128)
    parser.add_argument('--grant', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    for name in ('BATCHES', 'BATCH', 'SERVING_BATCH', 'BATCH_SIZE', 'SGLANG_BATCH_SIZE', 'SGLANG_MUSE_BATCH_SIZE', 'MUSE_BATCH_SIZE', 'SERVING_BATCH_SIZE', 'MUSE_BS', 'BS'):
        if os.environ.get(name, '8') != '8':
            parser.error(f'B8-only: {name}')
    for name, expected in {'ISL': '4096', 'OSL': '16', 'TP': '1', 'EP': '1'}.items():
        if os.environ.get(name, expected) != expected:
            parser.error(f'Wrong production scope: {name}')
    gpu = 11 if args.phase == 'accuracy' else 7
    if os.environ.get('ZE_AFFINITY_MASK') != str(gpu):
        parser.error(f'Accuracy11/performance7 main allocation required: {gpu}')
    args.authorization = json.loads(args.grant.read_text())
    required = {'scope': 'memory-position-model-performance-v2', 'module': args.component,
                'phase': args.phase, 'gpu': gpu, 'coordinator_release': True,
                'accuracy_authorized': args.phase == 'accuracy', 'performance_authorized': args.phase == 'performance',
                'serving_batch': 8, 'isl': 4096, 'osl': 16, 'tp': 1, 'ep': 1}
    for key, expected in required.items():
        if type(args.authorization.get(key)) is not type(expected) or args.authorization[key] != expected:
            parser.error(f'Wrong coordinator grant: {key}')
    if args.authorization['driver_sha256'] != digest(Path(__file__)):
        parser.error('Changed frozen driver')
    if args.samples != 101 or args.calls != 128:
        parser.error('Fixed comparable device-event protocol required')
    if args.output.exists() or args.output.with_suffix('.pt').exists():
        parser.error('Refusing to overwrite evidence')
    if args.authorization.get('graph_helper_sha256') != digest(ROOT / 'simple_verified_graph_v1.py'):
        parser.error('Changed verified graph helper')
    deadline = args.authorization['start_epoch']
    if not math.isfinite(deadline) or deadline <= 0:
        parser.error('Invalid future grant epoch')
    time.sleep(max(0, deadline - time.time()))
    return args


def main():
    args = arguments()
    provenance = json.loads((ROOT / 'SIMPLE_MODEL_CONFIG_PROVENANCE_v1.json').read_text())
    config = Path(provenance['model_config'].replace('/home/syk/', '/root/', 1))
    assert digest(config) == provenance['model_config_sha256'] == args.authorization['model_config_sha256']
    official = ROOT / args.component / 'canonical-v4-icpx'
    comparator_path = ROOT / args.component / 'canonical-validation-v4-icpx/source-comparator-manifest.json'
    assert digest(official / 'source-binary-manifest.json') == args.authorization['package_manifest_sha256']
    assert digest(comparator_path) == args.authorization['source_comparator_manifest_sha256']
    comparator = json.loads(comparator_path.read_text())
    assert digest(Path(comparator['source_native_manifest'])) == args.authorization['source_native_manifest_sha256']
    stage = official / 'site' if args.label == 'after' else Path(comparator['stage'])
    import sgl_kernel
    import torch

    assert Path(sgl_kernel.__file__).resolve() == stage / 'sgl_kernel/__init__.py'
    assert torch.xpu.is_available() and sgl_kernel.is_xe3_arch()
    manifest = json.loads((official / 'source-binary-manifest.json').read_text())
    expected_files = manifest['installed_files'] | manifest['python_snapshot_files'] if args.label == 'after' else comparator['files']
    mapped = {line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
              if '/common_ops' in line or 'libsgl-ops-sycl-' in line}
    mapped_sha = {}
    for filename in mapped:
        path = Path(filename).resolve()
        relative = path.relative_to(stage).as_posix()
        assert digest(path) == expected_files[relative]['sha256']
        mapped_sha[relative] = digest(path)
    names = ('PagedAlloc', 'WriteReqToTokenPool') if args.component == 'memory' else ('ComputePosition',)
    assert all(f'sgl_kernel/libsgl-ops-sycl-{name}.so' in mapped_sha for name in names)
    tensors, records = {}, []
    operators = torch.ops.sgl_kernel

    def capture(name, launch, expected, inputs):
        if records:
            torch.xpu.synchronize()
            time.sleep(2)
        outputs = launch()
        torch.xpu.synchronize()
        actual = {key: value.cpu() for key, value in outputs.items()}
        for key in actual:
            torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
        tensors[name] = {'inputs': inputs, 'outputs': actual}
        record = {'case': name, 'accuracy': 'PASS',
                  'full_exact_reference_elements': sum(value.numel() for value in actual.values())}
        if args.phase == 'performance':
            from simple_verified_graph_v1 import time_graph
            record.update(time_graph(torch, launch, args))
        records.append(record)
        print(json.dumps(records[-1]), flush=True)

    if args.component == 'memory':
        for active, first_slot in ((3, 0), (5, 3)):
            prefixes_cpu = torch.zeros(active, dtype=torch.int64)
            lengths_cpu = torch.full((active,), 4096, dtype=torch.int64)
            pages_cpu = torch.arange(1024 + first_slot * 64, 1024 + (first_slot + active) * 64, dtype=torch.int64)
            requests_cpu = torch.arange(first_slot, first_slot + active, dtype=torch.int64)
            prefixes, lengths, pages, requests = (value.to('xpu') for value in (prefixes_cpu, lengths_cpu, pages_cpu, requests_cpu))
            last_cpu = torch.full((active,), -1, dtype=torch.int64)
            pointers_cpu = torch.zeros(active, dtype=torch.uint64)
            last, pointers = last_cpu.to('xpu'), pointers_cpu.to('xpu')
            out = torch.empty(active * 4096, dtype=torch.int64, device='xpu')
            pool = torch.full((8, 4112), -1, dtype=torch.int32, device='xpu')
            expected_out = torch.arange(int(pages_cpu[0]) * 64, int(pages_cpu[0]) * 64 + active * 4096, dtype=torch.int64)
            expected_pool = torch.full((8, 4112), -1, dtype=torch.int32)
            for request in range(active):
                expected_pool[first_slot + request, :4096] = expected_out[request * 4096:(request + 1) * 4096].int()
            inputs = {'prefixes': prefixes_cpu, 'lengths': lengths_cpu, 'pages': pages_cpu,
                      'requests': requests_cpu, 'last': last_cpu, 'pointers': pointers_cpu}
            def extend():
                operators.paged_alloc_extend_xe35(prefixes, lengths, last, pages, out, 64)
                return {'locations': out}
            capture(f'extend-wave{active}-tokens{active * 4096}', extend, {'locations': expected_out}, inputs)
            def write():
                operators.write_req_to_token_pool_xe35(pool, requests, pointers, prefixes, lengths, lengths, out)
                return {'pool': pool}
            capture(f'write-wave{active}-tokens{active * 4096}', write, {'pool': expected_pool}, inputs)
            pool.fill_(-1)
            def fused():
                operators.paged_alloc_extend_and_write_xe35(prefixes, lengths, last, pages, out, pool, requests, pointers, 64)
                return {'locations': out, 'pool': pool}
            capture(f'fused-wave{active}-tokens{active * 4096}', fused, {'locations': expected_out, 'pool': expected_pool}, inputs)
        pages_cpu = torch.arange(2048, 2056, dtype=torch.int64)
        for step in range(1, 16):
            seq_cpu = torch.full((8,), 4096 + step, dtype=torch.int64)
            last_cpu = torch.arange(8, dtype=torch.int64) * 4096 + 1024 * 64 + 4095 if step == 1 else pages_cpu * 64 + step - 2
            seq, last, pages = (value.to('xpu') for value in (seq_cpu, last_cpu, pages_cpu))
            out = torch.empty(8, dtype=torch.int64, device='xpu')
            def decode():
                operators.paged_alloc_decode_xe35(seq, last, pages, out, 64)
                return {'locations': out}
            capture(f'decode-output{step}-cache{4096 + step}', decode,
                    {'locations': pages_cpu * 64 + step - 1}, {'seq': seq_cpu, 'last': last_cpu, 'pages': pages_cpu})
    else:
        cases = [(f'prefill-wave{active}', active, 4096, 0) for active in (3, 5)]
        cases += [(f'decode-output{step}', 8, 1, 4095 + step) for step in range(1, 16)]
        for name, active, length, prefix in cases:
            lengths_cpu = torch.full((active,), length, dtype=torch.int32)
            prefix_cpu = torch.full((active,), prefix, dtype=torch.int32)
            lengths, prefixes = lengths_cpu.to('xpu'), prefix_cpu.to('xpu')
            positions = torch.empty(active * length, dtype=torch.int64, device='xpu')
            starts = torch.empty(active, dtype=torch.int32, device='xpu')
            expected = {'positions': torch.arange(length, dtype=torch.int64).repeat(active) + prefix,
                        'starts': torch.arange(active, dtype=torch.int32) * length}
            def position():
                operators.compute_position_xe35(positions, starts, prefixes, lengths)
                return {'positions': positions, 'starts': starts}
            capture(name, position, expected, {'lengths': lengths_cpu, 'prefix': prefix_cpu})
            def hinted():
                operators.compute_position_with_hint_xe35(positions, starts, prefixes, lengths, length)
                return {'positions': positions, 'starts': starts}
            capture(name + '-hint', hinted, expected, {'lengths': lengths_cpu, 'prefix': prefix_cpu})
    torch.save(tensors, args.output.with_suffix('.pt'))
    report = {'component': args.component, 'label': args.label, 'phase': args.phase, 'gpu': os.environ['ZE_AFFINITY_MASK'],
              'serving_batch': 8, 'isl': 4096, 'osl': 16, 'tp': 1, 'ep': 1,
              'driver_sha256': digest(Path(__file__)), 'model_config_sha256': digest(config),
              'mapped_native_sha256': mapped_sha, 'results': records,
              'timing': 'none' if args.phase == 'accuracy' else 'device graph events, reset/setup excluded',
              'samples': 0 if args.phase == 'accuracy' else args.samples,
              'calls': 0 if args.phase == 'accuracy' else args.calls,
              'page_size': 64, 'synthetic_pool_shape': [8, 4112],
              'scope': 'config/call-shape matching synthetic operator contracts; B8 waves active3/5 and15decode forwards; not allocator/raw/framework replay'}
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
