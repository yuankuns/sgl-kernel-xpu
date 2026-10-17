import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('component', choices=('memory', 'position', 'logits-vocab'))
    parser.add_argument('--label', choices=('before', 'after'), required=True)
    parser.add_argument('--api', choices=('xe35', 'source-native'), default='xe35')
    parser.add_argument('--serving-batch', type=int, choices=(8,), default=8)
    parser.add_argument('--samples', type=int, default=31)
    parser.add_argument('--calls', type=int, default=32)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if any(os.environ.get(name, '8') != '8' for name in ('BATCHES', 'BATCH_SIZE', 'SERVING_BATCH')):
        parser.error('Muse-Spark serving scope requires B8/ISL4096/OSL16/TP1/EP1')
    if args.samples <= 0 or args.calls <= 0:
        parser.error('samples and calls must be positive')
    return args


def time_graph(torch, function, args, reset=None):
    for _ in range(5):
        if reset:
            reset()
        function()
    torch.xpu.synchronize()
    if reset:
        reset()
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        for _ in range(args.calls):
            function()
    for _ in range(5):
        if reset:
            reset()
        graph.replay()
    torch.xpu.synchronize()
    samples = []
    for _ in range(args.samples):
        if reset:
            reset()
        torch.xpu.synchronize()
        start = torch.xpu.Event(enable_timing=True)
        end = torch.xpu.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / args.calls)
    median = statistics.median(samples)
    return {
        'p50_us': median,
        'mad_us': statistics.median(abs(sample - median) for sample in samples),
        'raw_us': samples,
    }


def native_operations(torch, args):
    names = {
        'memory': {
            'paged_alloc_extend': 'paged_alloc_extend',
            'paged_alloc_extend_and_write': 'paged_alloc_extend_and_write',
            'paged_alloc_decode': 'paged_alloc_decode',
            'write_req_to_token_pool': 'write_req_to_token_pool',
        },
        'position': {
            'compute_position': 'muse_compute_position',
            'compute_position_with_hint': 'muse_compute_position_with_hint',
        },
        'logits-vocab': {
            'softcap_logits': 'softcap_logits',
            'vocab_parallel_embedding': 'muse_vocab_parallel_embedding',
            'vocab_mask': 'vocab_mask',
        },
    }[args.component]
    return {
        name: getattr(torch.ops.sgl_kernel, source_name if args.api == 'source-native' else name + '_xe35')
        for name, source_name in names.items()
    }


def memory_cases(torch, ops):
    dtype = torch.int64
    device = 'xpu'
    prefix = torch.zeros(8, dtype=dtype, device=device)
    lengths = torch.full((8,), 4096, dtype=dtype, device=device)
    last = torch.full((8,), -1, dtype=dtype, device=device)
    free = torch.arange(1024, 1024 + 512, dtype=dtype, device=device)
    out = torch.empty(32768, dtype=dtype, device=device)
    pool = torch.full((8, 4112), -1, dtype=torch.int32, device=device)
    requests = torch.arange(7, -1, -1, dtype=dtype, device=device)
    pointers = torch.zeros(8, dtype=torch.uint64, device=device)
    expected = torch.arange(1024 * 64, 1024 * 64 + 32768, dtype=dtype)

    def check_out():
        torch.testing.assert_close(out.cpu(), expected, atol=0, rtol=0)

    def check_pool():
        expected_pool = torch.full((8, 4112), -1, dtype=torch.int32)
        for request in range(8):
            expected_pool[7 - request, :4096] = expected[request * 4096:(request + 1) * 4096].to(torch.int32)
        torch.testing.assert_close(pool.cpu(), expected_pool, atol=0, rtol=0)

    extend = lambda: ops['paged_alloc_extend'](prefix, lengths, last, free, out, 64)
    extend()
    check_out()
    yield 'alloc-extend-prefill-b8', extend, check_out, None
    write = lambda: ops['write_req_to_token_pool'](pool, requests, pointers, prefix, lengths, lengths, out)
    yield 'write-prefill-b8', write, check_pool, None
    fused = lambda: ops['paged_alloc_extend_and_write'](prefix, lengths, last, free, out, pool, requests, pointers, 64)

    def check_fused():
        check_out()
        check_pool()

    yield 'alloc-write-prefill-b8', fused, check_fused, None
    free_decode = torch.arange(2048, 2056, dtype=dtype, device=device)
    decode_out = torch.empty(8, dtype=dtype, device=device)
    last_decode = torch.arange(4095, 4095 + 8 * 4096, 4096, dtype=dtype, device=device)
    for step in (1, 16):
        seq = torch.full((8,), 4096 + step, dtype=dtype, device=device)
        previous = last_decode if step == 1 else free_decode * 64 + step - 2
        expected_decode = free_decode.cpu() * 64 if step == 1 else previous.cpu() + 1
        launch = lambda seq=seq, previous=previous: ops['paged_alloc_decode'](seq, previous, free_decode, decode_out, 64)
        check = lambda expected_decode=expected_decode: torch.testing.assert_close(decode_out.cpu(), expected_decode, atol=0, rtol=0)
        yield f'alloc-decode-osl{step}-b8', launch, check, None


def position_cases(torch, ops):
    for length in (1, 4096):
        lengths = torch.full((8,), length, dtype=torch.int32, device='xpu')
        prefix = torch.full((8,), 4096 if length == 1 else 0, dtype=torch.int32, device='xpu')
        positions = torch.empty(8 * length, dtype=torch.int64, device='xpu')
        starts = torch.empty(8, dtype=torch.int32, device='xpu')
        expected = torch.arange(length, dtype=torch.int64).repeat(8) + prefix[0].item()
        expected_starts = torch.arange(8, dtype=torch.int32) * length

        def check():
            torch.testing.assert_close(positions.cpu(), expected, atol=0, rtol=0)
            torch.testing.assert_close(starts.cpu(), expected_starts, atol=0, rtol=0)

        launch = lambda: ops['compute_position'](positions, starts, prefix, lengths)
        yield f'position-t{8 * length}-b8', launch, check, None
        hinted = lambda: ops['compute_position_with_hint'](positions, starts, prefix, lengths, length)
        yield f'position-hint-t{8 * length}-b8', hinted, check, None


def logits_cases(torch, ops):
    vocab, hidden = 202048, 4096
    original = torch.randn((8, vocab), device='xpu', dtype=torch.float32) * 4
    logits = original.clone()
    cap = 20.0
    expected_logits = torch.tanh(original.cpu() / cap) * cap
    softcap = lambda: ops['softcap_logits'](logits, cap)
    reset = lambda: logits.copy_(original)
    check_logits = lambda: torch.testing.assert_close(logits.cpu(), expected_logits, atol=3e-5, rtol=3e-5)
    yield 'softcap-fp32-b8-v202048', softcap, check_logits, reset
    weight_cpu = torch.randn((vocab, hidden), dtype=torch.bfloat16)
    weight = weight_cpu.to('xpu')
    for tokens in (8, 32768):
        ids_cpu = torch.randint(0, vocab, (tokens,), dtype=torch.int64)
        ids_cpu[:8] = torch.tensor([-1, 0, 1, vocab - 1, vocab, 100, 70000, 150000])
        ids = ids_cpu.to('xpu')
        output = torch.empty(tokens, hidden, dtype=torch.bfloat16, device='xpu')
        invalid = (ids_cpu < 0) | (ids_cpu >= vocab)
        expected = weight_cpu[ids_cpu.clamp(0, vocab - 1)].clone()
        expected[invalid] = 0
        embedding = lambda: ops['vocab_parallel_embedding'](ids, weight, output, 0, vocab, 0, vocab, vocab)
        check_embedding = lambda: torch.testing.assert_close(output.cpu(), expected, atol=0, rtol=0)
        yield f'embedding-bf16-t{tokens}-b8', embedding, check_embedding, None
        results = []

        def mask():
            results[:] = ops['vocab_mask'](ids, 0, vocab, 0, vocab, vocab)

        def check_mask():
            torch.testing.assert_close(results[0].cpu(), ids_cpu * (~invalid), atol=0, rtol=0)
            torch.testing.assert_close(results[1].cpu(), invalid, atol=0, rtol=0)

        yield f'vocab-mask-t{tokens}-b8', mask, check_mask, None


def main():
    args = arguments()
    import sgl_kernel
    import torch

    if not torch.xpu.is_available() or not sgl_kernel.is_xe3_arch():
        raise SystemExit('These kernels require Xe35 compute capability 3.5')
    torch.manual_seed(42)
    cases = {'memory': memory_cases, 'position': position_cases, 'logits-vocab': logits_cases}[args.component]
    ops = native_operations(torch, args)
    results = []
    for index, (name, launch, check, reset) in enumerate(cases(torch, ops)):
        if index:
            torch.xpu.synchronize()
            time.sleep(2)
        if reset:
            reset()
        launch()
        torch.xpu.synchronize()
        check()
        result = {'case': name, 'accuracy': 'PASS', **time_graph(torch, launch, args, reset)}
        results.append(result)
        print(json.dumps(result), flush=True)
    torch.xpu.synchronize()
    library_names = {
        'memory': ('PagedAlloc', 'WriteReqToTokenPool'),
        'position': ('ComputePosition',),
        'logits-vocab': ('SoftcapLogits', 'VocabMask'),
    }[args.component]
    libraries = {
        line.split()[-1]
        for line in Path('/proc/self/maps').read_text().splitlines()
        if any('libsgl-ops-sycl-' + name + '.so' in line for name in library_names)
    }
    if len(libraries) != len(library_names):
        raise RuntimeError(f'expected all tested native libraries to be mapped: {library_names}; got {libraries}')
    report = {
        'component': args.component,
        'label': args.label,
        'api': args.api,
        'registration': 'normal package initialization; bound torch.ops.sgl_kernel references',
        'serving_batch': 8,
        'isl': 4096,
        'osl': 16,
        'tp': 1,
        'ep': 1,
        'gpu_affinity': os.environ.get('ZE_AFFINITY_MASK'),
        'device': torch.xpu.get_device_name(0),
        'torch': torch.__version__,
        'package': str(Path(sgl_kernel.__file__).resolve()),
        'samples': args.samples,
        'calls': args.calls,
        'timing': 'XPU graph events; per graph node; input reset excluded',
        'libraries': {library: hashlib.sha256(Path(library).read_bytes()).hexdigest() for library in sorted(libraries)},
        'results': results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
