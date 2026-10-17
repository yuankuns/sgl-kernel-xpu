import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--serving-batch', type=int, choices=[8], default=8)
    parser.add_argument('--isl', type=int, choices=[4096], default=4096)
    parser.add_argument('--osl', type=int, choices=[16], default=16)
    parser.add_argument('--tp', type=int, choices=[1], default=1)
    parser.add_argument('--ep', type=int, choices=[1], default=1)
    options = parser.parse_args()
    import torch
    import sgl_kernel
    from input_forensics import tensor_sha256

    directory = Path(__file__).resolve().parent
    check = json.loads((directory / 'normal-source-precheck.json').read_text())
    assert Path(sgl_kernel.__file__).resolve() == Path(check['package'])
    assert sorted(os.sched_getaffinity(0)) == list(range(64, 80)) and os.environ['ZE_AFFINITY_MASK'] == '8'
    assert torch.xpu.is_available() and sgl_kernel.is_xe3_arch(), 'Real Xe3.5 SOURCE only'
    torch.set_num_threads(8)
    mapped = {Path(line.split()[-1]).resolve() for line in Path('/proc/self/maps').read_text().splitlines() if '/sgl_kernel/' in line and '.so' in line}
    assert mapped == {Path(path) for path in check['mapped_native_sha256']}
    for path in mapped:
        with path.open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == check['mapped_native_sha256'][str(path)]
    fixtures = json.loads((directory / 'CPU-original-fixtures.json').read_text())['fixtures']
    results = []
    failed = False
    for fixture in fixtures:
        torch.xpu.synchronize()
        time.sleep(2)
        reference = torch.load(directory / fixture['fixture_file'], weights_only=True, map_location='cpu')
        for name, value in reference.items():
            assert tensor_sha256(value) == fixture['tensors'][name]
        original_input = reference['original_input']
        actual_input = original_input.to('xpu')
        if fixture['kind'] == 'surface':
            activation = torch.empty(1, 32, dtype=torch.bfloat16, device='xpu')
            sgl_kernel.silu_and_mul(actual_input, out=activation)
        else:
            activation = sgl_kernel.silu_and_mul(actual_input)
        separate_q, separate_s = sgl_kernel.sgl_per_token_group_quant_fp4(activation)
        fused_q, fused_s = sgl_kernel.sgl_per_token_group_quant_fp4(actual_input, fuse_silu_and_mul=True)
        torch.xpu.synchronize()
        tensors = reference | {'actual_input': actual_input.cpu(), 'separate_activation': activation.cpu(),
                               'separate_q': separate_q.cpu(), 'separate_s': separate_s.cpu(),
                               'fused_q': fused_q.cpu(), 'fused_s': fused_s.cpu()}
        fingerprints = {name: tensor_sha256(value) for name, value in tensors.items()}
        assert fingerprints['actual_input'] == fixture['tensors']['original_input']
        output = directory / f"actual-{fixture['kind']}-{fixture['hidden']}-full-fixture.pt"
        torch.save(tensors, output)
        gates = {}
        for component in ['separate_q', 'separate_s', 'fused_q', 'fused_s']:
            expected = tensors['expected_s' if component.endswith('_s') else 'expected_q']
            actual = tensors[component]
            try:
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                passed = True
                error = None
            except AssertionError as failure:
                passed = False
                error = str(failure)
            different = torch.count_nonzero(actual != expected).item() if actual.shape == expected.shape else None
            gates[component] = {'passed': passed, 'rtol': 0, 'atol': 0, 'different_elements': different,
                                'checked_elements': expected.numel(), 'assertion_failure': error,
                                'scope': 'independent original zero assertion on complete actual fixture; not original pytest assertion coverage'}
            failed |= not passed
        activation_different = torch.count_nonzero(tensors['separate_activation'].view(torch.int16) != tensors['original_CPU_BF16_activation'].view(torch.int16)).item()
        results.append({'node_suffix': fixture['node_suffix'], 'fixture_file': str(output), 'tensor_SHA256': fingerprints,
                        'four_independent_original_zero_gates': gates, 'separate_activation_BF16_different_elements': activation_different,
                        'normal_SOURCE_package': sgl_kernel.__file__, 'original_pytest_assertions': 'run subsequently without instrumentation',
                        'native_GPU_calls': 'original silu_and_mul and original nonfused/fused quant, once each; no bridge or monkeypatch',
                        'performance': 'not run'})
        print(json.dumps(results[-1]), flush=True)
        (directory / 'independent-four-component-accuracy.json').write_text(json.dumps(results, indent=2) + '\n')
        del actual_input, activation, separate_q, separate_s, fused_q, fused_s
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()
