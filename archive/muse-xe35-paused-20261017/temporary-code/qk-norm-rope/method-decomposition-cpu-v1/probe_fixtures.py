import argparse
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN.parent / 'normal-v4-model-config-perf-v1'))
from run_perf import digest, local_path, normal_audit, scope, verify


def main():
    scope()
    parser = argparse.ArgumentParser()
    parser.add_argument('--role', required=True, choices=('source', 'port'))
    args = parser.parse_args()
    assert set(os.sched_getaffinity(0)) == set(range(32, 48))
    import torch

    assert not torch.xpu.is_initialized()
    data = verify()
    mapped = normal_audit(args.role)
    cases = []
    for case in data['cases']:
        path = local_path(case['fixture']['path'])
        assert digest(path) == case['fixture']['sha256']
        tensors = torch.load(path, map_location='cpu', weights_only=True)
        positions = tensors['positions']
        expected = torch.int64 if case['group'] == 'actual-config-semantic-synthetic' else torch.int32
        assert positions.dtype == expected and positions.device.type == 'cpu'
        assert tensors['original'].dtype == torch.bfloat16
        cases.append({'case': case['name'], 'group': case['group'], 'input_sha256': case['fixture']['sha256'],
            'QKVO_dtype': str(tensors['original'].dtype), 'position_dtype': str(positions.dtype),
            'position_min': positions.min().item(), 'position_max': positions.max().item(),
            'adapter_int32_conversion_branch_taken': positions.dtype != torch.int32,
            'query_weight_original_dtype': str(tensors['query_weight'].dtype),
            'actual_call_query_weight_dtype': 'torch.bfloat16', 'legacy_weight_BF16_roundtrip_previously_verified': True})
        del tensors, positions
    assert not torch.xpu.is_initialized()
    result = {'actual_exit': 0, 'role': args.role, 'normal_full_package_audit': mapped,
        'XPU_initialized': False, 'cases': cases, 'fixture_count': len(cases), 'GPU_or_compile_executed': False}
    (RUN / f'normal-host-dtype-{args.role}.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'actual_exit': 0, 'role': args.role, 'cases': len(cases), 'normal_mapped_natives': mapped['mapped_count'], 'XPU_initialized': False}))


if __name__ == '__main__':
    main()
