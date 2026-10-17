import json
import os
from contracts import ROOT, provider, write_json, code_hashes


def main():
    assert os.environ.get('ONEAPI_DEVICE_SELECTOR') == 'level_zero:cpu'
    assert os.sched_getaffinity(0) == set(range(96, 112))
    torch, semantic, reference, framework, plan, bundles, proof = provider()
    assert not torch.xpu.is_initialized()
    write_json(ROOT / 'CURRENT_NORMAL_STAGE_INPUT_MAPS.json', {'actual_exit': 0, 'provider': proof,
               'cases': len(plan['cases']), 'GPU_initialized': False, 'driver_files_sha256': code_hashes()})
    print(json.dumps({'current_normal_stage_input_maps': 0, 'GPU_initialized': False}))


if __name__ == '__main__':
    main()
