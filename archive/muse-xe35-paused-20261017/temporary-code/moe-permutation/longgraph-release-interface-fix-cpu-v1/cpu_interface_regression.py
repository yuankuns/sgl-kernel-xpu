import ast
import hashlib
import json
import os
import sys
import time
from pathlib import Path

FOLDER = Path(__file__).resolve().parent
EVIDENCE = FOLDER.parent / 'formal-v4-perf-gpu4-longgraph-v4'


def main():
    if os.environ.get('ONEAPI_DEVICE_SELECTOR') != 'level_zero:cpu':
        raise RuntimeError('CPU-only regression requires selector level_zero:cpu')
    if os.sched_getaffinity(0) != set(range(96, 112)):
        raise RuntimeError('Container CPUs96–111 required')
    original = (EVIDENCE / 'checks.py').read_text()
    candidate = (FOLDER / 'checks_return_candidate.py').read_text()
    expected = original.replace('    _original_grant_preflight(root, grant_file, accuracy_only)\n',
                                '    result = _original_grant_preflight(root, grant_file, accuracy_only)\n', 1) + '    return result\n'
    if candidate != expected:
        raise RuntimeError('Candidate differs beyond the single return contract fix')
    ast.parse(candidate)
    sys.path.insert(0, str(EVIDENCE))
    namespace = {'__name__': 'cpu_only_candidate_contract', '__file__': str(EVIDENCE / 'checks.py')}
    exec(compile(candidate, str(FOLDER / 'checks_return_candidate.py'), 'exec'), namespace)
    result = namespace['grant_preflight'](EVIDENCE.parent, EVIDENCE / 'fresh-grant.json', False)
    if not isinstance(result, tuple) or len(result) != 4:
        raise RuntimeError('Real original grant four-tuple was not forwarded')
    manifest, manifest_path, site, grant = result
    if manifest != json.loads(manifest_path.read_text()) or grant != json.loads((EVIDENCE / 'fresh-grant.json').read_text()):
        raise RuntimeError('Returned actual manifest/grant bytes differ')
    if site != EVIDENCE.parent / 'canonical-v4-icpx/site/sgl_kernel':
        raise RuntimeError('Returned genuine site is not current AFTER')
    rejections = []
    for name, root, path, accuracy_only in [
        ('accuracy_only_not_performance', EVIDENCE.parent, EVIDENCE / 'fresh-grant.json', True),
        ('held_v3_grant_not_accepted', EVIDENCE.parent, EVIDENCE.parent / 'formal-v4-perf-gpu4-longgraph-v3/fresh-grant.json', False),
        ('wrong_module_root', EVIDENCE, EVIDENCE / 'fresh-grant.json', False),
    ]:
        try:
            namespace['grant_preflight'](root, path, accuracy_only)
        except RuntimeError:
            rejections.append(name)
        else:
            raise RuntimeError('Negative contract accepted: ' + name)
    if 'torch' in sys.modules or 'sgl_kernel' in sys.modules:
        raise RuntimeError('Interface-only regression must not initialize torch/package/GPU')
    record = {'epoch': time.time(), 'actual_exit': 0, 'actual_container_CPU_affinity': sorted(os.sched_getaffinity(0)),
              'real_original_manifest_site_grant_four_tuple_forwarded': True,
              'single_return_contract_fix_only': True, 'negative_rejections': rejections,
              'candidate_sha256': hashlib.sha256(candidate.encode()).hexdigest(),
              'old_v4_checks_sha256': hashlib.sha256(original.encode()).hexdigest(),
              'v4_grant_sha256': hashlib.sha256((EVIDENCE / 'fresh-grant.json').read_bytes()).hexdigest(),
              'CPU_contract_validation_only_not_a_grant_consumption_or_new_GPU_authorization': True,
              'torch_imported': False, 'sgl_kernel_imported': False, 'GPU_started': False,
              'original_v4_terminal_exit': (EVIDENCE / 'supervisor.exit').read_text().strip(),
              'original_v4_drained': json.loads((EVIDENCE / 'GPU-drained.json').read_text())}
    (FOLDER / 'CPU_INTERFACE_REGRESSION.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
