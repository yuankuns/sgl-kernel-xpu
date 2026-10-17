import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import torch


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    directory = Path(__file__).resolve().parent
    previous = directory.parent / 'current-source-accuracy-gpu8-v1'
    assert sorted(os.sched_getaffinity(0)) == list(range(64, 80))
    assert not torch.xpu.is_initialized()
    for phase, filename in [('normal-package-maps-source', 'verify_runtime_cpu.py'), ('original-collection', 'prepare_cpu.py')]:
        result = subprocess.run(['/opt/gfx-deps/venv/bin/python', str(directory / filename)], capture_output=True, text=True)
        (directory / f'CPU-{phase}.log').write_text(result.stdout + result.stderr)
        (directory / f'CPU-{phase}.exit').write_text(str(result.returncode) + '\n')
        assert result.returncode == 0, result.stdout + result.stderr
    for path in directory.glob('*.py'):
        ast.parse(path.read_text(), filename=str(path))
    for path in directory.glob('*.sh'):
        subprocess.run(['bash', '-n', str(path)], check=True)
    evidence = torch.load(previous / 'CPU-first-failure-full-reference-comparison.pt', map_location='cpu', weights_only=True, mmap=True)
    cpu_rank = evidence['CPU_FP32_ranking']
    device_rank = evidence['saved_PyTorch_XPU_ranking']
    ids = evidence['actual_native_ids'].long()
    selected_cpu = cpu_rank.gather(1, ids)
    selected_device = device_rank.gather(1, ids)
    different = selected_cpu.contiguous().view(torch.int32) != selected_device.contiguous().view(torch.int32)
    cpu_tied = torch.empty_like(different)
    device_tied = torch.empty_like(different)
    for slot in range(ids.shape[1]):
        cpu_tied[:, slot] = (cpu_rank == selected_cpu[:, slot, None]).sum(dim=1) > 1
        device_tied[:, slot] = (device_rank == selected_device[:, slot, None]).sum(dim=1) > 1
    coordinates = different.nonzero()
    torch.save({'coordinates': coordinates, 'actual_ids': ids[different], 'CPU_selected_ranking': selected_cpu[different],
                'saved_PyTorch_XPU_selected_ranking': selected_device[different], 'CPU_tied': cpu_tied[different],
                'saved_XPU_tied': device_tied[different]}, directory / 'CPU-XPU-selected-score-differences.pt')
    quality = {'selected_CPU_XPU_bit_differences': int(different.sum()),
               'different_and_CPU_rank_non_tied': int((different & ~cpu_tied).sum()),
               'different_and_both_rank_non_tied': int((different & ~cpu_tied & ~device_tied).sum()),
               'selected_CPU_XPU_max_abs_error': float((selected_cpu - selected_device).abs().max()),
               'not_native_internal_scores': True,
               'limitation': 'CPU-vs-PyTorch-XPU reference bit equality is not itself native selected-value correctness. New CPU gather(actualIDs)==CPU topk.values remains exact; cross-reference non-tie score differences reported without tolerance adjustment.'}
    (directory / 'CPU-reference-quality-statistics.json').write_text(json.dumps(quality, indent=2) + '\n')
    assert int((directory / 'CPU-contract-audit.exit').read_text()) == 0
    prior = directory.parent / 'current-source-accuracy-gpu8-v2'
    assert int((prior / 'CPU-contract-audit.exit').read_text()) == 1
    old_ready = json.loads((previous / 'CPU-READY-v2.json').read_text())
    bound = {}
    for name, expected in old_ready['bound_files_SHA256'].items():
        path = Path(name.replace('/home/syk/', '/root/', 1))
        assert digest(path) == expected, str(path)
        bound[name] = expected
    preserved = ['host-launch.exit', 'gpu-runtime.exit', 'six-production-shapes.exit',
                 'CPU-first-failure-classification-v2.json', 'CPU-first-failure-classification-v2.exit',
                 'CPU-first-failure-all256-candidates.json', 'CPU-first-failure-full-reference-comparison.pt',
                 'first-failure-six-production-shapes-full-local-tensors.pt', 'first-failure-six-production-shapes.json']
    for name in preserved:
        path = previous / name
        bound[str(path).replace('/root/', '/home/syk/', 1)] = digest(path)
    for path in prior.iterdir():
        if path.is_file() and path.suffix in ('.py', '.sh', '.json', '.exit', '.pt'):
            bound[str(path).replace('/root/', '/home/syk/', 1)] = digest(path)
    for path in directory.iterdir():
        if path.is_file() and path.suffix in ('.py', '.sh', '.json', '.exit', '.pt'):
            bound[str(path).replace('/root/', '/home/syk/', 1)] = digest(path)
    stage = directory.parent / 'fresh-current-source-icpx-native-v3'
    bundle = {'epoch': time.time(), 'CPU_preparation_actual_exit': 0, 'CPU_READY': True,
              'reference_contract': 'CPU selected ranking equals CPU topk.values bit-exact, legal unique IDs, CPU actual-ID unbiased weights original atol2.5e-4/rtol0; cross-reference score differences diagnostic only.',
              'scope': 'original14_then_original1300_then_sixproduction_then_threefixtures',
              'original_silu_opmath_cases': 14, 'original_full_precision_cases': 1300, 'production_shapes': 6,
              'complete_original_fixtures': 3, 'gpu': 8, 'accuracy_only': True, 'performance_authorized': False,
              'CPU_affinity': list(range(64, 80)), 'OMP_NUM_THREADS': 8,
              'current_stage_manifest_SHA256': digest(stage / 'source-native-package-manifest.json'),
              'bound_files_SHA256': bound, 'CPU_reference_quality': quality, 'cross_reference_bit_differences_diagnostic_only_not_acceptance': True,
              'v1_GPU_actual_exit_preserved': 1, 'CPU_contract_audit_script_actual_exit': 0, 'prior_v2_cross_reference_hold_actual_exit_preserved': 1,
              'GPU_grant_issued': False, 'GPU_work_started': False, 'no_compile_PR_math_changes': True}
    assert not torch.xpu.is_initialized()
    (directory / 'CPU-READY.json').write_text(json.dumps(bundle, indent=2) + '\n')
    print(json.dumps({name: value for name, value in bundle.items() if name != 'bound_files_SHA256'}, indent=2))


if __name__ == '__main__':
    main()
