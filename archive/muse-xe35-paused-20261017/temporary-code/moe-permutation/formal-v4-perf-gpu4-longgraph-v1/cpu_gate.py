import hashlib
import json
import os
import shlex
import subprocess
import sys
import time

from checks import ROOT, ROUND, actual_normal_import, digest, package_gate, runtime_path, write_json


def main():
    if os.environ.get('ONEAPI_DEVICE_SELECTOR') != 'level_zero:cpu':
        raise RuntimeError('Normal-provider pregrant gate is CPU only')
    expected_affinity = set(range(96, 112))
    if os.sched_getaffinity(0) != expected_affinity:
        raise RuntimeError('Real container affinity must be CPU96–111')
    manifest, manifest_file, site, protected = package_gate()
    torch, sgl_kernel, mapped = actual_normal_import(manifest, manifest_file, site, True)
    source = json.loads((ROOT / 'source-native-icpx-v1-accuracy/fresh.json').read_text())
    shared_inputs = {}
    for case in source['cases']:
        dtype = torch.int32 if case['dtype'] == 'torch.int32' else torch.int64
        active_experts = int(case['case'].split('active')[1])
        ids = (torch.arange(case['tokens'] * case['topk'], dtype=dtype).reshape(case['tokens'], case['topk']) * 7) % active_experts
        checksum = hashlib.sha256(ids.numpy().tobytes()).hexdigest()
        if checksum != case['input_sha256']:
            raise RuntimeError('Reconstructed complete input bytes differ from actual validated SOURCE: ' + case['case'])
        shared_inputs[case['case']] = ids
    if len(shared_inputs) != 6:
        raise RuntimeError('All six SOURCE input cases are required')
    inputs_file = ROUND / 'shared-SOURCE-inputs.pt'
    torch.save(shared_inputs, inputs_file)
    test = runtime_path(manifest['source']['source'], ROOT) / 'tests/test_mxfp4_small_permutation.py'
    command = [sys.executable, '-B', '-m', 'pytest', str(test), '--collect-only', '-q', '-p', 'no:cacheprovider']
    (ROUND / 'CPU-collection.command').write_text(shlex.join(command) + '\n')
    with (ROUND / 'CPU-collection.log').open('w') as output:
        status = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT).returncode
    (ROUND / 'CPU-collection.exit').write_text(str(status) + '\n')
    if status != 0 or '30 tests collected' not in (ROUND / 'CPU-collection.log').read_text():
        raise RuntimeError('All original 30 precision cases must collect without any case deletion')
    proof = {'epoch': time.time(), 'version': 'canonical-v4-icpx', 'scope': 'formal-performance CPU normal-provider gate',
             'actual_normal_import_verified': True, 'actual_xpu_available': False, 'actual_package': sgl_kernel.__file__,
             'actual_registered_Xe35_XPU_op': True, 'CPU_collection_actual_exit': status,
             'original_precision_cases_collected': 30, 'source_binary_manifest_sha256': digest(manifest_file),
             'protected_source': protected, 'actual_container_CPU_affinity': sorted(os.sched_getaffinity(0)),
             'installed_native_count': len(manifest['binaries']), 'snapshot_Python_count': len(manifest['python_snapshot_files']),
             'shared_SOURCE_inputs_sha256': digest(inputs_file), 'all_six_full_input_byte_SHA_match_actual_SOURCE': True,
             'SOURCE_input_provenance': 'CPU reconstruction of deterministic original SOURCE inputs; complete byte SHA verified against actual SOURCE run, not claimed to be historical saved input files',
             'GPU_started': False}
    proof.update(mapped)
    write_json(ROUND / 'cpu-normal-package-gate.json', proof)
    print(json.dumps({'CPU_actual_gate_exit': 0, 'mapped_DSO_count': len(mapped['loaded_libraries']),
                      'actual_container_affinity': sorted(os.sched_getaffinity(0)), 'GPU_started': False}), flush=True)


if __name__ == '__main__':
    main()
