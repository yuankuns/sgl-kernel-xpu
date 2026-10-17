import json
import os
import subprocess
import sys
import time

from checks import ROOT, ROUND, actual_normal_import, digest, package_gate, runtime_path, write_json


def main():
    if os.environ.get('ONEAPI_DEVICE_SELECTOR') != 'level_zero:cpu':
        raise RuntimeError('Normal-package pregrant gate is CPU only')
    manifest, manifest_file, site, protected = package_gate()
    torch, sgl_kernel, mapped = actual_normal_import(manifest, manifest_file, site, True)
    test = runtime_path(manifest['source']['source'], ROOT) / 'tests/test_bf16_gemm_nt_xpu.py'
    command = [sys.executable, '-B', '-m', 'pytest', str(test), '-q', '-p', 'no:cacheprovider',
               '-k', 'dispatch_capability_without_device_name or output_n_validation']
    (ROUND / 'CPU-units.command').write_text(' '.join(command) + '\n')
    with (ROUND / 'CPU-units.log').open('w') as output:
        status = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT).returncode
    (ROUND / 'CPU-units.exit').write_text(str(status) + '\n')
    if status != 0:
        raise RuntimeError('Original real CPU units failed: ' + str(status))
    proof = {'epoch': time.time(), 'version': 'canonical-v4-icpx', 'scope': 'pre-wave CPU normal-package gate',
             'actual_normal_import_verified': True, 'actual_xpu_available': False, 'actual_package': sgl_kernel.__file__,
             'actual_registered_Xe35_XPU_op': True, 'CPU_units_actual_exit': status,
             'source_binary_manifest_sha256': digest(manifest_file), 'protected_source': protected,
             'installed_native_count': len(manifest['binaries']), 'snapshot_Python_count': len(manifest['python_snapshot_files']),
             'GPU_started': False}
    proof.update(mapped)
    write_json(ROUND / 'cpu-normal-package-gate.json', proof)
    print(json.dumps({'CPU_actual_gate_exit': 0, 'mapped_DSO_count': len(mapped['loaded_libraries']),
                      'installed_native_count': proof['installed_native_count'], 'GPU_started': False}), flush=True)


if __name__ == '__main__':
    main()
