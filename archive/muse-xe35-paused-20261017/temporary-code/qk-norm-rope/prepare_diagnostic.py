import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


root = Path(__file__).resolve().parent
diagnostic = root / 'diagnostic'
target = root.parents[1] / 'pr/muse-qk-norm-rope-xe35'
evidence = root.parents[2] / 'sgl-kernel-xpu/review-validation-artifacts/qk-before-after-20261017'
assert (diagnostic / 'build.exit').read_text().strip() == '0'
subprocess.run(['python3', str(root / 'verify_source.py')], check=True)
subprocess.run(['python3', str(root / 'audit_port.py')], check=True)
source_site = evidence / 'after/site'
site = diagnostic / 'port/site'
assert not (diagnostic / 'provenance.json').exists(), 'Never overwrite a prepared diagnostic package'
source_manifest = json.loads((evidence / 'after/site-manifest.json').read_text())
for record in source_manifest:
    assert digest(evidence / record['path']) == record['sha256']
if not site.exists():
    shutil.copytree(source_site, site)
package = site / 'sgl_kernel'
bridge = diagnostic / 'build/libsgl-ops-sycl-FusedQKNormRope.so'
kernel = target / 'build/src/libsgl-ops-sycl-FusedQKNormRope_xe35.so'
assert digest(kernel) == '6d8d51b64cba7276fa6fed886d53f51b0485cb0b35e79a021956abacbf7b1079'
for binary in (bridge, kernel):
    before = digest(binary)
    shutil.copy2(binary, package / binary.name)
    assert digest(binary) == digest(package / binary.name) == before
differences = []
for original in sorted(source_site.rglob('*')):
    if original.is_file():
        relative = original.relative_to(source_site)
        if digest(original) != digest(site / relative):
            differences.append(str(relative))
assert differences == ['sgl_kernel/libsgl-ops-sycl-FusedQKNormRope.so'], differences
assert len([path for path in site.rglob('*') if path.is_file()]) == len(source_manifest) + 1
elf = subprocess.check_output(['readelf', '-d', str(bridge)], text=True)
assert 'libsgl-ops-sycl-FusedQKNormRope_xe35.so' in elf and '$ORIGIN' in elf
(diagnostic / 'native-forward-elf.log').write_text(elf)
manifest = [{'path': str(path.relative_to(diagnostic)), 'sha256': digest(path)} for path in sorted(site.rglob('*')) if path.is_file()]
(diagnostic / 'port/site-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
source_verification = json.loads((root / 'source-verification.json').read_text())
record = {
    'mode': 'source-full-package/native-abi-port-diagnostic',
    'official_full_package_pass': False,
    'source_commit': source_verification['source_commit'],
    'target_base': '833c95208dccd2af5628c121222f91186a5b2de7',
    'source_site': str(source_site),
    'port_site': str(site),
    'source_qk_library_sha256': source_verification['source_qk_binary_sha256'],
    'port_qk_library_sha256': digest(kernel),
    'native_forward_sha256': digest(bridge),
    'native_forward_source_sha256': digest(diagnostic / 'xe35/FusedQKNormRopeAbiForward.cpp'),
    'source_manifest_sha256': digest(evidence / 'after/site-manifest.json'),
    'port_manifest_sha256': digest(diagnostic / 'port/site-manifest.json'),
    'unchanged_source_package_files': len(source_manifest) - 1,
    'replaced_source_package_files': differences,
    'added_port_package_files': ['sgl_kernel/libsgl-ops-sycl-FusedQKNormRope_xe35.so'],
    'common_ops_sha256': digest(next(package.glob('common_ops*.so'))),
    'port_source_hashes': {str(path.relative_to(target)): digest(path) for path in sorted((target / 'src/sycl/xe35').glob('FusedQKNormRope*'))},
    'source_package_known_difference': source_verification['known_full_package_source_difference'],
    'limitations': ['Not the official full wheel', 'Original SOURCE native registration unchanged', 'No new operator registration, shim, or load_library', 'Direct xe35 API and official pytest await official full wheel'],
}
(diagnostic / 'provenance.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2))
