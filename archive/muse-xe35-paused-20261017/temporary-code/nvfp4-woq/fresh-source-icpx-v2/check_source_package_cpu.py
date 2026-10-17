import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def digest(path):
    return hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


directory = Path(__file__).resolve().parent
attempt = directory / 'runs' / 'nv-source-icpx-native-v2-20261017'
baseline = directory.parent / 'native-diagnostic/source/site'
source = Path('/root/worktrees/sgl-kernel-xpu/muse-spark-nvfp4-woq-main-20261014')
require(set(os.sched_getaffinity(0)) == set(range(80, 96)), 'CPU80–95 required')
for phase in ('torch-cpu-probe', 'configure', 'tested-native-build', 'supervisor'):
    require((attempt / (phase + '.exit')).read_text().strip() == '0', phase + ' did not exit0')
    drain = json.loads((attempt / (phase + '.drain.json')).read_text())
    require(drain['drained'] and not drain['live_after'], phase + ' was not drained')
    for identity in drain['identities']:
        try:
            fields = Path('/proc') / str(identity['pid']) / 'stat'
            fields = fields.read_text().rpartition(')')[2].split()
        except FileNotFoundError:
            continue
        require(int(fields[19]) != int(identity['start_ticks']) or fields[0] == 'Z', 'Owned process still live')
require((directory / 'nv-source-icpx-native-v2-20261017-host-supervisor.exit').read_text().strip() == '0', 'Host did not exit0')
policy = json.loads((attempt / 'compiler-policy-verified.json').read_text())
require(policy['compiler_id'] == 'IntelLLVM', 'Actual IntelLLVM compiler required')
require(policy['original_source_flags_preserved'] and policy['NV_original_GRF512_preserved'], 'Original flags required')
for name in ('CMAKE_CXX_COMPILER', 'SYCL_COMPILER', 'SYCL_HOST_COMPILER'):
    require(Path(policy[name]).name == 'icpx', 'Actual icpx required: ' + name)
require(Path(policy['CMAKE_C_COMPILER']).name == 'icx', 'Actual C icx required')
post = json.loads((attempt / 'source-readonly-postverify.json').read_text())
require(post['all_kernel_bodies_helpers_preserved'] and post['only_source_delta'] == ['cmake/BuildFlags.cmake'], 'SOURCE body/helpers changed')
archive = json.loads((attempt / 'runtime-supervisor-SHA.json').read_text())
for name, expected in archive.items():
    require(digest(directory / name) == expected, 'Supervisor SHA changed: ' + name)
    require(digest(attempt / 'supervisor-source-used' / name) == expected, 'Archived supervisor SHA changed: ' + name)
manifest = json.loads((attempt / 'source-binary-manifest.json').read_text())
site = attempt / 'site'
require(Path(manifest['site']).resolve() == site.resolve(), 'Unexpected site')
require(manifest['source_revision'] == 'd983e877d6bf97015027bd3b1379ff047009a5dd', 'Original SOURCE pin changed')
require(manifest['source_cutlass_revision'] == 'e87870bb7aacf83bfb3be9da685f148316b8df23', 'Original CUTLASS pin changed')
actual = {str(path.relative_to(site)): digest(path) for path in site.rglob('*') if path.is_file()}
require(actual == manifest['package_files_sha256'], 'Package inventory SHA changed')
original = {str(path.relative_to(baseline)): digest(path) for path in baseline.rglob('*') if path.is_file()}
require(actual.keys() == original.keys(), 'Package files added or removed')
changed = sorted(name for name in actual if actual[name] != original[name])
require(changed == sorted('sgl_kernel/' + name for name in manifest['native']), 'Only four fresh natives may differ')
require(len(manifest['native']) == 4, 'Exactly four fresh native libraries required')
registration = list((site / 'sgl_kernel').glob('common_ops*.so'))
require(len(registration) == 1 and digest(registration[0]) == manifest['registration_sha256'], 'Legacy registration changed')
for name in ('nvfp4.py', 'cutlass_moe.py'):
    content = subprocess.check_output(['git', '-c', 'safe.directory=' + str(source), '-C', str(source), 'show', manifest['source_revision'] + ':python/sgl_kernel/' + name])
    require(hashlib.sha256(content).hexdigest() == digest(site / 'sgl_kernel' / name), 'Original NV Python differs')
sys.path.insert(0, str(site))
import torch

require(not torch.xpu.is_initialized(), 'GPU initialized before import')
import sgl_kernel

require(Path(sgl_kernel.__file__).resolve() == (site / 'sgl_kernel/__init__.py').resolve(), 'Package not imported from isolated site')
require(Path(sgl_kernel.common_ops.__file__).resolve() == registration[0].resolve(), 'Wrong registration imported')
schemas = {}
for name in ('nvfp4_woq_prep', 'nvfp4_woq_bf16_grouped_mm', 'nvfp4_woq_mxfp8_scaled_grouped_mm', 'mxfp4_moe_scale_to_mn_major', 'fp8_blockwise_scaled_grouped_mm'):
    schemas[name] = str(getattr(torch.ops.sgl_kernel, name).default._schema)
mapped = set()
for line in Path('/proc/self/maps').read_text().splitlines():
    fields = line.split(maxsplit=5)
    if len(fields) == 6 and fields[5].startswith('/'):
        mapped.add(str(Path(fields[5]).resolve()))
native = {}
for name, metadata in manifest['native'].items():
    library = site / 'sgl_kernel' / name
    require(str(library.resolve()) in mapped, 'Fresh native not mapped: ' + name)
    require(digest(library) == metadata['sha256'] == digest(attempt / 'build' / name), 'Fresh native SHA mismatch')
    exports = subprocess.check_output(['nm', '-D', '--defined-only', str(library)], text=True)
    public = {line.split()[-1] for line in exports.splitlines() if len(line.split()) >= 3 and line.split()[-2] == 'T'}
    require(set(metadata['original_exports_mangled']).issubset(public), 'Original ABI export missing')
    dynamic = subprocess.check_output(['readelf', '-d', str(library)], text=True)
    require('Library soname: [' + name + ']' in dynamic and '_port_payload' not in dynamic, 'Native SONAME/provenance mismatch')
    native[name] = {'mapped_path': str(library.resolve()), 'sha256': metadata['sha256'], 'original_ABI_exports_verified': True}
require(not torch.xpu.is_initialized(), 'GPU initialized by package import')
report = {'actual_check_exit': 0, 'scope': manifest['mode'], 'site': str(site), 'package_file_count': len(actual), 'changed_package_files': changed, 'source_revision': manifest['source_revision'], 'cutlass_revision': manifest['source_cutlass_revision'], 'registration_revision': manifest['registration_revision'], 'registration_sha256': manifest['registration_sha256'], 'compiler_policy_sha256': digest(attempt / 'compiler-policy-verified.json'), 'binary_manifest_sha256': digest(attempt / 'source-binary-manifest.json'), 'normal_package_import_verified': True, 'GPU_initialized': False, 'kernel_invoked': False, 'accuracy_verified': False, 'performance_verified': False, 'official_full_acceptance': False, 'schemas': schemas, 'native': native}
print(json.dumps(report, indent=2))
