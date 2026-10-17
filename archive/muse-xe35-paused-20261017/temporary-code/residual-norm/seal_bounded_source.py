import hashlib
import json
from pathlib import Path
import re
import subprocess
import time


ROOT = Path(__file__).resolve().parent
JOB = ROOT / 'source-before-icpx-bounded-v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


assert (JOB / 'supervisor.exit').read_text().strip() == '0'
terminals = {path.name: json.loads(path.read_text()) for path in JOB.glob('*.terminal.json')}
assert all(record['exit'] == 0 and record['resource_violation'] is None and record['descendants_drained'] for record in terminals.values())
snapshot = json.loads((JOB / 'snapshot-manifest.json').read_text())
post = json.loads((JOB / 'postcheck.json').read_text())
pre = json.loads((JOB / 'precheck.json').read_text())
assert pre['source_digest'] == post['source_digest'] == snapshot['source_digest']
assert pre['dependency'] == post['dependency']
site = json.loads((JOB / 'site-sha.json').read_text())
assert [name for name in site['pre'] if site['pre'][name] != site['post'][name]] == ['libsgl-ops-sycl-FusedNormResidual.so']
package = json.loads((JOB / 'normal-import-proof.json').read_text())
assert package['mapped_libraries']['libsgl-ops-sycl-FusedNormResidual.so'] == post['native_sha256']
assert package['mapped_libraries']['common_ops.cpython-312-x86_64-linux-gnu.so'] == '467502417bd111cbc2df1eecc565a91cc0f96372a83f6692bc2cf0b478f6c896'
dependency = JOB / 'build/src/CMakeFiles/sgl-ops-sycl-FusedNormResidual.dir/sycl/sgl-ops-sycl-FusedNormResidual_gen_FusedNormResidual.cpp.o.depend'
headers = re.findall(r'"([^"]+)"', dependency.read_text())
cutlass_headers = [path for path in headers if 'cutlass' in path.lower()]
helpers = {name: snapshot['original_files'][name]['sha256'] for name in
           ['src/sycl/MemoryAccess.h', 'src/sycl/SYCLHelpers.h', 'src/sycl/Utils.h']}
rows = subprocess.check_output(['ps', '-eo', 'pid,ppid,stat,args'], text=True).splitlines()[1:]
active = []
for row in rows:
    parts = row.strip().split(None, 3)
    if len(parts) == 4 and 'Z' not in parts[2] and parts[3].startswith('/root/') and 'build_bounded_source.py' in parts[3]:
        active.append(row)
assert not active, active
record = {'schema': 1, 'epoch': time.time(), 'session_handle': 3015, 'host_supervisor_pid': 2910952,
    'host_container_supervisor_pid': 2911026, 'container_supervisor_pid': 172303,
    'supervisor_actual_exit': 0, 'native_build_actual_exit': 0, 'normal_import_actual_exit': 0,
    'active_owned_build_processes': active, 'source_ref': snapshot['source_ref'],
    'source_snapshot_digest': snapshot['source_digest'],
    'only_source_delta': snapshot['only_delta'], 'source_kernel_sha256': snapshot['original_files']['src/sycl/FusedNormResidual.cpp']['sha256'],
    'original_helpers_sha256': helpers, 'cutlass': post['dependency'],
    'native_compiler_id': 'IntelLLVM', 'native_compilers': {'cxx': '/opt/gfx-deps/dpcpp/bin/icpx',
        'sycl': '/opt/gfx-deps/dpcpp/bin/icpx', 'host': '/opt/gfx-deps/dpcpp/bin/icpx', 'AOT_link': '/opt/gfx-deps/dpcpp/bin/icpx', 'c': '/opt/gfx-deps/dpcpp/bin/icx'},
    'target': 'sgl-ops-sycl-FusedNormResidual', 'closure_steps': 3,
    'Device_rebuilt': False, 'all_common_ops_HCHead_built': False,
    'prefix': package['prefix'], 'python': package['prefix'] + '/bin/python',
    'tested_native_sha256': post['native_sha256'],
    'historical_GNU_registration_DSO_rebuilt': False,
    'registration_DSO_sha256': package['mapped_libraries']['common_ops.cpython-312-x86_64-linux-gnu.so'],
    'all_mapped_package_libraries': package['mapped_libraries'],
    'site_sha_manifest_sha256': sha(JOB / 'site-sha.json'),
    'fresh_TU_dependency_sha256': sha(dependency), 'fresh_TU_dependency_count': len(headers),
    'fresh_TU_cutlass_headers': cutlass_headers,
    'dependency_limit': 'fresh residual TU depfile records no CUTLASS headers; original clean e878 retained regardless; not a claim for other DSOs',
    'CPU': '48-63', 'outer_jobs': 2, 'MAX_JOBS': 1,
    'peak_tree_rss_bytes': max(record['peak_tree_rss'] for record in terminals.values()),
    'peak_compiler_subtree_rss_bytes': max(record['peak_compiler_subtree_rss'] for record in terminals.values()),
    'compiler_RSS_budget_bytes': 8 * 1024 ** 3, 'tree_RSS_budget_bytes': 16 * 1024 ** 3,
    'RSS_accounting': 'recursive descendants with PID starttime identity, not PGID',
    'flags_and_AOT_exact_match_historical': True, 'Device_ABI_addition_needed': False,
    'GPU_executed': False, 'accuracy_performance_verified': False, 'official_full_acceptance': False,
    'AFTER_requirement': 'later genuine canonical-v4-icpx full/current-install/post0 normal new-op package; no finite PORT bridge'}
(JOB / 'native-package-manifest.json').write_text(json.dumps(record, indent=2) + '\n')
paths = [path for path in JOB.iterdir() if path.is_file() and path.name != 'evidence.sha256']
paths += [ROOT / name for name in ['prepare_bounded_source.py', 'build_bounded_source.py', 'check_bounded_source_import.py', 'run_bounded_source.sh']]
paths += [JOB / 'build/CMakeCache.txt', JOB / 'build/build.ninja', dependency,
          JOB / 'build/src/libsgl-ops-sycl-FusedNormResidual.so']
(JOB / 'evidence.sha256').write_text(''.join(f'{sha(path)}  {path}\n' for path in sorted(paths)))
print(json.dumps({name: value for name, value in record.items() if name != 'all_mapped_package_libraries'}, indent=2))
