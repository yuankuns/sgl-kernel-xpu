import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time


art = Path(__file__).resolve().parent
root = art.parent.parent / 'pr/muse-fmha-xe35'
archive = art / 'canonical-handoff-20261017'
archive.mkdir(exist_ok=False)
stop = json.loads((art / 'full-build-v2-stop-snapshot.json').read_text())
assert stop['all_old_processes_terminal'] is True
assert stop['residual_active_build_processes'] == {}
assert (art / 'full-build-v2-j8.exit').read_text().strip() == '130'
assert (art / 'official-stage-watch-v2.exit').read_text().strip() == '143'
files = set()
for pattern in ('full-build*.log', 'full-build*.exit', 'full-build*.handle', 'full-build*.json',
                'full-build*-resource-*.txt', 'official-stage-watch-v2.*'):
    files.update(art.glob(pattern))
files.update(art / name for name in ('build_full_v2.sh', 'supervise_full_v2.sh', 'source-port-manifest.json',
                                   'source-binary-sha256.txt', 'changed-files.txt'))
records = []
for path in sorted(files):
    target = archive / path.name
    shutil.copy2(path, target)
    records.append({'source': str(path), 'archive': str(target),
                    'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
for name in ('CMakeCache.txt', 'build.ninja', 'cmake_install.cmake', '.ninja_log', '.ninja_deps'):
    path = root / 'build-xe35' / name
    if path.exists():
        target = archive / ('build-metadata-' + name.lstrip('.'))
        shutil.copy2(path, target)
        records.append({'source': str(path), 'archive': str(target),
                        'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
resource = subprocess.check_output(['free', '-h'], text=True)
(archive / 'resource-after-stop.txt').write_text(resource)
source = json.loads((art / 'source-port-manifest.json').read_text())
for name, digest in source['files'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
report = {'epoch': time.time(), 'handoff': 'main_canonical_real_incremental_all_fresh_install',
          'own_full_supervisor_host_pid': 2245659, 'own_full_ninja_host_pid': 2248955,
          'own_full_root_container_pid': 84746, 'own_full_ninja_container_pid': 85233,
          'old_full_tree_terminal_processes': len(stop['processes']), 'old_full_exit': 130,
          'old_official_wait_supervisor_host_pid': 2256532, 'old_official_wait_child_host_pid': 2256535,
          'old_official_wait_exit': 143, 'residual_active_build_processes': {},
          'own_gpu5_flock_held': False, 'gpu5_lease_retained': True, 'runtime_cpus': '16-31',
          'diagnostic_source_native_and_worktree_unchanged_by_stop': True,
          'source_write_set_count': len(source['files']), 'public_cutlass_pin': source['public_cutlass_pin'],
          'no_objects_or_old_libraries_copied': True, 'archive': str(archive), 'files': records,
          'remaining_scope': 'pending upstream fallback correction; canonical real default-full official package; GPU5 independent official barrier full38 precision and BAAB revalidation',
          'acceptance': 'not_official_pass_no_claim_of_clean_independent_build'}
(archive / 'handoff-manifest.json').write_text(json.dumps(report, indent=2) + '\n')
(art / 'canonical-stop-complete.json').write_text(json.dumps({key: value for key, value in report.items()
                                                           if key != 'files'}, indent=2) + '\n')
print(json.dumps({'stop_complete': True, 'old_full_exit': 130, 'old_wait_exit': 143,
                  'source_files_verified_unchanged': len(source['files']), 'archived_files': len(records),
                  'archive': str(archive), 'gpu5_lease_retained': True}), flush=True)
print(resource)
