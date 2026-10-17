import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import build_pool_icpx_v2 as pool
from overlay_snapshot import manifest_digest, source_inventory


VERSION = 'canonical-v5-icpx'
COMPONENT = 'fmha'
ENTRY = 'src/sycl/xe35/fmha/entry.cpp'
PATCH_SHA256 = 'd990aacd4e5b22eec71ee1ad8bb85a43339d266f0474b0950ae1812b94e5946b'


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def expected_entry(previous):
    replacements = (
        ('#include "sycl/xe35/fmha/dispatch.hpp"',
         '#include <limits>\n#include "sycl/xe35/fmha/dispatch.hpp"'),
        ('  const bool production_shape_xe35 =\n'
         '      (max_seqlen_q == 1 && batch_xe35 == 8 && max_seqlen_k >= 4097 && max_seqlen_k <= 4112) ||\n'
         '      (max_seqlen_q == 4096 && (batch_xe35 == 1 || batch_xe35 == 7 || batch_xe35 == 8) && max_seqlen_k == 4096);',
         '  const bool capacity_hint_xe35 = max_seqlen_k <= 0 && page_table.has_value() &&\n'
         '      page_table->dim() == 2 && k.dim() == 4 && k.size(1) == 64 &&\n'
         '      page_table->size(1) > 0 && page_table->size(1) <= std::numeric_limits<int>::max() / 64;\n'
         '  const int resolved_max_seqlen_k_xe35 = capacity_hint_xe35\n'
         '      ? static_cast<int>(page_table->size(1) * k.size(1)) : max_seqlen_k;\n'
         '  const bool production_shape_xe35 =\n'
         '      (max_seqlen_q == 1 && batch_xe35 == 8 &&\n'
         '       ((resolved_max_seqlen_k_xe35 >= 4097 && resolved_max_seqlen_k_xe35 <= 4112) ||\n'
         '        (capacity_hint_xe35 && resolved_max_seqlen_k_xe35 == 4160))) ||\n'
         '      (max_seqlen_q == 4096 &&\n'
         '       (batch_xe35 == 1 || batch_xe35 == 3 || batch_xe35 == 5 || batch_xe35 == 7 || batch_xe35 == 8) &&\n'
         '       resolved_max_seqlen_k_xe35 == 4096);'),
        ('  mha_fwd_xe35(\n'
         '      q, k, v, q_v_, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,',
         '  mha_fwd_xe35(\n'
         '      q, k, v, q_v_, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, resolved_max_seqlen_k_xe35,'),
    )
    for original, replacement in replacements:
        if previous.count(original) != 1:
            raise RuntimeError('Approved entry delta does not uniquely match original source')
        previous = previous.replace(original, replacement, 1)
    return previous


def host_build():
    artifact = pool.ROOT / COMPONENT
    preparation = artifact / 'v5-metadata-bridge-preparation'
    candidate_path = preparation / 'candidate-source-freeze-v5.json'
    candidate = json.loads(candidate_path.read_text())
    old = json.loads((artifact / 'canonical-source-freeze-v4-icpx.json').read_text())
    source = pool.ROOT.parent / 'pr/muse-fmha-xe35'
    files, selection = source_inventory(source)
    assert files == candidate['files'] and selection == candidate['selection'] == old['selection']
    assert manifest_digest(files) == candidate['source_digest']
    assert files.keys() == old['files'].keys()
    changed = [name for name in files if files[name] != old['files'][name]]
    assert changed == [ENTRY], changed
    previous = preparation / 'before-entry.cpp'
    assert digest(previous) == old['files'][ENTRY]['sha256']
    assert expected_entry(previous.read_text()) == (source / ENTRY).read_text()
    patch = artifact / 'MODEL51_XE35_METADATA_BRIDGE_PROPOSAL_v1.patch'
    assert digest(patch) == PATCH_SHA256
    quiet_path = pool.ROOT / 'MAINTENANCE_QUIET_60S_v2.json'
    quiet = json.loads(quiet_path.read_text())
    assert quiet['actual_exit'] == 0 and quiet['continuous_quiet_seconds'] >= 60
    accuracy_grant_path = pool.ROOT / 'residual-norm/gpu7-history24-accuracy-v1-grant.json'
    accuracy_grant = json.loads(accuracy_grant_path.read_text())
    assert accuracy_grant['gpu'] == 7 and accuracy_grant['accuracy_only'] is True
    assert accuracy_grant['no_timing_events_graphs_warmup'] is True
    assert accuracy_grant['formal_perf_barrier_consumed'] is False
    with ExitStack() as stack:
        for path in (pool.GLOBAL_LOCK, *(Path(f'/tmp/intel-gpu-{gpu}.lock') for gpu in range(4, 7))):
            lock = stack.enter_context(path.open('a'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        compiler_names = {'icpx', 'icx', 'clang', 'clang++', 'cc1', 'cc1plus', 'gcc', 'g++',
                          'ninja', 'make', 'dkms', 'dpkg', 'apt', 'apt-get'}
        live = {pid: item for pid, item in pool.process_snapshot().items() if item['name'] in compiler_names}
        assert not live, live
        assert pool.memory_available_gib() >= pool.JOBS * 8 + pool.MIN_AVAILABLE_GIB
        evidence = artifact / VERSION
        evidence.mkdir(exist_ok=False)
        freeze = artifact / 'canonical-source-freeze-v5-icpx.json'
        with freeze.open('x') as output:
            output.write(json.dumps(candidate, indent=2) + '\n')
        pool.save_json(evidence / 'preflight.json', {
            'epoch': time.time(), 'issuer': 'main-coordinator', 'only_source_delta': changed,
            'approved_patch_sha256': PATCH_SHA256, 'kernel_math_and_generic_fallback_unchanged': True,
            'freeze_sha256': digest(freeze), 'builder_dependency_sha256': digest(pool.TOOLS / 'build_pool_icpx_v2.py'),
            'stable_gpu_locks_reserved_no_GPU_work': list(range(4, 7)), 'live_compilers': live,
            'GPU7_accuracy_only_grant_sha256': digest(accuracy_grant_path),
            'all_new_stable_performance_grants_withheld_during_build': True,
            'maintenance_quiet_60s_sha256': digest(quiet_path),
            'outer_jobs': pool.JOBS, 'inner_jobs': 1, 'cpu_limit': 128,
        })
        pointer = pool.POOL / 'latest-snapshot-manifest.json'
        command = [sys.executable, str(pool.TOOLS / 'overlay_snapshot.py'), '--source', str(source),
                   '--destination', str(pool.SNAPSHOT), '--manifest', str(evidence / 'source-snapshot.json')]
        if pointer.exists():
            command.extend(['--previous-manifest', json.loads(pointer.read_text())['path']])
        subprocess.run(command, check=True)
        pool.save_json(pointer, {'path': str(evidence / 'source-snapshot.json')})
        script = str(pool.TOOLS / 'build_fm_metadata_icpx_v5.py').replace('/home/syk/', '/root/', 1)
        wrapper = [pool.WRAPPER, '--workdir', '/root/sgl-kernel-xpu-up', '--', 'taskset', '-c', '0-127',
                   '/opt/gfx-deps/venv/bin/python', script, '--container-build']
        environment = dict(os.environ, SGLANG_MUSE_CONTAINER='codex-muse-bf16-review-20261015',
                           PYTHONDONTWRITEBYTECODE='1')
        pool.run_command(wrapper, evidence / 'supervisor', environment)
        subprocess.run([*command, '--verify-only'], check=True)
        pool.save_json(evidence / 'source-readonly-postverify.json', {
            'verified': True, 'epoch': time.time(), 'source': str(source),
            'snapshot_manifest': str(evidence / 'source-snapshot.json'),
        })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--container-build', action='store_true')
    parser.add_argument('--main-authorized-build', action='store_true')
    args = parser.parse_args()
    if args.container_build:
        pool.container_build(COMPONENT, VERSION)
    elif args.main_authorized_build:
        host_build()
    else:
        parser.error('Explicit coordinator build authorization is required; preparation alone never starts compilation')


if __name__ == '__main__':
    main()
