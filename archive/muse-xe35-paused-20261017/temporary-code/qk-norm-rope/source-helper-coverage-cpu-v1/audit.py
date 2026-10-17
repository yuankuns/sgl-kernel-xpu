import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
ARTIFACT = ROOT.parent
PROVIDER = ARTIFACT / 'native-icpx-cpu-v3/optimized-source'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def host(path):
    return Path(str(path).replace('/root/', '/home/syk/', 1))


def main():
    assert sorted(os.sched_getaffinity(0)) == list(range(16))
    assert 'torch' not in sys.modules
    original = read(PROVIDER / 'inputs.json')
    complete = read(PROVIDER / 'native-complete.json')
    dependencies = read(ARTIFACT / 'native-icpx-cpu-v3/original-deps.json')
    manifest = read(PROVIDER / 'site-manifest.json')
    assert original['ref'] == complete['ref'] == 'feed315712ea91e6becbb983f3a4d56ac0cc3b96'
    assert dependencies['head'] == complete['original_deps_pin'] == 'e87870bb7aacf83bfb3be9da685f148316b8df23'
    assert complete['QK_target_actual_exit'] == complete['normal_package_CPU_import_actual_exit'] == 0
    depfile = PROVIDER / 'build/src/CMakeFiles/sgl-ops-sycl-FusedQKNormRope.dir/sycl/sgl-ops-sycl-FusedQKNormRope_gen_FusedQKNormRope.cpp.o.depend'
    paths = [host(path) for path in re.findall(r'"([^"\n]+)"', depfile.read_text())]
    project = {}
    cutlass = {}
    for path in paths:
        if path.is_relative_to(PROVIDER / 'source'):
            relative = str(path.relative_to(PROVIDER / 'source'))
            actual = digest(path)
            assert actual == original['all_tracked_original_files'][relative]
            project[relative] = actual
        elif path.is_relative_to(Path(dependencies['path'])):
            relative = str(path.relative_to(Path(dependencies['path'])))
            actual = digest(path)
            assert actual == dependencies['tracked_files'][relative]
            cutlass[relative] = actual
    expected = {'FusedQKNormRope.cpp', 'FusedQKNormRopeFrequency.h', 'FusedQKNormRopeXe35.h', 'MemoryAccess.h', 'Norm.h', 'SYCLHelpers.h', 'Utils.h'}
    assert expected.issubset({Path(path).name for path in project}) and cutlass
    binaries = {}
    for name in ('libsgl-ops-sycl-FusedQKNormRope.so', 'libsgl-ops-sycl-Device.so', 'common_ops.cpython-312-x86_64-linux-gnu.so'):
        relative = 'sgl_kernel/' + name
        actual = digest(PROVIDER / 'site' / relative)
        assert actual == manifest[relative]
        binaries[relative] = {'actual_sha256': actual, 'fresh_icpx_QK_build': 'FusedQKNormRope.so' in name,
            'legacy_build_source_correspondence_NOT_proven': 'FusedQKNormRope.so' not in name}
    assert binaries['sgl_kernel/libsgl-ops-sycl-FusedQKNormRope.so']['actual_sha256'] == complete['fresh_native_sha256']
    needed = (PROVIDER / 'native-readelf-needed.log').read_text()
    assert not re.search(r'NEEDED.*libsgl', needed)
    setup_sources = {}
    for name in ('src/torch_extension_sycl.cc', 'src/sycl/Device.cpp', 'include/sgl_kernel_ops.h'):
        actual = digest(PROVIDER / 'source' / name)
        assert actual == original['all_tracked_original_files'][name]
        setup_sources[name] = actual
    ready = ARTIFACT / 'controlled-method-runner-cpu-v1/READY.json'
    old_ready_sha = '5ed20090ee1e151a1e9198d7eb4ac242fadd07e35152a4a6a2147ee0dbfa3e38'
    assert digest(ready) == old_ready_sha
    result = {'actual_CPU_audit_exit': 0, 'epoch': time.time(), 'CPU_affinity': list(range(16)),
        'source_ref': complete['ref'], 'cutlass_pin': dependencies['head'], 'compiler_depfile_sha256': digest(depfile),
        'project_transitive_compile_inputs_SHA': project, 'CUTLASS_transitive_compile_inputs_SHA': cutlass,
        'compiler_depfile_project_inputs_count': len(project), 'compiler_depfile_CUTLASS_inputs_count': len(cutlass),
        'setup_sources_SHA_NOT_legacy_binary_build_proof': setup_sources, 'actual_provider_selected_binary_SHA': binaries,
        'existing_QK_DSO_needed_log_sha256': digest(PROVIDER / 'native-readelf-needed.log'),
        'QK_DSO_no_other_project_DSO_DT_NEEDED': True,
        'SOURCE_full_all_helper_fresh_icpx_claim': False,
        'gaps': ['common_ops registration bridge retains legacy binary; ABI/functionality tested, exact build-to-feed315 correspondence not proven',
            'Device query_device setup op retains legacy binary; prior actual3.5 success does not prove fresh icpx/source correspondence',
            'Torch copy/clone/positions cast, native Torch/SYCL/UR/LevelZero core stack is external; exact current whole loaded stack identity awaits new main grant/runtime check'],
        'controlled_READY_SHA_unchanged': old_ready_sha, 'GPU_compile_ISA_metadata_import_Torch_executed': False,
        'stable_perf_grant_still_HELD': True, 'GPU6_owner': 'Mill W4', 'new_runtime_helpers_rebuild_NOT_authorized': True}
    (ROOT / 'coverage.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f'CPU coverage audit actual0: {len(project)} project compile inputs, {len(cutlass)} CUTLASS compile inputs; legacy setup gaps disclosed; READY unchanged')


if __name__ == '__main__':
    main()
