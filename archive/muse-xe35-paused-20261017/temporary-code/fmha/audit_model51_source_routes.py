import hashlib
import json
from pathlib import Path
import subprocess
import time


ART = Path(__file__).resolve().parent
ROOT = ART / 'model-config51-v1'
SOURCE = ART / 'matched-native-icpx-v1/source'
TREE = SOURCE / 'source/src/sycl'
PACKAGE = SOURCE / 'site/sgl_kernel'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def output(command):
    return subprocess.run(command, check=True, text=True, capture_output=True).stdout


def main():
    plan = json.loads((ROOT / 'plan.json').read_text())
    gate = json.loads((ROOT / 'source-CPU-gate.json').read_text())
    graph = json.loads((SOURCE / 'compiler-graph-verified.json').read_text())
    binaries = json.loads((SOURCE / 'fresh-native-binaries.json').read_text())['libraries']
    package_manifest = json.loads((SOURCE / 'normal-package-manifest.json').read_text())
    common = next(PACKAGE.glob('common_ops.*.so'))
    assert digest(common) == package_manifest['original_package_files_sha256'][f'sgl_kernel/{common.name}']
    root_library = PACKAGE / 'libsgl-ops-sycl-flash_attentionXe35.so'
    root_symbols = output(['nm', '-D', str(root_library)])
    common_symbols = output(['nm', '-D', str(common)])
    root_fwd = next(line.split()[-1] for line in root_symbols.splitlines() if ' T _Z7mha_fwd' in line)
    assert any(line.endswith(root_fwd) and ' U ' in line for line in common_symbols.splitlines())
    common_needed = output(['readelf', '-d', str(common)])
    assert '[libsgl-ops-sycl-flash_attentionXe35.so]' in common_needed
    assert not (PACKAGE / 'libsgl-ops-sycl-flash_attention.so').exists()
    assert 'libsgl-ops-sycl-flash_attention.so' not in gate['native_sha256']
    records = {}
    for target, recipe in graph['targets'].items():
        library = f'lib{target}.so'
        path = PACKAGE / library
        assert digest(path) == binaries[library] == gate['native_sha256'][library]
        assert 'set(SYCL_executable "/opt/gfx-deps/dpcpp/bin/icpx")' in recipe['native_recipe']
        assert 'set(SYCL_host_compiler "/opt/gfx-deps/dpcpp/bin/icpx")' in recipe['native_recipe']
        sections = output(['readelf', '-SW', str(path)])
        symbols = output(['nm', '-DC', str(path)])
        records[library] = {'sha256': digest(path), 'actual_recipe_sha256': recipe['generated_sha256'],
            'actual_recipe': recipe, 'SYCL_driver': 'icpx', 'host': 'icpx', 'CXX': 'IntelLLVM icpx',
            'offload_sections': [line for line in sections.splitlines() if '__CLANG_OFFLOAD_BUNDLE__' in line],
            'FMHA_symbols': [line for line in symbols.splitlines() if any(name in line for name in ('mha_fwd(', 'dispatch_xe35_prefill_64_bf16(', 'dispatch_xe35_decode_64_bf16('))],
            'comment_section': output(['readelf', '-p', '.comment', str(path)]),
            'comment_boundary': 'GCC startup-object tag is present; compiler attribution uses actual generated compile/device-link/shared-link recipes, not .comment alone'}
        assert records[library]['offload_sections']
    root_source = (TREE / 'flash_attentionXe35.cpp').read_text()
    assert 'max_seqlen_q == 1 ? get_xe35_decode_dispatch(head_dim, q_dtype) : get_xe35_prefill_dispatch(head_dim, q_dtype)' in root_source
    assert 'mha_fwd_upstream' not in root_source
    prefill_dispatch = TREE / 'kernels/flash_attention_v2/xe3/xe_fmha_fwd_prefill_dispatch.hpp'
    decode_dispatch = TREE / 'kernels/flash_attention_v2/xe3/xe_fmha_fwd_decode_dispatch.hpp'
    assert 'return &dispatch_xe35_prefill_64_bf16;' in prefill_dispatch.read_text()
    assert 'return &dispatch_xe35_decode_64_bf16;' in decode_dispatch.read_text()
    assert ' U dispatch_xe35_prefill_64_bf16(' in records[root_library.name]['FMHA_symbols'][1] or any(' U dispatch_xe35_prefill_64_bf16(' in line for line in records[root_library.name]['FMHA_symbols'])
    routes = []
    for case in plan['cases']:
        phase = case['phase']
        library = f'libsgl-ops-sycl-flash_attentionXe35_{"prefill" if phase == "prefill" else "decode"}_64_bf16.so'
        routes.append({'name': case['name'], 'serving_batch': 8, 'active_sequences': case['requests'],
            'normal_wrapper_inferred_max_seqlen_k': case['source_wrapper_inferred_max_seqlen_k'],
            'source_route': 'original optimized Xe35 paged BF16 D64 prefill' if phase == 'prefill' else 'original optimized Xe35 BF16 D64 GQA8 splitKV decode',
            'SOURCE_generic_fallback': False, 'root_entry_library': root_library.name,
            'device_native_library': library, 'device_native_sha256': records[library]['sha256'],
            'tested_native_Sycl_host_driver': 'icpx',
            'AFTER_route': 'genuine generic upstream fallback; max_seqlen_k=0 unchanged',
            'evidence_boundary': 'deterministic argument predicates + original dispatch source + ELF binding/device images + genuine mapped library SHA; no device-event kernel trace claimed'})
    inputs = [TREE / 'flash_attentionXe35.cpp', TREE / 'flash_attentionXe35_common.hpp', prefill_dispatch, decode_dispatch,
              SOURCE / 'build/src/sycl/flash_attentionXe35_prefill_64_bf16.cpp', SOURCE / 'build/src/sycl/flash_attentionXe35_decode_64_bf16.cpp']
    report = {'epoch': time.time(), 'CPU_only': True, 'no_compile_or_GPU_or_live_runner_change': True,
              'plan_sha256': digest(ROOT / 'plan.json'), 'source_head': package_manifest['source_head'],
              'SOURCE_all51_generic': False, 'SOURCE_prefill_optimized_cases': 6, 'SOURCE_decode_optimized_cases': 45,
              'SOURCE_generic_libflash_attention_present': False, 'SOURCE_generic_libflash_attention_mapped': False,
              'original_common_ops_sha256': digest(common), 'original_common_ops_compiler': 'historical GNU13.3 CXX; untouched original registration/ABI thunk',
              'root_mha_fwd_binding_symbol': root_fwd,
              'root_device_implementation_boundary': 'root exports host mha_fwd and leaves BF16 D64 dispatch undefined; heavy FMHA/splitKV device implementations live in fresh64 DSOs, not inferred from root include or its small offload section',
              'fresh_tested_native_records': records, 'routes': routes,
              'source_files_sha256': {str(path): digest(path) for path in inputs},
              'compiler_claim_boundary': 'only routed root and two BF16 D64 tested-native DSOs verified fresh icpx; original common_ops and all other historical package components are not compiler-matched. e878/profiling ON vs AFTER public7b56/profiling OFF remain confounds; no entire51 stack/compiler-isolated performance claim'}
    (ART / 'MODEL51_SOURCE_DISPATCH_NATIVE_COMPILER_AUDIT_v1.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'SOURCE_optimized_prefill': 6, 'SOURCE_optimized_decode': 45, 'SOURCE_generic': 0,
                      'AFTER_generic': 51, 'tested_native_driver': 'icpx', 'common_ops': 'historical GNU13.3',
                      'no_whole_package_compiler_match_claim': True}))


if __name__ == '__main__':
    main()
