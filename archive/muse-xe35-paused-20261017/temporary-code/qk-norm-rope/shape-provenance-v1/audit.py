import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
ARTIFACT = ROOT.parent
TARGET = ARTIFACT.parents[1] / 'pr/muse-qk-norm-rope-xe35'
MODEL_REPO = Path('/home/syk/sglang')
MODEL = MODEL_REPO / 'sglang_995c774/sglang_meta_common/sglang_meta_common/models/muse_spark.py'
CONFIG = MODEL_REPO / 'artifacts/muse-runtime-20261017/weights/Muse-Spark-1.2-820B-A42B-MXFP4-L3/config.json'
RUNTIME = MODEL_REPO / 'artifacts/muse-runtime-20261017/results'
TRACE_DERIVED = Path('/home/syk/intel_workspace/model_enablement/profiling/results/profile_muse_spark_l3_tp1_fmha-fbd1d80-20260910_b8_i4096_o16_serving')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def calls(data):
    return [ast.dump(node, include_attributes=False) for node in ast.walk(ast.parse(data))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'fused_qk_norm_rope']


def main():
    assert set(os.sched_getaffinity(0)) == set(range(32, 48))
    git = ['git', '-c', f'safe.directory={MODEL_REPO}', '-C', str(MODEL_REPO)]
    head = subprocess.check_output([*git, 'rev-parse', 'HEAD'], text=True).strip()
    committed = subprocess.check_output([*git, 'show', f'HEAD:{MODEL.relative_to(MODEL_REPO)}'])
    assert calls(committed) == calls(MODEL.read_bytes()) and len(calls(committed)) == 1
    config = json.loads(CONFIG.read_text())
    assert {name: config[name] for name in ('head_dim', 'num_attention_heads', 'num_key_value_heads', 'use_attn_o_gate')} == {
        'head_dim': 64, 'num_attention_heads': 128, 'num_key_value_heads': 16, 'use_attn_o_gate': True}
    runtime_provenance = json.loads((RUNTIME / 'provenance.json').read_text())
    archived_config = Path('/home/syk/muse-container-root/Muse-Spark-1.2-820B-A42B-MXFP4-L3/config.json')
    assert digest(CONFIG) == digest(archived_config) == runtime_provenance['sha256'][str(archived_config)]
    server = json.loads((RUNTIME / 'tp1/server-info.json').read_text())
    assert server['tp_size'] == server['ep_size'] == 1
    derived = json.loads((TRACE_DERIVED / 'buckets.json').read_text())
    step_labels = [name for name in derived['steps'] if name != '<outside step>']
    projection_shapes = []
    for filename in ('gemm-decode.txt', 'gemm-extend.txt'):
        for line in (TRACE_DERIVED / filename).read_text().splitlines():
            if '18432' in line:
                projection_shapes.append({'file': filename, 'shape_text': line[line.index('[['):]})
    frozen = json.loads((ARTIFACT / 'canonical-source-freeze-v4-icpx.json').read_text())
    preserved = {}
    for name in ('python/sgl_kernel/elementwise.py', 'python/sgl_kernel/__init__.py', 'src/torch_extension_sycl.cc',
                 'include/sgl_kernel_ops.h', 'src/sycl/FusedQKNormRope.cpp',
                 'src/sycl/xe35/FusedQKNormRope.cpp', 'src/sycl/xe35/FusedQKNormRopeKernel.h',
                 'src/sycl/xe35/FusedQKNormRopeFrequency.h', 'tests/test_fused_qk_norm_rope.py'):
        preserved[name] = digest(TARGET / name)
        assert preserved[name] == frozen['files'][name]['sha256']
    source_accuracy_audit = json.loads((ARTIFACT / 'source-accuracy-v1/final-evidence-audit.json').read_text())
    assert digest(ARTIFACT / 'source_accuracy_v1.py') == source_accuracy_audit['source_accuracy_script_sha256']
    preopt = ARTIFACT / 'source-accuracy-v1/fresh-preopt/accuracy-t32768'
    assert preopt.with_suffix('.exit').read_text().strip() == '1'
    assert json.loads(preopt.with_suffix('.json').read_text())['violations'] == 3508
    inspected = [MODEL, CONFIG, archived_config, RUNTIME / 'provenance.json', RUNTIME / 'tp1/server-info.json',
        RUNTIME / 'tp1/server-console.log', MODEL_REPO / 'python/sglang/kernels/ops/attention/fused_qknorm_rope.py',
        MODEL_REPO / 'sglang_995c774/sglang_meta_common/sglang_meta_common/layers/muse_spark_layers.py',
        TRACE_DERIVED / 'buckets.json', TRACE_DERIVED / 'gemm-decode.txt', TRACE_DERIVED / 'gemm-extend.txt',
        ARTIFACT / 'source_accuracy_v1.py', ARTIFACT / 'normal-v4-accuracy-only-v1/driver.py']
    payload = {'actual_exit': 0, 'scope': 'bounded CPU-only shape provenance; no GPU/build/performance',
        'model_repo_HEAD': head, 'model_current_file_sha256': digest(MODEL),
        'model_committed_file_sha256': hashlib.sha256(committed).hexdigest(),
        'model_full_file_clean_at_HEAD': MODEL.read_bytes() == committed,
        'QK_model_invocation_AST_same_as_HEAD': True,
        'model_config': {name: config.get(name) for name in ('num_hidden_layers', 'head_dim', 'num_attention_heads',
            'num_key_value_heads', 'use_attn_o_gate', 'use_qk_norm', 'rope_theta', 'rms_norm_eps', 'every_n_layers_nope')},
        'actual_serving_metadata': {name: server.get(name) for name in ('tp_size', 'ep_size', 'dtype', 'device',
            'chunked_prefill_size', 'disable_radix_cache', 'model_path')},
        'derived_trace_analysis_path': str(TRACE_DERIVED / 'buckets.json'),
        'historical_raw_trace_path_recorded_by_derived_analysis': derived['trace'],
        'raw_trace_directly_available_at_recorded_path': Path(derived['trace']).is_file(),
        'historical_B8_step_labels_only_no_timings': step_labels,
        'historical_B8_projection_shapes_only_no_timings': projection_shapes,
        'QKV_projection_columns': {'Q': 8192, 'K': 1024, 'V': 1024, 'O_gate': 8192, 'whole_QKVO': 18432},
        'untouched_tail_heads': {'physical_V_heads': 16, 'O_gate_head_equivalent_units': 128, 'API_tail_stride_heads': 144},
        'tokens8_assessment': 'Matches full B8 decode token count; synthetic positions0..7 and random activations are NOT actual model invocation replay',
        'tokens32768_assessment': 'B8*4096 aggregate-prefill representative shape; available derived B8 history splits12288+20480, NOT evidence of single T32768 op launch',
        'all_production_invocations_verified': False,
        'existing_SOURCE_reference_script_preserved': True, 'preopt3508_actual_exit1_preserved': True,
        'frozen_generic_BMG_API_math_tests_file_SHA_preserved': preserved,
        'inspected_file_SHA': {str(path): digest(path) for path in inspected}}
    (ROOT / 'evidence.json').write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps({name: value for name, value in payload.items() if name not in ('inspected_file_SHA', 'historical_B8_step_labels_only_no_timings', 'frozen_generic_BMG_API_math_tests_file_SHA_preserved')}, indent=2))


if __name__ == '__main__':
    main()
