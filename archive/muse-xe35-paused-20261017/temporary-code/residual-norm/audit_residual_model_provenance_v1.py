import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time


ROOT = Path(__file__).resolve().parent
FRAMEWORK = Path('/home/syk/sglang')
BASE = FRAMEWORK / 'sglang_995c774/sglang_meta_common/sglang_meta_common'
CONFIG = FRAMEWORK / 'artifacts/muse-runtime-20261017/weights/Muse-Spark-1.2-820B-A42B-MXFP4-L3/config.json'
TRACE = Path('/home/syk/intel_workspace/model_enablement/profiling/results/profile_muse_spark_l3_tp1_fmha-fbd1d80-20260910_b8_i4096_o16_serving')
files = {'model_config': CONFIG, 'model': BASE / 'models/muse_spark.py',
         'config_class': BASE / 'models/muse_spark_config.py', 'layer_math': BASE / 'layers/muse_spark_layers.py',
         'model_adapter': BASE / 'ops/fused_norm_residual.py', 'historical_B8_buckets': TRACE / 'buckets.json',
         'historical_B8_scope': TRACE / 'profile-summary.jsonl',
         'source_benchmark': Path('/home/syk/worktrees/sgl-kernel-xpu/muse-spark-residual-norm-main-20261014/benchmark/bench_fused_norm_residual.py'),
         'frozen_port_benchmark': ROOT.parent.parent / 'pr/muse-residual-norm-xe35/benchmark/bench_fused_norm_residual.py',
         'official_registration': ROOT.parent.parent / 'pr/muse-residual-norm-xe35/src/torch_extension_sycl.cc'}
config = json.loads(CONFIG.read_text())
assert config['hidden_size'] == 8192 and config['num_hidden_layers'] == 3
assert config['rms_norm_eps'] == 1e-5 and config['rms_post_norm_eps'] == 1e-8
assert config['norm_zero_centered_gamma'] is True and config['post_norm_affine'] is False
assert config['use_ffn_learnable_scales'] is True and config['use_residual_gate'] is True
assert config['residual_dtype'] == 'fp32' and config['moe']['router_in_fp32'] is True
buckets = json.loads(files['historical_B8_buckets'].read_text())
steps = [name for name in buckets['steps'] if name != '<outside step>']
waves = [dict(tokens=int(match.group(2)), wave_requests=int(match.group(1)), step=name)
         for name in steps if (match := re.search(r'EXTEND bs=(\d+) toks=(\d+)', name))]
assert {(entry['wave_requests'], entry['tokens']) for entry in waves} == {(3, 12288), (5, 20480)}
history = json.loads(files['historical_B8_scope'].read_text().splitlines()[0])
assert history['max_concurrency'] == 8 and history['random_input_len'] == 4096 and history['random_output_len'] == 16
server = history['server_info']
assert server['tp_size'] == 1 and server['ep_size'] == 1
locations = {}
for key in ['model', 'config_class', 'layer_math', 'model_adapter']:
    tree = ast.parse(files[key].read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for method in node.body:
                if isinstance(method, ast.FunctionDef):
                    locations[f'{node.name}.{method.name}'] = {'file': str(files[key]), 'line': method.lineno}
adapter = files['model_adapter'].read_text()
registration = files['official_registration'].read_text()
assert 'torch.ops.sgl_kernel.fused_post_norm_residual_pre_norm\n' in adapter
assert 'fused_post_norm_residual_pre_norm_xe35' not in adapter
assert 'm.impl("fused_post_norm_residual_pre_norm_xe35"' in registration
assert 'm.impl("fused_post_norm_residual_pre_norm"' not in registration
roles = [
    {'role': 'attention_boundary', 'invocations_per_model_forward': 3, 'post_mode': 'none',
     'post_weight_empty': True, 'pre_effective_gain_center': 1, 'round_pre_norm_input': False,
     'decode_T8_output_fp32': True, 'prefill_T12288_T20480_output_fp32': False},
    {'role': 'inter_layer_FFN_boundary', 'invocations_per_model_forward': 2, 'post_mode': 'input_scale',
     'post_effective_gain_center': 1, 'pre_effective_gain_center': 1, 'round_pre_norm_input': False,
     'output_fp32': False},
    {'role': 'last_layer_FFN_to_output_norm', 'invocations_per_model_forward': 1, 'post_mode': 'input_scale',
     'post_effective_gain_center': 1, 'pre_effective_gain_center': 0, 'round_pre_norm_input': True,
     'output_fp32': False}]
record = {
    'epoch': time.time(), 'scope': 'CPU-only bounded model/config/callpath provenance; no Torch import or GPU',
    'serving_scope': {'B': 8, 'ISL': 4096, 'OSL': 16, 'TP': 1, 'EP': 1},
    'framework_HEAD': subprocess.check_output(['git', '-C', str(FRAMEWORK), 'rev-parse', 'HEAD'], text=True).strip(),
    'framework_whole_file_clean_not_claimed': True,
    'authoritative_files': {key: {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                            for key, path in files.items()},
    'config_parameters': {name: config.get(name) for name in ['hidden_size', 'num_hidden_layers', 'rms_norm_eps',
        'rms_post_norm_eps', 'norm_zero_centered_gamma', 'norm_affine', 'post_norm_affine',
        'post_norm_gain_center_type', 'output_norm_gain_center_type', 'use_ffn_learnable_scales',
        'use_residual_gate', 'residual_dtype', 'numerics_args']},
    'dtype_source': {'config_dtype_and_torch_dtype_absent': 'dtype' not in config and 'torch_dtype' not in config,
        'config_class_default': 'bfloat16 when both dtype and torch_dtype absent',
        'model_branch_argument': 'torch.bfloat16', 'initial_stream_cast': 'config.residual_dtype -> torch.float32',
        'kernel_cached_post_pre_weights': 'effective_weight.to(branch_dtype).contiguous -> BF16',
        'kernel_cached_alpha_beta': 'reshape(-1).float().contiguous -> FP32',
        'raw_checkpoint_weight_values_dtypes_not_inspected': True},
    'effective_weight_semantics': {'formula': 'raw gamma + gain_center only if zero_centered_gamma; then cache/cast',
        'attention_post': 'empty, post_norm_affine false', 'ordinary_pre_and_FFN_post_gain_center': 1,
        'last_output_pre_gain_center': 0, 'kernel_does_not_receive_zero_centered_flag': True,
        'random_benchmark_weight_is_direct_kernel_effective_input_not_verified_raw_gamma': True},
    'residual_gate_coefficients': {'temperature': 0.3, 'alpha_sq_floor_eps': 1e-3,
        'beta': 'sigmoid(gate.float()/temperature)',
        'alpha': 'sqrt(clamp(sigmoid(-gate.float()/temperature)*(1+beta), min=1e-3))',
        'current24_alpha_beta_independent_random_not_model_coefficients': True},
    'router_widen_semantics': {'router_weight_dtype_from_config': 'FP32', 'split_min_tokens': 512,
        'attention_output_fp32': 'router.weight.dtype == FP32 and T < 512',
        'actual_widen_values': 'output_BF16.float(), not unrounded FP32 pre-norm output'},
    'actual_model_boundary_roles': roles,
    'historical_B8_shape_labels_only': {'prefill_waves': waves, 'decode_rows': 8,
        'aggregate32768_not_actual_single_prefill_launch': True,
        'wave_requests3and5_not_new_serving_B3_or_B5': True,
        'no_historical_timing_used': True, 'raw_trace_path': buckets['trace'],
        'raw_residual_input_activation_tensors_available': False},
    'current24_classification': 'representative direct kernel API contract precision coverage, not all actual model invocation acceptance',
    'matches_current24': ['H8192', 'post/pre eps1e-8/1e-5', 'BF16 branch/post/pre/output; FP32 stream/alpha/beta/widen',
        'decode rows8', 'none/input_scale and rounded-final/widened-decode modes supported'],
    'gaps': ['actual prefill12288/20480 absent', 'effective gamma+gain cache/cast not exercised through real adapter',
        'gate-derived correlated alpha/beta not exercised', 'actual roles/mode selection not exercised',
        'current model adapter binds old unsuffixed op; official package new op only _xe35',
        'no checkpoint activations/weights raw replay or full model trace', 'genuine AFTER accuracy case18 currently failed'],
    'actual_framework_AFTER_callpath_verified': False, 'no_fake_binding_or_monkeypatch': True,
    'source_locations': locations, 'original_sources_refs_raw_inputs_modified': False,
    'kernel_helper_math_flags_modified': False, 'new_GPU_or_compiler_started': False,
    'normal_AFTER_precision_accepted': False, 'formal_performance_grant_present': False,
    'next_supplement_scope_if_authorized': 'separate immutable actual-config fixtures for three boundary roles at T8/T12288/T20480, real guarded framework adapter; retain all original24/oracles/raw/precisiontests',
    'supplement_GPU_not_started': True}
destination = ROOT / 'RESIDUAL_MODEL_CONFIG_PROVENANCE_v1.json'
patch = '*** Begin Patch\n*** Add File: ' + str(destination) + '\n'
patch += ''.join('+' + line + '\n' for line in json.dumps(record, indent=2).splitlines())
subprocess.run(['apply_patch', patch + '*** End Patch\n'], check=True)
print(json.dumps({'metadata': str(destination), 'config_H': 8192, 'eps': [1e-8, 1e-5],
                  'boundary_roles': roles, 'gaps': record['gaps'], 'GPU_started': False}, indent=2))
