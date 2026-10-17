import ast
import hashlib
import json
from pathlib import Path
import time
from typing import Optional


ART = Path(__file__).resolve().parent
MODEL_ROOT = Path('/home/syk/sglang/sglang_995c774/sglang_meta_common/sglang_meta_common/models')
FRAMEWORK = Path('/home/syk/sglang/python/sglang/srt')
PROFILE = Path('/home/syk/intel_workspace/model_enablement/profiling/results/profile_muse_spark_l3_tp1_fmha-fbd1d80-20260910_b8_i4096_o16_serving')
CONFIG = Path('/home/syk/sglang/artifacts/muse-runtime-20261017/weights/Muse-Spark-1.2-820B-A42B-MXFP4-L3/config.json')
ENTRY = ART.parent.parent / 'pr/muse-fmha-xe35/src/sycl/xe35/fmha/entry.cpp'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    config = json.loads(CONFIG.read_text())
    model = MODEL_ROOT / 'muse_spark.py'
    tree = ast.parse(model.read_text())
    namespace = {'Optional': Optional}
    for name in ('_compute_use_rope', '_compute_sliding_window'):
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(model), 'exec'), namespace)
    layers = []
    for layer in range(config['num_hidden_layers']):
        window = namespace['_compute_sliding_window'](layer, config['num_hidden_layers'], config['every_n_layers_nope'], config['global_attn_cfg'])
        rope = namespace['_compute_use_rope'](layer, config['num_hidden_layers'], config['every_n_layers_nope'], config['use_rope'])
        layers.append({'layer': layer, 'window_left': window if window is not None else -1,
                       'window_right_model_call': 0 if window is not None else -1, 'causal': True, 'use_rope': rope})
    assert layers == [{'layer': 0, 'window_left': 2047, 'window_right_model_call': 0, 'causal': True, 'use_rope': True},
                      {'layer': 1, 'window_left': 2047, 'window_right_model_call': 0, 'causal': True, 'use_rope': True},
                      {'layer': 2, 'window_left': -1, 'window_right_model_call': -1, 'causal': True, 'use_rope': False}]
    summary = json.loads((PROFILE / 'profile-summary.jsonl').read_text().splitlines()[0])
    buckets = json.loads((PROFILE / 'buckets.json').read_text())
    assert summary['completed'] == summary['max_concurrency'] == 8
    assert summary['random_input_len'] == 4096 and summary['random_output_len'] == 16
    server = summary['server_info']
    assert server['tp_size'] == server['ep_size'] == 1 and server['page_size'] == 64
    assert server['disable_radix_cache'] and server['speculative_algorithm'] is None
    assert '001 step[EXTEND bs=3 toks=12288]' in buckets['steps']
    assert '002 step[EXTEND bs=5 toks=20480]' in buckets['steps']
    model_config = MODEL_ROOT / 'muse_spark_config.py'
    assert 'kwargs["dtype"] = "bfloat16"' in model_config.read_text()
    assert 'pre_attn_norm(stream).to(torch.bfloat16)' in model.read_text()
    assert config['num_attention_heads'] == 128 and config['num_key_value_heads'] == 16 and config['head_dim'] == 64
    assert '(batch_xe35 == 1 || batch_xe35 == 7 || batch_xe35 == 8)' in ENTRY.read_text()
    files = [CONFIG, model, model_config, FRAMEWORK / 'layers/radix_attention.py', FRAMEWORK / 'layers/attention/xpu_backend.py',
             FRAMEWORK / 'model_executor/forward_batch_info.py', PROFILE / 'buckets.json', PROFILE / 'profile-summary.jsonl', ENTRY,
             ART.parent.parent / 'pr/muse-fmha-xe35/benchmark/bench_muse_fmha.py']
    mappings = []
    for requests in (1, 7, 8):
        for window in (-1, 2047):
            mappings.append({'fixture': f'prefill-wave{requests}-w{window}', 'serving_batch': 8,
                'fixture_active_sequences': requests, 'fixture_query_tokens': requests * 4096,
                'query_positions': [0, 4095], 'cache_lengths': [4096] * requests, 'prefix_lengths': [0] * requests,
                'window_left': window, 'window_right_fixture': 0, 'causal': True,
                'classification': 'representative equal-length synthetic kernel shape; not observed B8 3/5 prefill wave',
                'model_layers_matching_mask': [item['layer'] for item in layers if item['window_left'] == window]})
    for step in range(1, 17):
        for window in (-1, 2047):
            mappings.append({'fixture': f'decode-osl{step}-w{window}', 'serving_batch': 8, 'active_sequences': 8,
                'cache_lengths': [4096 + step] * 8, 'query_position': 4095 + step,
                'window_left': window, 'window_right_fixture': 0, 'causal': True,
                'classification': 'semantic reconstruction for generated output index1..15, not raw activation replay' if step <= 15 else
                                  'additional end-boundary kernel shape; OSL16 includes first output from prefill and needs only15 decode forwards'})
    proposed = []
    for wave, (requests, first_slot) in enumerate(((3, 0), (5, 3)), start=1):
        for layer in layers:
            proposed.append({'name': f'model-prefill-wave{wave}-active{requests}-layer{layer["layer"]}',
                'serving_batch': 8, 'active_sequences': requests, 'synthetic_request_slots': list(range(first_slot, first_slot + requests)),
                'query_shape': [requests * 4096, 128, 64], 'paged_KV_pool_shape': [512, 64, 16, 64],
                'page_table_shape': [requests, 64], 'cu_seqlens_q': [request * 4096 for request in range(requests + 1)],
                'cache_lengths': [4096] * requests, 'positions': [0, 4095], 'prefix_lengths': [0] * requests,
                'softmax_scale': 0.125, 'dtype': 'BF16', **layer,
                'frozen_port_route': 'upstream fallback; optimized production guard excludes active3/5', 'GPU_run_authorized': False})
    for output_index in range(1, 16):
        for layer in layers:
            proposed.append({'name': f'model-decode-output{output_index}-layer{layer["layer"]}', 'serving_batch': 8,
                'active_sequences': 8, 'query_shape': [8, 128, 64], 'cache_lengths': [4096 + output_index] * 8,
                'position': 4095 + output_index, 'page_size': 64, 'cu_seqlens_q': list(range(9)),
                'softmax_scale': 0.125, 'dtype': 'BF16', **layer, 'GPU_run_authorized': False})
    report = {'epoch': time.time(), 'authoritative_files_sha256': {str(path): digest(path) for path in files},
        'model_entry': 'MuseSpark1xDecoderLayer.forward_to_ffn_output -> self_attn_input_norm(BF16) -> MuseSpark1xAttention.forward(QK norm/RoPE) -> RadixAttention.forward -> XPUAttentionBackend.forward_extend/forward_decode -> normal sgl_kernel.flash_attn.flash_attn_with_kvcache -> registered torch.ops.sgl_kernel.fwd',
        'parameters': {'serving_batch': 8, 'isl': 4096, 'osl': 16, 'tp': 1, 'ep': 1, 'heads_q': 128, 'heads_kv': 16, 'head_dim': 64,
            'page_size': 64, 'attention_dtype': 'BF16 from actual config class default and model BF16 branch, not JSON dtype:auto inference',
            'softmax_scale': 0.125, 'q_norm_eps': config['rms_norm_eps'], 'q_scale': 43.7840518911 / 8,
            'rope_theta': config['rope_theta'], 'layers': layers},
        'derived_observation': {'prefill_active_sequences': [3, 5], 'prefill_tokens': [12288, 20480], 'serving_batch': 8,
            'profiled_decode_steps': len([name for name in buckets['steps'] if 'DECODE' in name]),
            'raw_trace_recorded_path': buckets['trace'], 'raw_trace_readable_at_recorded_path': Path(buckets['trace']).is_file()},
        'old_fixture_mappings': mappings, 'old_raw_math_inputs_PR_changed': False,
        'actual_production_prefill_optimized_dispatch_gap': 'current frozen entry only active1/7/8; observed3/5 falls back to genuine upstream path. Needs main-reviewed dispatch/freeze change before optimized real-wave claim; no silent PR edit.',
        'gaps': ['No readable historical raw trace at recorded path; derived buckets show3/5 and8decode profile steps only.',
            'No real per-request Q/K/V activations, allocator page identities, request-order or positions tensor replay; fixture Gaussian inputs are synthetic.',
            'Serving OSL16 gets first generated token from prefill;15 actual decode forwards position4096..4110; fixture16th decode is boundary extension.',
            'Current model opts hybrid SWA; XPU backend may translate full to SWA-pool page table. Synthetic dense/random pool does not replay allocator eviction/remapping.',
            'Proposed8-slot request ids/order are explicit reconstruction, not captured trace; model-config matching cases require separate approval.'],
        'proposed_model_semantic_cases_CPU_only': proposed, 'proposed_case_count': len(proposed),
        'production_math_no_kernel_changes': True, 'no_full_trace_replay_claim': True}
    (ART / 'MODEL_FIXTURE_PROVENANCE_v1.json').write_text(json.dumps(report, indent=2) + '\n')
    (ART / 'MODEL_CONFIG_MATCHING_CPU_CASE_PLAN_v1.json').write_text(json.dumps(proposed, indent=2) + '\n')
    print(json.dumps({'old_fixture_count': len(mappings), 'actual_prefill_active_sequences': [3, 5],
                      'proposed_serving_batch': 8, 'proposed_CPU_case_count': len(proposed),
                      'frozen_dispatch_prefill_gap': True, 'raw_replay_claimed': False, 'GPU_launched': False}))


if __name__ == '__main__':
    main()
