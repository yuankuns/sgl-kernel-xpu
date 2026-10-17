import ast
import hashlib
import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def main():
    reports = {
        'original_and_representative': ROOT / 'SIMPLE_NORMAL_ICPX_ACCURACY_AUDIT_v4.json',
        'memory_position_semantic': ROOT / 'memory-position-model-config-accuracy-v1-run1/independent-bitwise-audit.json',
        'embedding_mask_semantic': ROOT / 'logits-model-config-accuracy-v2-run1/bitwise-audit.json',
    }
    for path in reports.values():
        assert json.loads(path.read_text())['actual_exit'] == 0
    drivers = [ROOT / name for name in ('benchmark_memory_position_model_config_v1.py',
               'benchmark_logits_model_config_v2.py', 'benchmark_simple_canonical_v1.py', 'benchmark_simple.py')]
    for path in drivers:
        ast.parse(path.read_text())
    provenance_path = ROOT / 'SIMPLE_MODEL_CONFIG_PROVENANCE_v1.json'
    provenance = json.loads(provenance_path.read_text())
    assert digest(Path(provenance['model_config'])) == provenance['model_config_sha256']
    components = {}
    for name, semantic_count, representative_count in (('memory', 21, 5), ('position', 34, 4), ('logits-vocab', 6, 5)):
        stage = ROOT / name / 'canonical-v4-icpx'
        comparator_path = ROOT / name / 'canonical-validation-v4-icpx/source-comparator-manifest.json'
        comparator = json.loads(comparator_path.read_text())
        components[name] = {
            'semantic_cases': semantic_count, 'separate_representative_cases': representative_count,
            'package_manifest_sha256': digest(stage / 'source-binary-manifest.json'),
            'compiler_policy_sha256': digest(stage / 'compiler-policy-verified.json'),
            'source_freeze_sha256': digest(ROOT / name / 'canonical-source-freeze-v4-icpx.json'),
            'source_comparator_manifest_sha256': digest(comparator_path),
            'source_native_manifest_sha256': comparator['source_native_manifest_sha256'],
        }
    report = {
        'version': 'simple-model-performance-preparation-v1', 'epoch': time.time(), 'actual_exit': 0,
        'CPU_only': True, 'GPU_authorized': False, 'native_compile_authorized': False,
        'accuracy_audits_sha256': {name: digest(path) for name, path in reports.items()},
        'driver_sha256': {path.name: digest(path) for path in drivers},
        'model_config_sha256': provenance['model_config_sha256'], 'components': components,
        'scope': {'serving_batch': 8, 'isl': 4096, 'osl': 16, 'tp': 1, 'ep': 1,
                  'claim': 'Configuration-matched synthetic registered-kernel contracts, not framework allocator/raw replay/whole model'},
        'future_gpu': 7, 'future_CPU_affinity': list(range(112, 128)),
        'protocol': {'rounds': 3, 'provider_order': ['before', 'after', 'after', 'before'],
                     'samples': 101, 'graph_calls': 128, 'eager_warmup': 5, 'graph_warmup': 5,
                     'initial_wrapper_cooling_seconds': 5, 'synchronized_shape_cooling_seconds': 2,
                     'semantic_cases_per_provider_phase': 61, 'separate_representative_cases_per_provider_phase': 14,
                     'capture_or_reference_failure': 'retain actual failure; no fallback or timing acceptance',
                     'raw_samples_and_role_drift_retained': True, 'outlier_deletion': False},
        'future_requirements': [
            'Residual and routing stable7 owners physically drained; new explicit main handoff and future grant',
            'Maintenance quiet and loaded-stack checks; no compiler activity during measured interval',
            'Fresh normal-package/provider/native inventory CPU gates and identical shared input witnesses',
            'Actual capture node/op evidence before event timing; no fake replay or registration',
            'Full per-case independent exact reference and cross-provider saved output comparison',
            'Separate semantic versus representative report; softcap20 and embedding4096 are not active model settings',
            'No summation into whole-model timing or eight-card claim without independent evidence',
        ],
        'future_runner_implemented': False, 'final_performance_acceptance_complete': False,
    }
    with (ROOT / 'SIMPLE_MODEL_PERFORMANCE_CPU_PLAN_v1.json').open('x') as output:
        output.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'actual_exit': 0, 'semantic_cases': 61, 'representative_cases': 14,
                      'GPU_authorized': False, 'plan_sha256': digest(ROOT / 'SIMPLE_MODEL_PERFORMANCE_CPU_PLAN_v1.json')}))


if __name__ == '__main__':
    main()
