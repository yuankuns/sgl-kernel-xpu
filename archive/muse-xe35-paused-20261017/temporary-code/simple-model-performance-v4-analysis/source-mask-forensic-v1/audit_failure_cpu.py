import hashlib
import json
import sys
from pathlib import Path


DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(DIRECTORY.parent))
import analyzer


def save(path, record):
    with path.open('x') as output:
        output.write(json.dumps(record, indent=2, allow_nan=False) + '\n')


def main():
    torch = analyzer.cpu_environment()
    evidence = analyzer.Evidence()
    root = analyzer.ROOT
    run = root / analyzer.RUN_NAME
    phase = run / 'round-0-slot-0-before'
    analyzer.context(evidence, run)
    analyzer.require(evidence.file(phase / 'wrapper.exit').read_text().strip() == '1', 'original true phase failure required')
    analyzer.validate_phase_watch(evidence.json(phase / 'maintenance-watch.json'))
    completed = []
    for component in ('memory', 'position'):
        for family in analyzer.FAMILIES:
            prefix = phase / f'{component}-{family}'
            evidence.exit0(prefix.with_suffix('.exit'))
            report = evidence.json(prefix.with_suffix('.json'))
            expected = analyzer.names(component, family)
            analyzer.validate_manifest_case_set(report['results'], expected)
            for index, name in enumerate(expected):
                result = analyzer.audit_case(evidence, prefix.with_suffix('.graphs') / f'case-{index:04d}-{name}', index, name)
                analyzer.require(all(report['results'][index].get(key) == value for key, value in result.items()), 'original finalized report mismatch')
            completed.append({'component': component, 'family': family, 'cases': len(expected), 'actual_exit': 0})
    analyzer.require(sum(record['cases'] for record in completed) == 64, 'four family actual count must be64')
    prefix = phase / 'logits-vocab-semantic'
    analyzer.require(evidence.file(prefix.with_suffix('.exit')).read_text().strip() == '1', 'original logits failure absent')
    embedding = analyzer.audit_case(evidence, prefix.with_suffix('.graphs') / 'case-0000-embedding-decode8', 0, 'embedding-decode8')
    mask_dir = prefix.with_suffix('.graphs') / 'case-0001-mask-decode8'
    single = evidence.json(mask_dir / 'single-call-proof.json')
    analyzer.exact(single, {'captured_calls': 1, 'logical_calls': 1, 'DOT_nodes': 1, 'keep_graph': True, 'timing_events_created': False}, 'failed mask singleton')
    dot = evidence.file(mask_dir / 'single-call.dot', single['DOT_sha256']).read_text()
    analyzer.require(analyzer.parse_native_dot(dot) == single['node_ids'], 'failed mask singleton native nodes mismatch')
    analyzer.require('VocabMaskKernelIlEE' in dot and 'zeCommandListAppendLaunchKernelWithArguments' in dot, 'wrong actual native mask kernel')
    log = evidence.file(prefix.with_suffix('.log')).read_text()
    analyzer.require('Mismatched elements: 5 / 8' in log and 'check_mask_replay' in log, 'actual mapped failure evidence missing')
    absent = [prefix.with_suffix('.pt'), prefix.with_suffix('.json')] + [mask_dir / name for name in ('calls-128.dot', 'graph-proof.json', 'samples.jsonl', 'result.json')]
    analyzer.require(not any(path.exists() for path in absent), 'unexpected old failed output/timing evidence; review scope')
    eager = root / 'logits-model-config-accuracy-v2-run1'
    audit = evidence.json(eager / 'bitwise-audit.json')
    analyzer.exact(audit, {'actual_exit': 0}, 'previous eager audit')
    comparator = evidence.json(root / 'logits-vocab/canonical-validation-v4-icpx/source-comparator-manifest.json')
    source_native = comparator['files']['sgl_kernel/libsgl-ops-sycl-VocabMask.so']['sha256']
    eager_records = {}
    for label in ('before', 'after'):
        evidence.exit0(eager / (label + '.exit'))
        report = evidence.json(eager / (label + '.json'), audit['raw'][label]['report_sha256'])
        analyzer.exact(report, {'phase': 'accuracy', 'samples': 0, 'calls': 0}, 'eager non-timing scope')
        if label == 'before':
            analyzer.require(report['mapped_native_sha256']['sgl_kernel/libsgl-ops-sycl-VocabMask.so'] == source_native, 'eager SOURCE binary differs from failed graph SOURCE')
        path = evidence.file(eager / (label + '.pt'), audit['raw'][label]['tensor_sha256'])
        data = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        tree = data['mask-decode8']
        expected_invalid = (tree['ids'] < 0) | (tree['ids'] >= 202048)
        expected_mapped = tree['ids'] * (~expected_invalid)
        torch.testing.assert_close(tree['mapped'], expected_mapped, atol=0, rtol=0)
        torch.testing.assert_close(tree['invalid'], expected_invalid, atol=0, rtol=0)
        eager_records[label] = {name: tensor.tolist() for name, tensor in tree.items()}
    analyzer.require(eager_records['before'] == eager_records['after'], 'old eager exact SOURCE/AFTER mismatch')
    source_tree = root / 'logits-vocab/fresh-source-icpx-v1/runs/source-native-icpx-20261017-r1/source'
    original = analyzer.local_path('/root/worktrees/sgl-kernel-xpu/muse-spark-logits-vocab-main-20261014')
    cpp = evidence.file(source_tree / 'src/sycl/VocabMask.cpp')
    authoritative_cpp = evidence.file(original / 'src/sycl/VocabMask.cpp')
    analyzer.require(analyzer.digest(cpp) == analyzer.digest(authoritative_cpp), 'fresh SOURCE math differs from authoritative SOURCE')
    helper = evidence.file(source_tree / 'src/sycl/SYCLHelpers.h')
    analyzer.require(analyzer.digest(helper) == analyzer.digest(evidence.file(original / 'src/sycl/SYCLHelpers.h')), 'fresh SOURCE launch helper differs')
    bridge = evidence.file(root / 'logits-vocab/fresh-source-icpx-v1/xe35/VocabMask_api_bridge.cpp')
    evidence.stable()
    analyzer.require(not torch.xpu.is_initialized(), 'CPU review initialized XPU')
    report = {'actual_exit': 0, 'original_wrapper_exit': 1, 'original_logits_exit': 1, 'completed_families': completed, 'completed_family_cases': 64, 'completed_embedding_extra': {'case': 'embedding-decode8', 'samples': 101, 'accuracy': embedding['accuracy']}, 'result_json_count': 65, 'full75case_12phase_analysis_complete': False, 'performance_accepted': False, 'mask_failure': {'mapped_mismatch_elements': 5, 'total': 8, 'single_native_nodes': 1, 'native_kernel': 'VocabMaskKernel<long>', 'stage': 'single instantiate/replay -> first exact mapped check, before repeated capture/events/samples', 'actual_failed_tensor_saved': False, 'failed_values_known': False, 'post_replay_invalid_check_reached': False, 'old_absent_files': [str(path) for path in absent]}, 'eager_accuracy_v2_scope': {'six_cases_original_audit_exit': 0, 'CPU_independently_rechecked_here': 'mask-decode8 full ids/mapped/invalid only; other eager cases rely on preserved original audited evidence', 'SOURCE_same_native_SHA': source_native, 'actual_saved_eager_values_not_failed_values': eager_records}, 'SOURCE_body_SHA': analyzer.digest(cpp), 'SOURCE_helper_SHA': analyzer.digest(helper), 'SOURCE_alias_bridge_SHA': analyzer.digest(bridge), 'static_lifecycle': 'original C++ allocates and returns mapped+invalid via torch::empty, queue kernel uses input/mapped/invalid data_ptr; SOURCE-only alias forwards unchanged; Python mask_outputs list replaced inside capture and retained until replay check; singleton graph not reset before failed check', 'native_argument_address_scope': 'old DOT shows hCommandList/hKernel/pArguments addresses and typed arg[0..8], not input/mapped/invalid pointer values; command handles/pArguments are not tensor data_ptr evidence', 'root_cause_established': False, 'FM_related_cause_claimed': False, 'GPU_started': False, 'compiler_started': False, 'XPU_initialized': False, 'files': evidence.files}
    save(DIRECTORY / 'SOURCE_MASK_FAILURE_CPU_AUDIT.json', report)
    print(json.dumps({key: report[key] for key in ('actual_exit', 'completed_family_cases', 'result_json_count', 'performance_accepted', 'root_cause_established', 'GPU_started')}))


if __name__ == '__main__':
    main()
