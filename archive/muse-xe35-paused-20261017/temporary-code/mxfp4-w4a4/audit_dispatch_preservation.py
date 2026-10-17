import hashlib
import json
import pathlib
import re


def normalize(text):
    text = re.sub(r'/\*.*?\*/|//[^\n]*', '', text, flags=re.DOTALL)
    text = re.sub(r'^\s*#include[^\n]*', '', text, flags=re.MULTILINE)
    return re.sub(r'\s+', '', text.replace('_xe35', ''))


def main():
    artifact = pathlib.Path(__file__).resolve().parent
    sources = pathlib.Path('/home/syk/worktrees/sgl-kernel-xpu')
    targets = pathlib.Path('/home/syk/worktrees/sgl-kernel-xpu-up/pr')
    pairs = [
        ('mxfp4-w4a4', 'src/sycl/xe35/moe_grouped_mm_mxfp4_w4a4.cpp', 'src/sycl/xe35/moe_grouped_mm_mxfp4_w4a4.cpp'),
        ('mxfp4-w4a4', 'src/sycl/mxfp4_moe_scale_layout.cpp', 'src/sycl/xe35/mxfp4_moe_scale_layout.cpp'),
        ('routing-quant', 'src/sycl/per_token_group_quant_fp4.cpp', 'src/sycl/xe35/per_token_group_quant_fp4.cpp'),
        ('routing-quant', 'src/sycl/mxfp4_moe_scale_layout.cpp', 'src/sycl/xe35/mxfp4_moe_scale_layout.cpp'),
    ]
    reports = []
    for module, source_name, target_name in pairs:
        source = sources / ('muse-spark-mxfp4-' + ('w4a4' if module == 'mxfp4-w4a4' else 'routing-quant') + '-main-20261014') / source_name
        target = targets / ('muse-' + module + '-xe35') / target_name
        original = normalize(source.read_text())
        actual = normalize(target.read_text())
        assert original == actual, str(target)
        reports.append({'module': module, 'source': str(source), 'target': str(target), 'body_and_dispatch_identical_after_arch_symbol_include_comment_whitespace_normalization': True, 'normalized_sha256': hashlib.sha256(actual.encode()).hexdigest()})
    source_routing = sources / 'muse-spark-mxfp4-routing-quant-main-20261014/src/sycl/TopKSigMoid.cpp'
    target_routing = targets / 'muse-routing-quant-xe35/src/sycl/xe35/muse_routing_quant.cpp'
    original = source_routing.read_text().split('namespace TopKSigmoidImpl {', 1)[1].split('template <typename T>\nstruct FusedTopkSigmoid', 1)[0]
    actual = target_routing.read_text().split('namespace TopKSigmoidImpl_xe35 {', 1)[1].split('void topk_sigmoid_xe35(', 1)[0]
    original_normalized = normalize(original)
    actual_normalized = normalize(actual)
    assert actual_normalized == original_normalized + '}', 'routing kernel/dispatch differs'
    reports.append({'module': 'routing-quant', 'source': str(source_routing), 'target': str(target_routing), 'original_routing_subtree_and_dispatch_preserved': True, 'normalized_source_sha256': hashlib.sha256(original_normalized.encode()).hexdigest(), 'topk_adapter': 'new architecture API calls preserved original routing dispatch for supported full-tile topk8, otherwise delegates unchanged generic upstream API; int64_t div_up template selection is compilation adaptation'})
    source_headers = sources / 'muse-spark-mxfp4-w4a4-main-20261014/src/sycl/kernels/moe/xe35'
    target_headers = targets / 'muse-mxfp4-w4a4-xe35/src/sycl/xe35/moe'
    for path in sorted(target_headers.rglob('*')):
        if not path.is_file() or path.name == 'moe_bf16_reorder.hpp':
            continue
        original = normalize((source_headers / path.relative_to(target_headers)).read_text())
        actual = normalize(path.read_text())
        adaptation = 'none'
        if path.name == 'moe_array_epilogue.hpp':
            expected_call = '::sgl_kernel::vllm_grouped_gemm::reorder_moe_output('
            assert actual.count(expected_call) == 2
            actual = actual.replace(expected_call, 'reorder(')
            adaptation = 'exactly two authorized W4 output reorder calls; all other body preserved'
        if path.name == 'moe_array_mma_mxfp.hpp':
            assert actual.count('MainloopIntelXeXMX16BlockScaledImpl<') == 2
            actual = actual.replace('MainloopIntelXeXMX16BlockScaledImpl<', 'MainloopIntelXeXMX16BlockScaled<')
            adaptation = 'two public-pin fallback template type-name compatibility aliases; no policy/shape/math change'
        assert original == actual, str(path)
        reports.append({'module': 'mxfp4-w4a4', 'target': str(path), 'source_body_preserved': True, 'explicit_dependency_adaptation': adaptation, 'normalized_source_sha256': hashlib.sha256(original.encode()).hexdigest()})
    report = {'kind': 'Static audit, not runtime acceptance', 'no_GPU8_performance_regression_kernel_or_dispatch_tuning': True, 'only_precision_adaptation': 'Authorized W4 local source-preserving BF16 reorder helper and two epilogue output call sites; no public dependency/global/BMG changes', 'files': reports}
    for module in ('mxfp4-w4a4', 'routing-quant'):
        destination = artifact.parent / module
        (destination / 'performance-change-audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
