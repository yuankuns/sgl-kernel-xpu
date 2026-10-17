import ast
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def add_file(path, text):
    assert not path.exists(), path
    patch = '*** Begin Patch\n*** Add File: ' + str(path) + '\n' + ''.join('+' + line + '\n' for line in text.splitlines()) + '*** End Patch\n'
    subprocess.run(['apply_patch'], input=patch, text=True, check=True)


def main():
    directory = Path(__file__).resolve().parent
    artifact = directory.parent
    source = Path('/home/syk/worktrees/sgl-kernel-xpu/muse-spark-mxfp4-routing-quant-main-20261014')
    operations = ['muse_spark_moe_routing', 'sgl_per_token_group_quant_fp4',
                  'sgl_padded_gather_per_token_group_quant_fp4', 'mxfp4_moe_scale_to_mn_major', 'topk_sigmoid']
    records = {}
    for name in ['test_moe_routing.py', 'test_per_token_group_quant_mxfp4.py', 'test_topk_sigmoid.py', 'test_routing_benchmark_scope.py', 'conftest.py']:
        original = (source / 'tests' / name).read_text()
        text = original
        for operation in operations:
            text = re.sub(r'\b' + operation + r'\b', operation + '_xe35', text)
        old_path = 'Path(__file__).resolve().parents[1]'
        new_path = "Path('/root/worktrees/sgl-kernel-xpu/muse-spark-mxfp4-routing-quant-main-20261014')"
        text = text.replace(old_path, new_path)
        recovered = text.replace(new_path, old_path)
        for operation in reversed(operations):
            recovered = re.sub(r'\b' + operation + r'_xe35\b', operation, recovered)
        assert recovered == original
        ast.parse(text)
        add_file(directory / 'tests' / name, text)
        records['tests/' + name] = {'original_SHA256': hashlib.sha256(original.encode()).hexdigest(),
                                    'adapted_SHA256': hashlib.sha256(text.encode()).hexdigest(),
                                    'inverse_bytes_equal_original': True}
    source_worker = artifact / 'source_accuracy_only.py'
    original = source_worker.read_text()
    replacements = [('choices=("historical", "fresh")', 'choices=("after",)'),
                    ('torch.set_num_threads(16)', 'torch.set_num_threads(8)'), ('suffix = ""', 'suffix = "_xe35"')]
    worker = original
    for before, after in replacements:
        assert worker.count(before) == 1
        worker = worker.replace(before, after)
    recovered = worker
    for before, after in reversed(replacements):
        recovered = recovered.replace(after, before)
    assert recovered == original
    add_file(directory / 'production_accuracy.py', worker)
    records['production_accuracy.py'] = {'original_SHA256': digest(source_worker),
        'adapted_SHA256': hashlib.sha256(worker.encode()).hexdigest(), 'inverse_bytes_equal_original': True,
        'adaptations': replacements, 'math_reference_shape_body_unchanged': True}
    previous = artifact / 'fresh-three-source-diagnostic-v1'
    original = (previous / 'accuracy_capture.py').read_text()
    replacements = [("'normal-source-precheck.json'", "'fixture-provider-precheck.json'"),
                    ('list(range(64, 80))', 'list(range(72, 80))'),
                    ("os.environ['ZE_AFFINITY_MASK'] == '8'", "os.environ['ZE_AFFINITY_MASK'] == '11'"),
                    ('torch.set_num_threads(16)', 'torch.set_num_threads(8)'),
                    ('sgl_kernel.sgl_per_token_group_quant_fp4(', 'sgl_kernel.sgl_per_token_group_quant_fp4_xe35(')]
    worker = original
    for before, after in replacements:
        assert before in worker
        worker = worker.replace(before, after)
    recovered = worker
    for before, after in reversed(replacements):
        recovered = recovered.replace(after, before)
    assert recovered == original
    add_file(directory / 'fixture_accuracy.py', worker)
    records['fixture_accuracy.py'] = {'original_SHA256': digest(previous / 'accuracy_capture.py'),
        'adapted_SHA256': hashlib.sha256(worker.encode()).hexdigest(), 'inverse_bytes_equal_original': True,
        'adaptations': replacements, 'separate_silu_and_mul_shared_unchanged': True}
    shutil.copyfile(previous / 'CPU-original-fixtures.json', directory / 'CPU-original-fixtures.json')
    for fixture in json.loads((previous / 'CPU-original-fixtures.json').read_text())['fixtures']:
        shutil.copyfile(previous / fixture['fixture_file'], directory / fixture['fixture_file'])
        assert digest(previous / fixture['fixture_file']) == digest(directory / fixture['fixture_file'])
    for path in directory.rglob('*.py'):
        ast.parse(path.read_text())
    report = {'CPU_only_preparation': True, 'GPU_launched': False, 'PR_or_SOURCE_modified': False,
              'operations_adapter': operations, 'records': records,
              'test_fixture_math_assertions_parametrizations_preserved': True,
              'benchmark_scope_only_path_adapter': 'original readonly source benchmark CPU CLI tests preserved',
              'independent_fused_assertions_not_original_pytest_coverage': True}
    (directory / 'mechanical-adaptation-audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
