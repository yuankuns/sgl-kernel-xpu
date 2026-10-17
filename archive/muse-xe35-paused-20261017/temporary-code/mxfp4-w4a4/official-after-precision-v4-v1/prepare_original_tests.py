import hashlib
import json
from pathlib import Path
import subprocess


def main():
    directory = Path(__file__).resolve().parent
    source = Path('/home/syk/worktrees/sgl-kernel-xpu/muse-spark-mxfp4-w4a4-main-20261014/tests/test_mxfp4_moe_w4a4.py')
    original = source.read_text()
    substitutions = [
        ('moe_grouped_mm_nt_xe35_mxfp4_w4a4', 'moe_grouped_mm_nt_mxfp4_w4a4_xe35'),
        ('mxfp4_moe_scale_to_mn_major', 'mxfp4_moe_scale_to_mn_major_xe35'),
        ('pathlib.Path(__file__).resolve().parents[1] / "benchmark/accuracy_reference.py"',
         'pathlib.Path("/root/worktrees/sgl-kernel-xpu/muse-spark-mxfp4-w4a4-main-20261014/benchmark/accuracy_reference.py")'),
    ]
    adapted = original
    counts = {}
    for before, after in substitutions:
        counts[before] = adapted.count(before)
        assert counts[before] > 0 and after not in adapted
        adapted = adapted.replace(before, after)
    recovered = adapted
    for before, after in reversed(substitutions):
        recovered = recovered.replace(after, before)
    assert recovered == original
    target = directory / 'test_original_40_normal_package.py'
    assert not target.exists()
    patch = '*** Begin Patch\n*** Add File: ' + str(target) + '\n'
    patch += ''.join('+' + line + '\n' for line in adapted.splitlines())
    patch += '*** End Patch\n'
    subprocess.run(['apply_patch', patch], check=True, capture_output=True, text=True)
    assert target.read_text() == adapted
    report = {'source': str(source), 'source_sha256': hashlib.sha256(original.encode()).hexdigest(),
              'adapted_test': str(target), 'adapted_sha256': hashlib.sha256(adapted.encode()).hexdigest(),
              'substitutions': substitutions, 'occurrences': counts, 'inverse_substitution_original_bytes_equal': True,
              'math_reference_tolerances_parameters_and_cases_preserved': True,
              'normal_registered_API_only': True, 'collection_and_GPU_execution': 'not yet run'}
    (directory / 'original-test-adaptation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
