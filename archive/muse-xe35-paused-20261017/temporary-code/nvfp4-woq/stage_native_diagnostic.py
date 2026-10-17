import hashlib
import json
import pathlib
import shutil
import subprocess


artifact = pathlib.Path('/root/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017/nvfp4-woq')
historical = pathlib.Path('/root/worktrees/sgl-kernel-xpu/review-validation-artifacts/accuracy-20261016/nvfp4-woq-scope-correction')
source = pathlib.Path('/root/worktrees/sgl-kernel-xpu/muse-spark-nvfp4-woq-main-20261014')
assert (historical / 'full-default-build.exit').read_text().strip() == '0'
assert subprocess.check_output(['git', '-c', f'safe.directory={source}', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() == 'd983e877d6bf97015027bd3b1379ff047009a5dd'
original = historical / 'full-default-stage/sgl_kernel'
registration = list(original.glob('common_ops*.so'))
assert len(registration) == 1
expected = {
    'libsgl-ops-sycl-blockwise_moe_nvfp4_woq.so': '864f695ee11647af36ea1140c941d3ce6f68ae3c1c22a5928784d34766ff4e06',
    'libsgl-ops-sycl-nvfp4_woq.so': 'bad2595a97d75d903fecac9770ddc2bf659baaee1cff362c84204108b6c131fa',
    'libsgl-ops-sycl-mxfp4_moe_scale_layout.so': '2196b48e0e601bd78f9610bc2b6f5541c8942ba7b5e78d30af2946acef33e3d4',
}
for name, digest in expected.items():
    assert hashlib.sha256((source / 'build/src' / name).read_bytes()).hexdigest() == digest
report = {'scope': 'actual port kernels in real original full package; NOT official integration acceptance',
          'source_revision': 'd983e877d6bf97015027bd3b1379ff047009a5dd',
          'registration_package': str(original),
          'registration_build_provenance': str(historical / 'full-default-source.json'),
          'source_binary_sha256': expected, 'packages': {}}
for label in ('source', 'port'):
    stage = artifact / 'native-diagnostic' / label / 'site/sgl_kernel'
    shutil.copytree(original, stage, dirs_exist_ok=True)
    if label == 'source':
        for name in expected:
            shutil.copy2(source / 'build/src' / name, stage / name)
    else:
        for name in ('libsgl-ops-sycl-blockwise_moe_nvfp4_woq_port_payload.so',
                     'libsgl-ops-sycl-nvfp4_woq_port_payload.so',
                     'libsgl-ops-sycl-nvfp4_woq_scale_layout.so'):
            shutil.copy2(artifact / 'port-compile-audit/src' / name, stage / name)
    for name in ('nvfp4.py', 'cutlass_moe.py'):
        shutil.copy2(source / 'python/sgl_kernel' / name, stage / name)
    assert hashlib.sha256((stage / registration[0].name).read_bytes()).digest() == hashlib.sha256(registration[0].read_bytes()).digest()
    report['packages'][label] = {'path': str(stage), 'registration_sha256': hashlib.sha256((stage / registration[0].name).read_bytes()).hexdigest()}
(artifact / 'diagnostic-stage-provenance.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2), flush=True)
