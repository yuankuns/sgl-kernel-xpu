import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
JOB = ROOT / 'source-before-icpx-bounded-v1'
site = Path(sys.prefix) / 'lib/python3.12/site-packages/sgl_kernel'
expected = json.loads((JOB / 'site-sha.json').read_text())['post']
import sgl_kernel
import torch

assert Path(sgl_kernel.__file__).resolve() == (site / '__init__.py').resolve()
op = torch.ops.sgl_kernel.fused_post_norm_residual_pre_norm
mapped = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                 if 'libsgl-ops-sycl-' in line or '/common_ops' in line})
proof = {}
for filename in mapped:
    path = Path(filename).resolve()
    relative = path.relative_to(site.resolve()).as_posix()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert expected[relative] == digest
    proof[relative] = digest
assert 'libsgl-ops-sycl-FusedNormResidual.so' in proof
assert any(name.startswith('common_ops.') for name in proof)
record = {'package': str(sgl_kernel.__file__), 'prefix': sys.prefix, 'original_registered_op': str(op.default._schema),
          'mapped_libraries': proof, 'registration_common_ops': 'historical GNU, not recompiled',
          'tested_native': 'fresh icpx', 'Device': 'original retained; no native DT_NEEDED Device edge',
          'GPU_work_executed': False, 'normal_import_verified': True, 'accuracy_perf_not_run': True}
(JOB / 'normal-import-proof.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2))
