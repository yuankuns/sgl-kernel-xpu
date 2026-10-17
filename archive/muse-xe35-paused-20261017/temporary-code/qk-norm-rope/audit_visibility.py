import hashlib
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent
target = root.parents[1] / 'pr/muse-qk-norm-rope-xe35'
libraries = {
    'fresh_port': (target / 'build/src/libsgl-ops-sycl-FusedQKNormRope_xe35.so', 'fused_qk_norm_rope_xe35'),
    'native_abi_forward': (root / 'diagnostic/build/libsgl-ops-sycl-FusedQKNormRope.so', 'fused_qk_norm_rope'),
}
generic = target / 'build/src/libsgl-ops-sycl-FusedQKNormRope.so'
if generic.exists():
    libraries['official_generic'] = (generic, 'fused_qk_norm_rope')
records = {}
for name, (path, function) in libraries.items():
    output = subprocess.check_output(['nm', '-D', '-C', str(path)], text=True)
    (root / f'visibility-{name}.log').write_text(output)
    public = [line for line in output.splitlines() if f' T at::native::xpu::{function}(' in line]
    assert len(public) == 1, (name, public)
    records[name] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'public_symbol': public[0]}
(root / 'visibility-audit.json').write_text(json.dumps({'command': 'nm -D -C', 'records': records, 'official_generic_available': generic.exists()}, indent=2) + '\n')
print('PASS: actual dynamic symbol table exports native entry points as global T')
