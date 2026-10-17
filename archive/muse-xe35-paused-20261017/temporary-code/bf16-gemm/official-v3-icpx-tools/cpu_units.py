import json
import os
import subprocess
import sys
from pathlib import Path

from official_package_check import grant_preflight


root = Path(__file__).resolve().parent.parent
manifest, manifest_file, site, grant = grant_preflight(root, os.environ.get('CANONICAL_FRESH_GRANT'), os.environ.get('ACCURACY_ONLY', '0') == '1')
if os.environ.get('ONEAPI_DEVICE_SELECTOR') != 'level_zero:cpu':
    raise RuntimeError('CPU-only unit process must exclude GPU devices')
import torch
import sgl_kernel

if torch.xpu.is_available():
    raise RuntimeError('CPU-only test process unexpectedly has an available XPU')
if Path(sgl_kernel.__file__).resolve() != site / '__init__.py':
    raise RuntimeError('CPU tests must import the real new icpx package')
print(json.dumps({'actual_xpu_available': False, 'actual_package': sgl_kernel.__file__,
                  'source_version': 'canonical-v3-icpx', 'source_digest': manifest['source']['source_digest']}), flush=True)
test_file = Path(manifest['source']['source'].replace('/home/syk/', '/root/', 1)) / 'tests/test_bf16_gemm_nt_xpu.py'
status = subprocess.run([sys.executable, '-B', '-m', 'pytest', str(test_file), '-q', '-p', 'no:cacheprovider',
                         '-k', 'dispatch_capability_without_device_name or output_n_validation']).returncode
raise SystemExit(status)
