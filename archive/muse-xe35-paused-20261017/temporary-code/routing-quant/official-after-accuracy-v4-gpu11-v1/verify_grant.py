import hashlib
import json
from pathlib import Path
import time


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


directory = Path(__file__).resolve().parent
ready_path = directory / 'AFTER_CPU_READY_MANIFEST.json'
ready = json.loads(ready_path.read_text())
grant = json.loads((directory / 'main-normal-after-accuracy-gpu11-grant-v1.json').read_text())
assert grant['issuer'] == 'main' and grant['release']
assert grant['gpu'] == 11 and grant['CPU_affinity'] == list(range(72, 80))
assert grant['version'] == 'canonical-v4-icpx' and grant['module'] == 'routing-quant'
assert not grant['performance_authorized'] and not grant['compile_authorized']
assert grant['ready_manifest_sha256'] == digest(ready_path)
assert grant['start_epoch'] >= grant['grant_epoch'] + 120 and time.time() >= grant['start_epoch']
assert grant['original_full_1300_authorized'] and grant['independent_three_fixtures_authorized'] and grant['six_production_shapes_authorized']
for name, expected in ready['bound_files_sha256'].items():
    assert digest(Path(name)) == expected, name
print(json.dumps(grant, indent=2))
