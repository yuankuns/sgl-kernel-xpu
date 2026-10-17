import hashlib
import json
from pathlib import Path
import time


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    directory = Path(__file__).resolve().parent
    grant = json.loads((directory / 'main-current-source-accuracy-gpu8-grant-v1.json').read_text())
    ready = json.loads((directory / 'CPU-READY.json').read_text())
    assert grant['issuer'] == 'main' and grant['release'] is True
    assert grant['gpu'] == 8 and grant['accuracy_only'] is True and grant['performance_authorized'] is False
    assert grant['CPU_affinity'] == list(range(64, 80)) and grant['OMP_NUM_THREADS'] == 8
    assert grant['start_epoch'] >= grant['grant_epoch'] + 120 and time.time() >= grant['start_epoch']
    assert grant['bound_files_SHA256'] == ready['bound_files_SHA256']
    for name, expected in ready['bound_files_SHA256'].items():
        assert digest(Path(name)) == expected, name
    assert ready['original_silu_opmath_cases'] == 14 and ready['original_full_precision_cases'] == 1300
    assert ready['production_shapes'] == 6 and ready['complete_original_fixtures'] == 3
    print(json.dumps({'verified_grant_epoch': grant['grant_epoch'], 'gpu': 8, 'accuracy_only': True}))


if __name__ == '__main__':
    main()
