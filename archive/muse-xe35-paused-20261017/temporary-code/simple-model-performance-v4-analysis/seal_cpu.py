import json
from pathlib import Path

import analyzer


directory = Path(__file__).resolve().parent
expected = {'contract': '0', 'ready': '0', 'first-source': '0', 'incomplete': '2'}
for name, status in expected.items():
    analyzer.require((directory / f'{name}-r3.exit').read_text().strip() == status, 'unexpected CPU actual exit: ' + name)
analyzer.require(analyzer.decode((directory / 'CPU-contract-r3.json').read_text())['tests'] == 16, 'CPU contract coverage changed')
for name in ('READY-r3.json', 'first-SOURCE-readonly-r3.json'):
    report = analyzer.decode((directory / name).read_text())
    analyzer.require(report['actual_exit'] == 0 and not report['XPU_initialized'], 'CPU readiness failed')
    analyzer.require(report['analyzer_SHA256'] == analyzer.digest(analyzer.__file__), 'tested analyzer SHA changed')
files = {path.name: {'sha256': analyzer.digest(path), 'bytes': path.stat().st_size} for path in directory.iterdir() if path.is_file()}
analyzer.save_new(directory / 'CPU_READY_SEAL_v1.json', {'actual_exit': 0, 'CPU_contract_tests': 16, 'READY_exit': 0, 'first_SOURCE_completed_cases': 6, 'incomplete_full_run_actual_exit': 2, 'full75case_acceptance': False, 'GPU_started': False, 'compiler_started': False, 'files': files})
print(json.dumps({'actual_exit': 0, 'sealed_files': len(files), 'full75case_acceptance': False}))
