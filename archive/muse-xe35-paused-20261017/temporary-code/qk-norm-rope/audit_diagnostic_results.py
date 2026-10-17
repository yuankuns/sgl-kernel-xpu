import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--run', default='diagnostic-retry1')
args = parser.parse_args()
root = Path(__file__).resolve().parent
run = root / args.run
assert run.parent == root
assert (run / 'supervisor.exit').read_text().strip() == '0'
commands = json.loads((run / 'commands.json').read_text())
summary = json.loads((run / 'summary.json').read_text())
expected_timings = {tokens: 8 if 'retest' in summary[str(tokens)] else 4 for tokens in (8, 32768)}
expected_processes = 2 * sum(expected_timings.values())
assert len(commands) == 7 + expected_processes
for command in commands:
    log = Path(command['log'].replace('/root/', '/home/syk/', 1))
    assert command['exit'] == 0
    assert hashlib.sha256(log.read_bytes()).hexdigest() == command['log_sha256']
    assert log.with_suffix('.exit').read_text().strip() == '0'
accuracy = json.loads((run / 'accuracy.json').read_text())
samples = 0
provenance = json.loads((run / 'provenance.json').read_text())
for tokens in (8, 32768):
    source, port = (accuracy[f'{label}-t{tokens}'] for label in ('source', 'port'))
    assert source['output_sha256'] == port['output_sha256']
    assert source['input_sha256'] == port['input_sha256']
    for label in ('source', 'port'):
        record = accuracy[f'{label}-t{tokens}']
        assert record['violations'] == 0 and record['tail_equal']
        assert record['atol'] == record['rtol'] == 0.005
        timings = [json.loads(path.read_text()) for path in (run / label).glob(f'graph-*-t{tokens}-s*.json')]
        assert len(timings) == expected_timings[tokens]
        library = 'libsgl-ops-sycl-FusedQKNormRope_xe35.so' if label == 'port' else 'libsgl-ops-sycl-FusedQKNormRope.so'
        expected_path = str(Path(provenance[f'{label}_site'].replace('/home/syk/', '/root/', 1)) / 'sgl_kernel' / library)
        for timing in timings:
            assert timing['samples'] == 21 and timing['repeats'] == 32
            assert timing['dtype'] == 'bf16' and len(timing['raw_us']) == 21
            assert timing['loaded_libraries'][expected_path] == provenance[f'{label}_qk_library_sha256']
            samples += len(timing['raw_us'])
fp16 = next(json.loads(line) for line in (run / 'port/fp16-t8.log').read_text().splitlines() if line.startswith('{'))
assert fp16['violations'] == 0 and fp16['tail_equal']
assert fp16['atol'] == fp16['rtol'] == 0.001
record = {'run': args.run, 'verified_commands': len(commands), 'verified_graph_processes': expected_processes, 'verified_raw_samples': samples, 'log_hashes_and_exits_verified': True, 'full_bf16_cpu_reference_violations': 0, 'source_port_full_input_output_bitwise_equal': True, 'fp16_t8': fp16, 'official_full_package_pass': False}
(run / 'integrity.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2))
