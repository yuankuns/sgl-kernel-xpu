import base64
import gzip
import hashlib
import json
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parent
CRASH = Path('/var/crash/_usr_bin_python3.12.0.crash')
FIELDS = {'Date', 'ExecutablePath', 'ProcCmdline', 'Signal', 'ProcStatus'}


def main():
    started = time.time()
    selected = {}
    current = None
    output_path = ROOT / 'failed-r1-B0-original.core.gz'
    checksum = hashlib.sha256()
    total = 0
    with CRASH.open() as stream, output_path.open('xb') as output:
        for line in stream:
            assert time.time() - started < 240
            if not line.startswith(' '):
                current = line.split(':', 1)[0]
                if current in FIELDS:
                    selected[current] = line.split(':', 1)[1].strip()
                if current == 'CoreDump':
                    assert line.strip() == 'CoreDump: base64'
                    assert selected['Signal'] == '11'
                    assert 'Pid:\t3303047' in selected['ProcStatus']
                    assert 'performance_driver.py --comparison source-port --role port --round 1 --slot B0' in selected['ProcCmdline']
            elif current == 'CoreDump':
                block = base64.b64decode(line.strip(), validate=True)
                total += len(block)
                assert total <= 2 * 1024 ** 3
                output.write(block)
                checksum.update(block)
            elif current in FIELDS:
                selected[current] += '\n' + line.rstrip()
    assert total > 0
    with (ROOT / 'APPORT_MATCHED_PID_METADATA.json').open('x') as output:
        json.dump({'original_crash_readonly': str(CRASH), 'fields': selected,
            'matched_host_pid': 3303047, 'compressed_core_sha256': checksum.hexdigest(),
            'compressed_core_bytes': total, 'CPU_only': True, 'GPU_compile_started': False}, output, indent=2)
    core_path = ROOT / 'failed-r1-B0-original.core'
    checksum = hashlib.sha256()
    total = 0
    with gzip.open(output_path, 'rb') as source, core_path.open('xb') as output:
        while block := source.read(8 * 1024 ** 2):
            assert time.time() - started < 240
            total += len(block)
            assert total <= 16 * 1024 ** 3
            output.write(block)
            checksum.update(block)
    with (ROOT / 'CORE_EXTRACTION_CPU.json').open('x') as output:
        json.dump({'actual_exit': 0, 'core_sha256': checksum.hexdigest(), 'bytes': total,
            'elapsed_seconds': time.time() - started, 'CPU_only': True, 'core_read_not_execute': True}, output, indent=2)
    print(json.dumps({'matched_pid': 3303047, 'CPU_core_extract': 0, 'bytes': total, 'elapsed': time.time() - started}))


if __name__ == '__main__':
    main()
