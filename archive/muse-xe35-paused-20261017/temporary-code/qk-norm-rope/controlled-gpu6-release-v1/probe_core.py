import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'controlled-method-runner-cpu-v1'))
from protocol import digest, require, save
from runner import stack


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--role', choices=('source', 'port'), required=True)
    args = parser.parse_args()
    require(sorted(os.sched_getaffinity(0)) == list(range(16)), 'CPU probe0–15 only')
    import torch

    torch.set_num_threads(1)
    require(not torch.xpu.is_initialized(), 'CPU normal import cannot initialize XPU')
    sys.path.insert(0, str(ROOT.parent / 'normal-v4-model-config-accuracy-v2'))
    from run_accuracy_r2 import normal_audit

    normal = normal_audit(args.role)
    require(not torch.xpu.is_initialized(), 'CPU normal audit cannot initialize XPU')
    runtime = []
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit() or int(directory.name) == os.getpid():
            continue
        try:
            environment = (directory / 'environ').read_bytes().split(b'\0')
            if b'ZE_AFFINITY_MASK=7' not in environment:
                continue
            mapped = {}
            for line in (directory / 'maps').read_text().splitlines():
                parts = line.split(maxsplit=5)
                if len(parts) == 6 and parts[5].startswith('/') and any(token in Path(parts[5]).name for token in
                    ('libtorch', 'libc10', 'libsycl', 'libur_', 'libze_', 'libumf', 'libmpi')):
                    path = Path(parts[5]).resolve()
                    mapped[str(path)] = digest(path)
            if any('libur_adapter' in path for path in mapped) and any('libze_' in path for path in mapped):
                runtime.append({'pid': int(directory.name), 'observed_ZE_AFFINITY_MASK': '7', 'actual_core_loaded_stack_sha256': mapped})
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            continue
    save(ROOT / f'CPU-core-{args.role}.json', {'actual_exit': 0, 'epoch': time.time(), 'pid': os.getpid(),
        'CPU_affinity': sorted(os.sched_getaffinity(0)), 'XPU_initialized': torch.xpu.is_initialized(), 'normal': normal,
        'CPU_actual_core_loaded_stack_sha256': stack(), 'read_only_existing_main_GPU7_process_maps': runtime,
        'no_GPU_kernel_context_initialization_or_other_process_modification': True})
    print(f'Normal {args.role} package CPU audit0; XPU uninitialized; actual CPU maps plus {len(runtime)} read-only main7 runtime maps')


if __name__ == '__main__':
    main()
