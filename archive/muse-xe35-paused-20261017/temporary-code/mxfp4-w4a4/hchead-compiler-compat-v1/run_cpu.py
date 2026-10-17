import hashlib
import json
import pathlib
import re
import subprocess


def main():
    sidecar = pathlib.Path(__file__).resolve().parent
    root = sidecar.parent.parent
    source = pathlib.Path('/root/worktrees/sgl-kernel-xpu-up/pr/muse-mxfp4-w4a4-xe35')
    generated = root / 'canonical-icpx-pool/build/src/CMakeFiles/sgl-ops-sycl-HCHead.dir/sycl/sgl-ops-sycl-HCHead_gen_HCHead.cpp.o.Release.cmake'
    text = generated.read_text()
    includes = re.search(r'^set\(SYCL_include_dirs \[==\[(.*)\]==\]\)', text, re.MULTILINE).group(1).split(';')
    definitions = re.search(r'^set\(SYCL_compile_definitions \[==\[(.*)\]==\]\)', text, re.MULTILINE).group(1).split(';')
    flags = re.search(r'^set\(SYCL_compile_flags (.*)\) # list$', text, re.MULTILINE).group(1).split(';')
    compiler = '/opt/gfx-deps/dpcpp/bin/icpx'
    base = [compiler, '-std=c++20', '-O3', '-fPIC', '-Wno-deprecated-declarations'] + ['-I' + path for path in dict.fromkeys(includes + [str(source / 'src/sycl')])] + ['-D' + value for value in definitions]
    original = sidecar / 'HCHead.original.cpp'
    guarded = sidecar / 'HCHead.device-guard.cpp'
    if original.read_bytes() != (source / 'src/sycl/HCHead.cpp').read_bytes():
        raise RuntimeError('Original source snapshot changed')
    cases = [
        ('plain-host-original', base + ['-c', str(original), '-o', str(sidecar / 'plain-host-original.o')], 1),
        ('sycl-host-only-original', base + ['-fsycl', '-fsycl-host-only', '-c', str(original), '-o', str(sidecar / 'sycl-host-only-original.o')], 0),
        ('plain-host-device-guard', base + ['-c', str(guarded), '-o', str(sidecar / 'plain-host-device-guard.o')], 0),
        ('original-device-ir', base + flags + ['-fsycl-device-only', '-S', '-emit-llvm', str(original), '-o', str(sidecar / 'original-device.ll')], 0),
        ('guarded-device-ir', base + flags + ['-fsycl-device-only', '-S', '-emit-llvm', str(guarded), '-o', str(sidecar / 'guarded-device.ll')], 0),
    ]
    results = []
    for name, command, expected in cases:
        (sidecar / (name + '.command.json')).write_text(json.dumps(command, indent=2) + '\n')
        print(json.dumps({'launch': name, 'command': command, 'bounded_seconds': 120, 'compile_only_no_gpu': True}), flush=True)
        with (sidecar / (name + '.log')).open('w') as log:
            process = subprocess.run(['timeout', '--signal=TERM', '--kill-after=15s', '120s'] + command, stdout=log, stderr=subprocess.STDOUT)
        (sidecar / (name + '.exit')).write_text(str(process.returncode) + '\n')
        results.append({'case': name, 'actual_exit': process.returncode, 'expected_exit': expected})
        print(json.dumps(results[-1]), flush=True)
        if process.returncode != expected:
            raise RuntimeError(f'Unexpected bounded compile result: {name}: {process.returncode}')
    first = (sidecar / 'plain-host-original.log').read_text()
    if "attribute requires exactly 3 arguments" not in first:
        raise RuntimeError('Expected real one-argument SYCL attribute root cause not reproduced')
    report = {'kind': 'actual HCHead TU CPU compile compatibility diagnostic, not full or GPU acceptance', 'generated_original_flags_path': str(generated), 'generated_original_flags_sha256': hashlib.sha256(generated.read_bytes()).hexdigest(), 'compiler': compiler, 'original_source_sha256': hashlib.sha256(original.read_bytes()).hexdigest(), 'unchanged_precision_device_flags': flags, 'results': results, 'gpu': 'not run', 'full_build': 'not run', 'frozen_pr_writes': 0}
    (sidecar / 'compile-results.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
