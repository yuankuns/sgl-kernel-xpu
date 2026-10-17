import argparse
import hashlib
import json
import os
from pathlib import Path
import re


ART = Path(__file__).resolve().parent.parent
HOME = Path('/root') if str(ART).startswith('/root/') else Path('/home/syk')
ARCHIVE = HOME / 'worktrees/sgl-kernel-xpu/review-validation-artifacts/fmha-before-after-20261017'
TARGETS = ('sgl-ops-sycl-flash_attentionXe35', 'sgl-ops-sycl-flash_attentionXe35_decode_64_bf16',
           'sgl-ops-sycl-flash_attentionXe35_prefill_64_bf16')


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def variable(text, name):
    matches = re.findall(r'^set\(' + re.escape(name) + r' ([^\n]*)\)\s*(?:#[^\n]*)?$', text, re.MULTILINE)
    matches = [value for value in matches if '${' not in value]
    assert len(matches) == 1, name
    return matches[0].strip().strip('"')


def baseline_provenance(provider):
    assert provider in ('source', 'preopt')
    kind = os.environ.get('FMHA_BASELINE_KIND', 'archived')
    assert kind in ('archived', 'matched-native-icpx'), kind
    if kind == 'matched-native-icpx':
        from matched_baseline_provenance import provenance
        return provenance(provider)
    side = 'after' if provider == 'source' else 'before'
    build = ARCHIVE / f'build-{side}'
    package = ARCHIVE / f'site-{side}/sgl_kernel'
    source_manifest = json.loads((ARCHIVE / 'source-manifest.json').read_text())
    source = next(record for record in source_manifest['sources'] if record['side'] == side)
    expected = '08955d90fad52a40779705f259569db93e570a22' if provider == 'source' else 'ab3a26c43a38e6aea7d73cc1349e4b5e210b772b'
    assert source['head'] == expected and source['tracked_source_clean'] is True
    compiler_files = list((build / 'CMakeFiles').glob('*/CMakeCXXCompiler.cmake'))
    assert len(compiler_files) == 1
    compiler_text = compiler_files[0].read_text()
    manifest = json.loads((ARCHIVE / f'binary-manifest-{side}.json').read_text())
    hashes = {Path(record['path']).name: record['sha256'] for record in manifest}
    ninja_log = (build / '.ninja_log').read_text()
    ninja_blocks = (build / 'build.ninja').read_text().split('\n\n')
    rules_blocks = (build / 'CMakeFiles/rules.ninja').read_text().split('\n\n')
    records = {}
    for target in TARGETS:
        files = list((build / 'src/CMakeFiles' / f'{target}.dir/sycl').glob('*.Release.cmake'))
        assert len(files) == 1, target
        text = files[0].read_text()
        compiler = variable(text, 'SYCL_executable')
        source_input = variable(text, 'source_file')
        local_input = Path(source_input) if str(ART).startswith('/root/') else Path(source_input.replace('/root/', '/home/syk/', 1))
        host = variable(text, 'SYCL_host_compiler')
        flags = variable(text, 'SYCL_compile_flags').split(';')
        definitions = variable(text, 'SYCL_compile_definitions').removeprefix('[==[').removesuffix(']==]').split(';')
        assert 'COMMAND "${SYCL_executable}"' in text and '${SYCL_host_compiler}' in text
        assert 'set(SYCL_host_compiler "-fsycl-host-compiler=${SYCL_host_compiler}")' in text
        library = f'lib{target}.so'
        assert f'src/{library}' in ninja_log
        assert digest(build / 'src' / library) == digest(package / library) == hashes[library]
        link_blocks = [block for block in ninja_blocks if block.startswith(f'build src/{library}: ')]
        assert len(link_blocks) == 1, library
        rule = link_blocks[0].splitlines()[0].split(': ', 1)[1].split()[0]
        link_rules = [block for block in rules_blocks if block.startswith(f'rule {rule}\n')]
        assert len(link_rules) == 1, rule
        compile_blocks = [block for block in ninja_blocks if block.startswith('build ') and
                          files[0].name in block and '  COMMAND = ' in block]
        assert len(compile_blocks) == 1, target
        device_blocks = [block for block in ninja_blocks if block.startswith('build ') and
                         f'{target}_sycl_device_obj.o:' in block and ' -fsycl-link ' in block]
        assert len(device_blocks) == 1, target
        assert compiler + ' -fPIC -fsycl' in device_blocks[0]
        device_command = next(line.strip().removeprefix('COMMAND = ') for line in device_blocks[0].splitlines()
                              if line.startswith('  COMMAND = '))
        commands = '\n\n'.join(link_rules + link_blocks + compile_blocks + device_blocks)
        assert files[0].name in commands and '/usr/bin/c++' in commands
        records[target] = {'library': library, 'library_sha256': hashes[library],
                           'original_source_input': source_input, 'original_source_input_sha256': digest(local_input),
                           'generated_command_file': str(files[0]), 'generated_command_sha256': digest(files[0]),
                           'native_sycl_driver': compiler, 'native_invocation_basename_is_icpx': Path(compiler).name == 'icpx',
                           'sycl_host_compiler': host, 'sycl_compile_flags': flags,
                           'host_compile_flags': variable(text, 'CMAKE_HOST_FLAGS'), 'compile_definitions': definitions,
                           'actual_device_link_command': device_command,
                           'ninja_commands': commands,
                           'native_command_template': 'SYCL_executable -MD -MF dependency -c source -o object includes -fsycl-host-compiler=SYCL_host_compiler -fsycl-host-compiler-options=host_flags SYCL_compile_flags'}
    cutlass = source_manifest['cutlass']
    return {'provider': provider, 'source_head': source['head'], 'native_kernels': records,
            'cxx_compiler_identity_file': str(compiler_files[0]), 'cxx_identity_file_sha256': digest(compiler_files[0]),
            'cxx_compiler': variable(compiler_text, 'CMAKE_CXX_COMPILER'),
            'cxx_compiler_id': variable(compiler_text, 'CMAKE_CXX_COMPILER_ID'),
            'cxx_compiler_version': variable(compiler_text, 'CMAKE_CXX_COMPILER_VERSION'),
            'build_ninja_sha256': digest(build / 'build.ninja'),
            'rules_ninja_sha256': digest(build / 'CMakeFiles/rules.ninja'), 'completed_ninja_log_sha256': digest(build / '.ninja_log'),
            'cutlass_historical_label': cutlass['historical_revision_label'], 'cutlass_git_metadata_present': cutlass['git_metadata_present'],
            'precision_policy': 'actual generated flags retained; no baseline rebuild/precision changes',
            'evidence_boundary': 'completed Ninja targets plus actual generated CMake execute_process/Ninja compile edges/linker rules; not XPU_SYCL_COMPILER token inference; no historical verbose subprocess-exec trace claimed',
            'cross_compiler_performance_confound': True,
            'confound': 'historical icx SYCL with GNU host and profiling ON/e878 versus new icpx host+SYCL and profiling OFF/public pin; not an isolated compiler-controlled algorithm speedup'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--provider', choices=('source', 'preopt'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = baseline_provenance(args.provider)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'provider': args.provider, 'head': report['source_head'], 'host_compiler_id': report['cxx_compiler_id'],
                      'native_drivers': sorted({record['native_sycl_driver'] for record in report['native_kernels'].values()}),
                      'native_driver_is_icpx': all(record['native_invocation_basename_is_icpx'] for record in report['native_kernels'].values()),
                      'cross_compiler_performance_confound': True, 'output': str(args.output)}))


if __name__ == '__main__':
    main()
