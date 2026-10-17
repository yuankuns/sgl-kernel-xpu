import hashlib
import json
import pathlib
import re
import subprocess


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            checksum.update(chunk)
    return checksum.hexdigest()


def read_cache(path):
    result = {}
    for line in path.read_text().splitlines():
        if line and not line.startswith(('#', '//')) and '=' in line and ':' in line.split('=', 1)[0]:
            key, value = line.split('=', 1)
            result[key.split(':', 1)[0]] = value
    return result


def inspect_native(artifact, binary, expected):
    if digest(binary) != expected:
        raise RuntimeError(f'Original archived native changed: {binary}')
    build = binary.parent.parent
    target = binary.name.removeprefix('lib').removesuffix('.so')
    scripts = sorted((build / 'src/CMakeFiles' / (target + '.dir')).rglob('*.Release.cmake'))
    if not scripts:
        raise RuntimeError(f'Missing original generated compile settings: {binary}')
    settings = []
    for script in scripts:
        text = script.read_text()
        fields = {}
        for name in ('SYCL_executable', 'SYCL_host_compiler', 'SYCL_compile_flags', 'CMAKE_HOST_FLAGS', 'SYCL_include_dirs', 'SYCL_compile_definitions'):
            match = re.search(r'^set\(' + name + r' (.*)\)(?: #.*)?$', text, re.MULTILINE)
            if match:
                fields[name] = match.group(1)
        for flag in ('-fhonor-nans', '-fhonor-infinities', '-fno-associative-math', '-fno-approx-func', '-no-ftz'):
            if flag not in fields['SYCL_compile_flags'].split(';'):
                raise RuntimeError(f'Original source precision flag missing: {script}: {flag}')
        settings.append({'path': str(script), 'sha256': digest(script), 'fields': fields})
    command = ['ninja', '-C', str(build), '-t', 'commands', 'src/' + binary.name]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    output = artifact / ('source-before-v3-' + target + '-ninja-commands.log')
    output.write_text(result.stdout)
    output.chmod(0o644)
    links = [line for line in result.stdout.splitlines() if '-fsycl-link' in line and target in line]
    if len(links) != 1:
        raise RuntimeError(f'Missing original actual device-link command: {target}')
    cache_path = build / 'CMakeCache.txt'
    cache = read_cache(cache_path)
    compiler_record = next((build / 'CMakeFiles').glob('*/CMakeCXXCompiler.cmake')).read_text()
    identity = re.search(r'set\(CMAKE_CXX_COMPILER_ID "([^"]+)"\)', compiler_record).group(1)
    sites = list((artifact / 'source-venv/lib').glob('python*/site-packages/sgl_kernel'))
    if len(sites) != 1 or digest(sites[0] / binary.name) != expected:
        raise RuntimeError(f'Actual BEFORE installed source native differs: {binary.name}')
    return {'native': str(binary), 'native_sha256': expected, 'before_installed_native': str(sites[0] / binary.name), 'before_installed_sha_verified': True, 'build': str(build), 'cache_sha256': digest(cache_path), 'host_cxx_compiler_id_from_original_generated_cmake': identity, 'compiler_cache': {name: value for name, value in cache.items() if name in ('CMAKE_CXX_COMPILER', 'CMAKE_C_COMPILER', 'SYCL_COMPILER', 'SYCL_HOST_COMPILER', 'XPU_SYCL_COMPILER') or 'CUTLASS' in name.upper()}, 'generated_compile_settings': settings, 'actual_ninja_inspection_command': command, 'actual_ninja_inspection_exit': 0, 'ninja_command_output': str(output), 'ninja_command_output_sha256': digest(output), 'actual_device_link_command': links[0], 'historical_compiler_binary_hash': 'not available from immutable historical compilation; do not substitute current compiler hash'}


def main():
    root = pathlib.Path(__file__).resolve().parent.parent
    historical = pathlib.Path('/root/worktrees/sgl-kernel-xpu/review-validation-artifacts/accuracy-20261016/muse_mxfp4_w4a4')
    w4_artifact = root / 'mxfp4-w4a4'
    w4_package = json.loads((w4_artifact / 'source-package-provenance.json').read_text())
    name = 'libsgl-ops-sycl-moe_grouped_mm_mxfp4_w4a4.so'
    record = w4_package['reused_binary_hashes'][name]
    w4_native = inspect_native(w4_artifact, pathlib.Path(record['path']), record['sha256'])
    routing_artifact = root / 'routing-quant'
    routing_package = json.loads((routing_artifact / 'source-package-provenance.json').read_text())
    reviewed = pathlib.Path('/root/worktrees/sgl-kernel-xpu/review-validation-artifacts/cleanup-20261016/muse_routing_quant')
    candidates = {}
    for path in (historical / 'full-default-binaries.sha256', reviewed / 'after-built-sha256.txt'):
        for line in path.read_text().splitlines():
            expected, filename = line.split(maxsplit=1)
            candidates[pathlib.Path(filename).name] = pathlib.Path(filename)
    natives = []
    for name in ('libsgl-ops-sycl-TopKSigMoid.so', 'libsgl-ops-sycl-mxfp4_moe_scale_layout.so', 'libsgl-ops-sycl-per_token_group_quant_fp4.so'):
        binary = routing_artifact / 'source-quant-build/src' / name if 'per_token' in name else candidates[name]
        natives.append(inspect_native(routing_artifact, binary, routing_package['binary_hashes'][name]))
    binding_path = routing_artifact / 'source-common-compile.command.json'
    binding = json.loads(binding_path.read_text())
    if binding[0] != 'g++':
        raise RuntimeError('Original routing registration compiler command changed')
    wrapper = routing_artifact / 'source_icx_monitor.sh'
    if '/opt/gfx-deps/dpcpp/bin/icx' not in wrapper.read_text():
        raise RuntimeError('Original quant compiler wrapper no longer forwards real icx')
    reports = {
        'mxfp4-w4a4': {'source_revision': w4_package['source_head'], 'native_builds': [w4_native], 'original_full_source_build_command': json.loads((w4_artifact / 'diagnostic-build-precision-audit.json').read_text())['source_build_command'], 'source_provenance_limits': w4_package['limits']},
        'routing-quant': {'source_revision': '4cb5e929f84ebcebf98ec6702a26f1ee93190828', 'native_builds': natives, 'original_registration_compile_argv': binding, 'original_registration_compile_command_sha256': digest(binding_path), 'quant_driver_wrapper': str(wrapper), 'quant_driver_wrapper_sha256': digest(wrapper), 'quant_driver_actual_executable': '/opt/gfx-deps/dpcpp/bin/icx', 'source_provenance_limits': routing_package['reuse']},
    }
    for module, report in reports.items():
        artifact = root / module
        report.update({'module': module, 'after_version': 'canonical-v3-icpx', 'kind': 'BEFORE original complete SOURCE native/compiler/flags provenance, CPU-only read audit', 'source_wheel_sha256': w4_package['wheel_sha256'] if module == 'mxfp4-w4a4' else routing_package['wheel_sha256'], 'baseline_policy': 'Original real SOURCE package remains unchanged; native driver and GNU host are documented from commands, never guessed from Torch family token', 'source_rebuild_or_native_replacement': False, 'source_full_or_native_build_launched': False, 'gpu_launched': False, 'frozen_pr_or_precision_helper_writes': 0})
        output = artifact / 'source-before-provenance-v3-icpx.json'
        output.write_text(json.dumps(report, indent=2) + '\n')
        output.chmod(0o644)
        print(json.dumps({'module': module, 'report': str(output), 'report_sha256': digest(output), 'native_count': len(report['native_builds']), 'compiler_settings': [record['generated_compile_settings'][0]['fields'] for record in report['native_builds']]}), flush=True)


if __name__ == '__main__':
    main()
