import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess

import stage_fresh_source_site as staging
import fresh_current_source_resource_monitor_v1 as resource_monitor


SUPPORT = staging.SUPPORT
resource_monitor.attach(SUPPORT)
ORIGINAL_POLICY = SUPPORT.policy
PREPARATION = SUPPORT.ROOT / 'routing-quant/fresh-current-source-native-v3-preparation'
STEMS = ['TopKSigMoid', 'per_token_group_quant_fp4', 'mxfp4_moe_scale_layout', 'TripleOps']


def policy(artifact, stage, stems):
    assert stems == STEMS
    ORIGINAL_POLICY(artifact, stage, stems[:-1])
    reference = json.loads((PREPARATION / 'TripleOps-original-generated-flags.json').read_text())
    target = 'sgl-ops-sycl-TripleOps'
    build = stage / 'build'
    script = next((build / 'src/CMakeFiles' / (target + '.dir')).rglob('*.Release.cmake'))
    text = script.read_text()
    fields = {name: re.search(r'^set\(' + name + r' (.*)\)(?: #.*)?$', text, re.MULTILINE).group(1)
              for name in ('SYCL_executable', 'SYCL_host_compiler', 'SYCL_compile_flags', 'CMAKE_HOST_FLAGS', 'SYCL_compile_definitions')}
    assert fields['SYCL_executable'] == fields['SYCL_host_compiler'] == '"' + SUPPORT.ICPX + '"'
    for name in ('SYCL_compile_flags', 'CMAKE_HOST_FLAGS', 'SYCL_compile_definitions'):
        assert fields[name] == reference['fields'][name], (name, fields[name], reference['fields'][name])
    lines = (build / 'build.ninja').read_text().splitlines()
    device = next(lines[index + 1] for index, line in enumerate(lines) if line.startswith('build ') and
                  target in line and 'sycl_device_obj.o' in line and ': CUSTOM_COMMAND' in line)
    def device_options(command):
        arguments = shlex.split(command)
        return [argument.replace('-fsycl-max-parallel-link-jobs=256', '-fsycl-max-parallel-link-jobs=1')
                for argument in arguments[arguments.index('-fPIC'):arguments.index('-o')] if not argument.endswith('.o')]
    assert SUPPORT.ICPX + ' -fPIC -fsycl ' in device
    assert '-fsycl-max-parallel-link-jobs=1' in device
    assert device_options(device) == device_options(reference['actual_device_link_command'])
    commands = subprocess.check_output(['ninja', '-C', str(build), '-t', 'commands', target], text=True)
    assert 'common_ops' not in commands and 'torch_extension' not in commands
    assert '/usr/bin/c++' not in commands and not re.search(r'\b(g\+\+|gcc)\b', commands)
    (stage / (target + '.expanded.commands')).write_text(commands)
    report_path = stage / 'compiler-policy-verified.json'
    report = json.loads(report_path.read_text())
    report['targets'].append({'target': target, 'fields': fields, 'device_link': device,
                              'generated_settings_sha256': SUPPORT.digest(script),
                              'original_flags_and_definitions_byte_equal': True,
                              'original_device_math_AOT_GRF_options_equal': True,
                              'only_device_parallel_resource_option_changed': '256 to approved inner1',
                              'reference_kind': 'actual original checkout generated commands, not provenance of reused legacy binary',
                              'reference_SHA256': SUPPORT.digest(PREPARATION / 'TripleOps-original-generated-flags.json')})
    SUPPORT.record(report_path, report)


def package(artifact, stage, stems):
    status = 1
    try:
        staging.stage_site(artifact, stage, stems)
        native = stage / 'site/sgl_kernel/libsgl-ops-sycl-TripleOps.so'
        symbols = subprocess.check_output(['nm', '-a', '-C', str(native)], text=True)
        (stage / 'fresh-TripleOps.nm-demangled.txt').write_text(symbols)
        selected = sorted(set(line for line in symbols.splitlines() if 'silu_mul_dpcpp_functor<sycl::_V1::ext::oneapi::bfloat16,' in line))
        assert selected and all('silu_mul_dpcpp_functor<sycl::_V1::ext::oneapi::bfloat16, float>' in line for line in selected)
        SUPPORT.record(stage / 'fresh-helper-type-verification.json', {'TripleOps_SHA256': SUPPORT.digest(native),
                       'BF16_silu_actual_accumulator_type': 'float', 'actual_binary_symbols': selected,
                       'GPU_initialized': False, 'precision_not_run': True})
        status = 0
    finally:
        (stage / 'isolated-site-stage.exit').write_text(str(status) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--container', action='store_true')
    parser.add_argument('--module', choices=['routing-quant'], required=True)
    parser.add_argument('--stage-name', choices=['fresh-current-source-icpx-native-v3'], required=True)
    options = parser.parse_args()
    SUPPORT.STAGE_NAME = options.stage_name
    SUPPORT.ENTRY_SCRIPT = Path(__file__).name
    family = SUPPORT.FAMILIES['routing-quant']
    SUPPORT.FAMILIES = {'routing-quant': (family[0], family[1], STEMS)}
    SUPPORT.policy = policy
    SUPPORT.package = package
    assert os.environ.get('LD_LIBRARY_PATH', '').startswith('/opt/gfx-deps/profiling_tools/lib:') if options.container else True
    raise SystemExit(SUPPORT.container(options.module) if options.container else SUPPORT.host(options.module))
