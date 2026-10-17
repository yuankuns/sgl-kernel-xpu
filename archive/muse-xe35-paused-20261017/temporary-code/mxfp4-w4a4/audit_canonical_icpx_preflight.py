import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

from audit_canonical_v2_preflight import parser_only


ROOT = pathlib.Path(__file__).resolve().parent.parent
COMMON = ROOT / 'mxfp4-w4a4'
VERSION = 'canonical-v4-icpx'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', choices=('canonical-v3-icpx', 'canonical-v4-icpx'), default=VERSION)
    args = parser.parse_args()
    version = args.version
    tools = ['check_canonical_package.py', 'check_source_baseline.py', 'wait_canonical_performance_grant.py']
    for filename in tools:
        path = COMMON / filename
        compile(path.read_text(), str(path), 'exec')
        result = subprocess.run([sys.executable, str(path), '--help'], capture_output=True, text=True, check=True)
        assert '--version' in result.stdout
    for filename in ('launch_canonical_performance_v1.sh', 'run_canonical_validation_v1.sh'):
        subprocess.run(['bash', '-n', str(COMMON / filename)], check=True)
    main_audit = json.loads((ROOT / 'ICPX_SOURCE_FREEZE_AUDIT_v1.json').read_text())
    assert main_audit['production_flag_body_unchanged'] is True
    assert main_audit['torch_family_token'] == 'icx (actual executable is icpx)'
    main_components = {record['component']: record for record in main_audit['components']}
    for module in ('mxfp4-w4a4', 'routing-quant'):
        artifact = ROOT / module
        frozen = json.loads((artifact / ('canonical-source-freeze-' + version.removeprefix('canonical-') + '.json')).read_text())
        prior = 'v3-icpx' if version == 'canonical-v4-icpx' else 'v2'
        previous = json.loads((artifact / ('canonical-source-freeze-' + prior + '.json')).read_text())
        delta = [name for name in frozen['files'] if frozen['files'][name] != previous['files'].get(name)]
        assert delta == (['src/sycl/HCHead.cpp'] if prior == 'v3-icpx' else ['cmake/BuildFlags.cmake']), delta
        assert (previous if prior == 'v3-icpx' else frozen)['source_digest'] == main_components[module]['source_digest']
        target = ROOT.parent / 'pr' / ('muse-' + module + '-xe35')
        for name, record in frozen['files'].items():
            path = target / name
            assert not path.is_symlink() and hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256'] and path.stat().st_size == record['size'] and path.stat().st_mode & 0o777 == record['mode'], str(path)
        print(json.dumps({'module': module, 'version': version, 'readonly_source_inventory_verified': len(frozen['files']), 'source_digest': frozen['source_digest'], 'delta_from_prior': delta, 'changed_file_count': len((artifact / 'changed-files.txt').read_text().splitlines())}), flush=True)
        for phase in ('precision', 'performance'):
            for options, expected in [(['--version', version, '--phase', phase, '--serving-batch', '16'], 2), (['--version', '../bad', '--phase', phase], 2), (['--version', 'canonical-v2', '--phase', phase], 125), (['--version', 'canonical-v3-icpx', '--phase', phase], 125)]:
                command = ['bash', str(COMMON / 'launch_canonical_performance_v1.sh'), module] + options
                result = subprocess.run(command, capture_output=True, text=True)
                assert result.returncode == expected, (command, result.returncode, result.stderr)
                print(json.dumps({'negative_preflight': command, 'actual_exit': result.returncode}), flush=True)
            mode = ['--precision-only'] if phase == 'precision' else []
            prefix = version + '-' + phase + '-'
            base = ['--implementation', 'port', '--round', '0', '--evidence-prefix', prefix]
            parser_only(artifact / 'compare_production.py', base + mode, 0)
            parser_only(artifact / 'compare_production.py', base + mode + ['--port-original-abi'], 2)
            parser_only(artifact / 'compare_production.py', base + ([] if mode else ['--precision-only']), 2)
            parser_only(artifact / 'summarize.py', ['--prefix', prefix] + mode, 0)
        if not (artifact / version / 'site/sgl_kernel/__init__.py').exists():
            command = ['bash', str(COMMON / 'launch_canonical_performance_v1.sh'), module, '--version', version, '--phase', 'precision']
            result = subprocess.run(command, capture_output=True, text=True)
            assert result.returncode == 125
            print(json.dumps({'normal_package_not_ready_preflight': command, 'actual_exit': result.returncode, 'no_wrapper_or_gpu_started': True}), flush=True)
    print(json.dumps({'kind': 'CPU-only icpx syntax/parser/readonly-freeze/negative-launcher proof; not full/package/GPU acceptance', 'version': version, 'normal_package_check': 'not run', 'source_baseline_check': 'not run', 'gpu': 'not run or acquired', 'configure_or_build': 'not run', 'frozen_pr_writes': 0}), flush=True)


if __name__ == '__main__':
    main()
