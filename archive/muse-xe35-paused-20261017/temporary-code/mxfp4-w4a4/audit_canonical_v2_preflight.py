import ast
import contextlib
import hashlib
import io
import json
import pathlib
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parent.parent
COMMON = ROOT / 'mxfp4-w4a4'


def parser_only(path, arguments, expected):
    tree = ast.parse(path.read_text(), filename=str(path))
    imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom)) and not (isinstance(node, ast.Import) and any(alias.name == 'torch' for alias in node.names))]
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ('main', 'main_xe35'))
    body = []
    for node in function.body:
        if isinstance(node, ast.Import) and any(alias.name == 'torch' for alias in node.names):
            break
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'artifact' for target in node.targets):
            break
        body.append(node)
    function.body = body
    isolated = ast.fix_missing_locations(ast.Module(body=imports + [function], type_ignores=[]))
    namespace = {'__file__': str(path)}
    original_arguments = sys.argv
    sys.argv = [str(path)] + arguments
    status = 0
    errors = io.StringIO()
    try:
        exec(compile(isolated, str(path), 'exec'), namespace)
        with contextlib.redirect_stderr(errors):
            namespace[function.name]()
    except SystemExit as exception:
        status = exception.code
    finally:
        sys.argv = original_arguments
    if status != expected:
        raise AssertionError((path.name, arguments, status, expected, errors.getvalue()))
    print(json.dumps({'parser_only': str(path), 'arguments': arguments, 'actual_exit': status, 'no_torch_import_or_kernel_execution': True}), flush=True)


def main():
    launcher = COMMON / 'launch_canonical_performance_v1.sh'
    for filename in ('launch_canonical_performance_v1.sh', 'run_canonical_validation_v1.sh'):
        subprocess.run(['bash', '-n', str(COMMON / filename)], check=True)
    for path in [COMMON / 'check_canonical_package.py', COMMON / 'check_source_baseline.py', COMMON / 'wait_canonical_performance_grant.py'] + [ROOT / module / filename for module in ('mxfp4-w4a4', 'routing-quant') for filename in ('compare_production.py', 'summarize.py')]:
        compile(path.read_text(), str(path), 'exec')
        print(json.dumps({'syntax': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}), flush=True)
    for module in ('mxfp4-w4a4', 'routing-quant'):
        artifact = ROOT / module
        freeze = json.loads((artifact / 'canonical-source-freeze-v2.json').read_text())
        previous = json.loads((artifact / 'canonical-source-freeze-v1.json').read_text())
        changed = [name for name in freeze['files'] if freeze['files'][name] != previous['files'].get(name)]
        if changed != ['src/sycl/Device.cpp']:
            raise AssertionError(('Unexpected v2 change', changed))
        target = ROOT.parent / 'pr' / ('muse-' + module + '-xe35')
        for name, record in freeze['files'].items():
            path = target / name
            if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256'] or path.stat().st_size != record['size'] or path.stat().st_mode & 0o777 != record['mode']:
                raise AssertionError(('Actual PR freeze mismatch', str(path)))
        print(json.dumps({'module': module, 'v2_inventory_readonly_verified': len(freeze['files']), 'source_digest': freeze['source_digest'], 'v1_to_v2_changed_files': changed}), flush=True)
        for phase in ('precision', 'performance'):
            for options, expected in [(['--version', 'canonical-v2', '--phase', phase, '--serving-batch', '16'], 2), (['--version', '../bad', '--phase', phase], 2), (['--version', 'canonical-v1', '--phase', phase], 125)]:
                command = ['bash', str(launcher), module] + options
                result = subprocess.run(command, capture_output=True, text=True)
                if result.returncode != expected:
                    raise AssertionError((command, result.returncode, result.stdout, result.stderr))
                print(json.dumps({'host_launcher_negative_preflight': command, 'actual_exit': result.returncode, 'stderr': result.stderr}), flush=True)
        comparator = artifact / 'compare_production.py'
        summarizer = artifact / 'summarize.py'
        for version in ('canonical-v2', 'canonical-v3'):
            for phase in ('precision', 'performance'):
                prefix = version + '-' + phase + '-'
                mode = ['--precision-only'] if phase == 'precision' else []
                base = ['--implementation', 'port', '--round', '0', '--evidence-prefix', prefix]
                parser_only(comparator, base + mode, 0)
                parser_only(comparator, base + mode + ['--port-original-abi'], 2)
                parser_only(comparator, base + mode + ['--serving-batch', '16'], 2)
                parser_only(comparator, base + ([] if mode else ['--precision-only']), 2)
                parser_only(summarizer, ['--prefix', prefix] + mode, 0)
                parser_only(summarizer, ['--prefix', prefix] + ([] if mode else ['--precision-only']), 2)
    for filename in ('check_canonical_package.py', 'check_source_baseline.py', 'wait_canonical_performance_grant.py'):
        result = subprocess.run([sys.executable, str(COMMON / filename), '--help'], capture_output=True, text=True, check=True)
        if '--version' not in result.stdout:
            raise AssertionError(('Missing version argument', filename))
    print(json.dumps({'kind': 'CPU-only syntax/parser/freeze/negative-launcher audit', 'normal_package_checker': 'not run; no complete module v2 install yet', 'official_full': 'not launched by worker', 'gpu': 'not launched or acquired', 'actual_pr_writes': 0}), flush=True)


if __name__ == '__main__':
    main()
