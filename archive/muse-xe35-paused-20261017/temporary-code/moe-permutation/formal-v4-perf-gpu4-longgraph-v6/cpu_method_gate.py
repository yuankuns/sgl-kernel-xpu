import ast
import json
import os
import sys
import time
from pathlib import Path
from release import execute_frozen, maintenance_gate

FOLDER = Path(__file__).resolve().parent


def main():
    if os.environ.get('ONEAPI_DEVICE_SELECTOR') != 'level_zero:cpu':
        raise RuntimeError('CPU-only method gate required')
    approved = FOLDER.parent / 'longgraph-release-interface-fix-cpu-v1/checks_return_candidate.py'
    candidate = FOLDER / 'checks_return_candidate.py'
    if candidate.read_bytes() != approved.read_bytes():
        raise RuntimeError('Approved interface repair candidate changed')
    checks_tree = ast.parse((FOLDER / 'checks.py').read_text())
    grant_function = next(node for node in checks_tree.body if isinstance(node, ast.FunctionDef) and node.name == 'grant_preflight')
    first = grant_function.body[0]
    last = grant_function.body[-1]
    if not isinstance(first, ast.Assign) or first.targets[0].id != 'result' or first.value.func.id != '_original_grant_preflight':
        raise RuntimeError('Original grant result forwarding assignment missing')
    if not isinstance(last, ast.Return) or last.value.id != 'result':
        raise RuntimeError('Original four-tuple return forwarding missing')
    for path in FOLDER.glob('*.py'):
        ast.parse(path.read_text())
    namespace = {'__name__': 'CPU_parser_contract', '__file__': str(FOLDER / 'run_case.py')}
    execute_frozen('run_case.py', namespace)
    previous_argv = sys.argv[:]
    previous_mask = os.environ.get('ZE_AFFINITY_MASK')
    os.environ['ZE_AFFINITY_MASK'] = '4'
    prefix = ['CPU_parser_contract', 'source', 'canonical-v4-icpx-CPU-contract', '--grant-file', str(FOLDER / 'fresh-grant.json')]
    try:
        sys.argv = prefix[:]
        args = namespace['parse_args_xe35']()
        assert (args.calls, args.repeat, args.warmup, args.serving_batch, args.isl, args.osl, args.tp, args.ep) == (1000, 41, 5, 8, 4096, 16, 1, 1)
        rejected = []
        for arguments in [['--calls', '100'], ['--repeat', '30'], ['--warmup', '4'], ['--serving-batch', '1'], ['--accuracy-only']]:
            sys.argv = prefix + arguments
            try:
                namespace['parse_args_xe35']()
            except SystemExit as error:
                assert error.code == 2
                rejected.append(arguments)
            else:
                raise RuntimeError('Invalid method arguments accepted: ' + str(arguments))
    finally:
        sys.argv = previous_argv
        if previous_mask is None:
            os.environ.pop('ZE_AFFINITY_MASK', None)
        else:
            os.environ['ZE_AFFINITY_MASK'] = previous_mask
    graph = {'__name__': 'CPU_graph_contract', '__file__': str(FOLDER / 'graph_contract.py')}
    execute_frozen('graph_contract.py', graph)
    assert (graph['CALLS'], graph['REPEATS'], graph['WARMUP']) == (1000, 41, 5)
    shell = (FOLDER / 'gpu_container.sh').read_text()
    assert 'for cycle in 1 2 3;' in shell and 'labels=(source port port source)' in shell
    assert '--warmup 5 --calls 1000 --repeat 41' in shell and 'taskset -c 96-111' in shell
    assert 'torch' not in sys.modules and 'sgl_kernel' not in sys.modules
    record = {'epoch': time.time(), 'actual_exit': 0, 'accepted_B8_method': [1000, 41, 5, 8, 4096, 16, 1, 1],
              'negative_method_rejections': rejected, 'three_BAAB_six_cases_SHA_pinned': True,
              'actual_DOT1000_gate_before_events_implementation_unchanged': True,
              'actual_native_DOT_capture_not_yet_run': True, 'approved_return_fix_plus_new_release_metadata_only': True,
              'maintenance': maintenance_gate(), 'GPU_started': False}
    (FOLDER / 'CPU_METHOD_CONTRACT_GATE.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'actual_method_gate_exit': 0, 'GPU_started': False}), flush=True)


if __name__ == '__main__':
    main()
