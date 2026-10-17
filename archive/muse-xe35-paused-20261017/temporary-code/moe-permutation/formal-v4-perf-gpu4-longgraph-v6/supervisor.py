from release import execute_frozen, maintenance_gate

namespace = {'__name__': 'frozen_supervisor', '__file__': __file__}
execute_frozen('supervisor.py', namespace)
_original_compiler_drain = namespace['compiler_drain_snapshot']


def compiler_drain_snapshot(name):
    record = maintenance_gate()
    previous = _original_compiler_drain(name)
    previous['maintenance_release'] = record
    namespace['write_json'](namespace['ROUND'] / name, previous)
    return previous


namespace['compiler_drain_snapshot'] = compiler_drain_snapshot
if __name__ == '__main__':
    import os
    from pathlib import Path
    if os.environ.get('MOE_LONG_GRAPH_CONFIRMATION_MAIN_GRANTED') != '1':
        raise RuntimeError('New main authorization required')
    maintenance_gate()
    folder = Path(__file__).resolve().parent
    container_folder = str(folder).replace('/home/syk/', '/root/', 1)
    prefix = ['env', 'SGLANG_MUSE_CONTAINER=codex-muse-bf16-review-20261015',
              '/home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh']
    for label, filename in [('CPU-interface', 'cpu_interface_regression.py'), ('CPU-method', 'cpu_method_gate.py')]:
        command = prefix + ['--workdir', container_folder, '--', 'bash', '-c',
                            'export ONEAPI_DEVICE_SELECTOR=level_zero:cpu PYTHONDONTWRITEBYTECODE=1; '
                            'exec taskset -c 96-111 python -B "$1/$2"', '_', container_folder, filename]
        status = namespace['run'](label, command)
        if status:
            (folder / 'supervisor.exit').write_text(str(status) + '\n')
            raise SystemExit(status)
    namespace['main']()
