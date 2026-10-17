import os
from pathlib import Path
from release import execute_frozen, maintenance_gate

namespace = {'__name__': 'frozen_BF_supervisor', '__file__': __file__}
execute_frozen('supervisor.py', namespace)
_original_run = namespace['run']


def run(name, command):
    if name == 'analyze':
        command = [argument.replace('canonical-v4-icpx-formal-gpu4-v1-BAAB', 'canonical-v4-icpx-cleanconfirm-gpu4-v1-BAAB') for argument in command]
    return _original_run(name, command)


namespace['run'] = run
if __name__ == '__main__':
    if os.environ.get('BF_CLEAN_CONFIRMATION_MAIN_GRANTED') != '1':
        raise RuntimeError('Fresh conditional main BF clean authorization required')
    folder = Path(__file__).resolve().parent
    record = maintenance_gate()
    namespace['write_json'](folder / 'CPU_INITIAL_MAINTENANCE.json', record)
    import subprocess
    if subprocess.run(['flock', '-n', '/tmp/intel-gpu-4.lock', 'true']).returncode:
        raise RuntimeError('GPU4 not physically free before CPU prep')
    container = str(folder).replace('/home/syk/', '/root/', 1)
    root = str(folder.parent).replace('/home/syk/', '/root/', 1)
    prefix = ['env', 'SGLANG_MUSE_CONTAINER=codex-muse-bf16-review-20261015', '/home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh']
    for name, filename, source in [('CPU-protocol', 'cpu_protocol.py', False), ('CPU-source-provider', 'cpu_source_provider.py', True)]:
        site = root + '/fresh-source-icpx-v1/site' if source else root + '/canonical-v4-icpx/site'
        command = prefix + ['--workdir', container, '--', 'bash', '-c',
                            'export ONEAPI_DEVICE_SELECTOR=level_zero:cpu PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$2"; '
                            'export LD_LIBRARY_PATH="/opt/gfx-deps/profiling_tools/lib:${LD_LIBRARY_PATH:-}"; '
                            'exec taskset -c 0-15 python -B "$1/$3"', '_', container, site, filename]
        status = run(name, command)
        if status:
            (folder / 'supervisor.exit').write_text(str(status) + '\n')
            raise SystemExit(status)
    namespace['main']()
