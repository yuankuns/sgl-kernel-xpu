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
    namespace['main']()
