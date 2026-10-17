import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent / 'mxfp4-w4a4/aot-regression-main-cpu-v1'


def main():
    assert set(os.sched_getaffinity(0)) == set(range(112, 128))
    roles = {}
    for role in ('source', 'after'):
        summary = json.loads((ROOT / role / 'metadata.json').read_text())
        kernels = []
        for image in summary['images']:
            metadata = yaml.safe_load(Path(image['ze_info_file']).read_text())
            environment = {kernel['name']: kernel for kernel in metadata['kernels']}
            for kernel in image['kernels']:
                name = kernel['section_name'].removeprefix('.text.')
                if 'GemmUniversal' not in name:
                    continue
                assert name in environment
                result = subprocess.run(['c++filt', name], capture_output=True, text=True, check=True)
                demangled = result.stdout.strip()
                assert demangled != name
                pipeline = re.search(r'MainloopMXFPXGroup(?:_xe35)?<(\d+),', demangled)
                dimensions = re.findall(r'cute::C<(\d+)>', demangled)
                assert pipeline and len(dimensions) >= 3
                key = '-'.join([pipeline.group(1), *dimensions[:3]])
                kernels.append({'key': key, 'name': name, 'demangled_name': demangled,
                                'ISA_bytes': kernel['ISA_bytes'], 'ISA_sha256': kernel['ISA_sha256'],
                                'execution_env': environment[name].get('execution_env', {}),
                                'per_thread_memory_buffers': environment[name].get('per_thread_memory_buffers', []),
                                'per_thread_payload_arguments': environment[name].get('per_thread_payload_arguments', []),
                                'user_attributes': environment[name].get('user_attributes', {})})
        assert len(kernels) == 7 and len({kernel['key'] for kernel in kernels}) == 7
        roles[role] = {kernel['key']: kernel for kernel in kernels}
    assert roles['source'].keys() == roles['after'].keys()
    rows = []
    for key in sorted(roles['source']):
        before = roles['source'][key]
        after = roles['after'][key]
        rows.append({'pipeline_and_tile': key, 'source': before, 'after': after,
                     'execution_env_equal': before['execution_env'] == after['execution_env'],
                     'per_thread_memory_buffers_equal': before['per_thread_memory_buffers'] == after['per_thread_memory_buffers'],
                     'user_attributes_equal': before['user_attributes'] == after['user_attributes'],
                     'instruction_sections_identical': before['ISA_sha256'] == after['ISA_sha256']})
    report = {'actual_exit': 0, 'CPU_only': True, 'GPU_or_compiler_started': False,
              'kernel_identity_mapping': 'demangled pipeline stages plus first three tile dimensions; seven unique GEMM pairs; Intel symbol-table entry excluded',
              'runtime_selected_instance_grid_or_performance_cause_proven': False,
              'metadata_extraction_summary_sha256': hashlib.sha256((ROOT / 'SUMMARY.json').read_bytes()).hexdigest(),
              'rows': rows}
    with (ROOT / 'METADATA_COMPARISON_v3.json').open('x') as output:
        output.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'actual_exit': 0, 'pairs': len(rows), 'rows': [
        {'pipeline_and_tile': row['pipeline_and_tile'], 'source_ISA_bytes': row['source']['ISA_bytes'],
         'after_ISA_bytes': row['after']['ISA_bytes'], 'execution_env_equal': row['execution_env_equal'],
         'per_thread_memory_buffers_equal': row['per_thread_memory_buffers_equal']}
        for row in rows]}))


if __name__ == '__main__':
    main()
