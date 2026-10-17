import ast
import inspect
import json
import os
from pathlib import Path
from contracts import ROOT, ART, digest, write_json, scope_guard


def main():
    scope_guard()
    assert os.environ.get('ONEAPI_DEVICE_SELECTOR') == 'level_zero:cpu'
    assert os.sched_getaffinity(0) == set(range(96, 112))
    import torch
    import sgl_kernel
    from sgl_kernel import utils
    expected = ART / 'matched-native-icpx-v1/preopt/site/sgl_kernel'
    assert Path(sgl_kernel.__file__).resolve().parent == expected.resolve()
    binding = torch._C._XpuDeviceProperties
    assert not hasattr(binding, 'major') and not hasattr(binding, 'minor')
    assert callable(utils.is_xe3_arch) and callable(utils.get_device_capability)
    assert not inspect.signature(utils.is_xe3_arch).parameters
    assert not inspect.signature(utils.get_device_capability).parameters
    source = inspect.getsource(utils.is_xe3_arch)
    assert 'torch.ops.sgl_kernel.query_device.default(device)' in source
    assert 'major == 3 and minor == 5' in source
    query_schema = torch.ops.sgl_kernel.query_device.default._schema
    schema = str(query_schema)
    assert query_schema.name == 'sgl_kernel::query_device'
    assert len(query_schema.arguments) == 1 and str(query_schema.arguments[0].type) == 'int'
    assert len(query_schema.returns) == 2 and all(str(item.type) == 'int' for item in query_schema.returns)
    driver = ast.parse((ROOT / 'accuracy_driver.py').read_text())
    arch_asserts = [node for node in ast.walk(driver) if isinstance(node, ast.Assert) and isinstance(node.test, ast.Call) and isinstance(node.test.func, ast.Name) and node.test.func.id == 'is_xe3_arch']
    assert len(arch_asserts) == 1
    assert not any(isinstance(node, ast.Attribute) and node.attr in ('get_device_properties', 'major', 'minor') for node in ast.walk(driver))
    declarations = []
    for path in (Path(torch.__file__).parent / '_C/__init__.pyi', Path(torch.__file__).parent / '_C/_xpu.pyi'):
        if path.exists():
            lines = path.read_text().splitlines()
            excerpts = []
            for index, line in enumerate(lines):
                if '_XpuDeviceProperties' in line:
                    excerpts.append({'line': index + 1, 'text': '\n'.join(lines[max(0, index - 2):index + 35])})
            declarations.append({'path': str(path), 'sha256': digest(path), 'excerpts': excerpts})
    assert not torch.xpu.is_initialized()
    record = {'actual_exit': 0, 'torch_version': torch.__version__, 'torch_git_version': torch.version.git_version,
              'binding_class': str(binding), 'actual_binding_class_attributes': sorted(dir(binding)),
              'binding_has_major': hasattr(binding, 'major'), 'binding_has_minor': hasattr(binding, 'minor'),
              'get_device_properties_signature': str(inspect.signature(torch.xpu.get_device_properties)),
              'get_device_properties_installed_source': inspect.getsource(torch.xpu.get_device_properties),
              'utils_source_sha256': digest(Path(utils.__file__)),
              'is_xe3_arch_signature': str(inspect.signature(utils.is_xe3_arch)), 'is_xe3_arch_installed_source': source,
              'get_device_capability_signature': str(inspect.signature(utils.get_device_capability)),
              'get_device_capability_installed_source': inspect.getsource(utils.get_device_capability),
              'actual_registered_query_device_schema': schema, 'installed_declarations': declarations,
              'new_driver_real_arch_assert': True, 'GPU_query_called': False, 'GPU_initialized': False,
              'mock_hardware': False}
    write_json(ROOT / 'INSTALLED_API_CPU_CONTRACT.json', record)
    print(json.dumps({'installed_api_contract': 0, 'actual_registered_query_schema': schema, 'GPU_query_called': False}))


if __name__ == '__main__':
    main()
