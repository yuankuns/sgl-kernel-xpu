import importlib.util
import re
import subprocess
from pathlib import Path


ROOT = Path('/home/syk/worktrees/sgl-kernel-xpu-up/muse-upstream-artifacts-20261017')
spec = importlib.util.spec_from_file_location('simple_port_metadata', ROOT / 'port_simple.py')
metadata = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metadata)


def add_file(filename, content):
    if filename.exists():
        raise RuntimeError(f'refusing to overwrite diagnostic source: {filename}')
    patch = (
        '*** Begin Patch\n'
        + f'*** Add File: {filename}\n'
        + ''.join('+' + line + '\n' for line in content.splitlines())
        + '*** End Patch\n'
    )
    subprocess.run(['apply_patch'], input=patch, text=True, check=True)


def main():
    for component, info in metadata.SPECS.items():
        output = ROOT / component / 'native-diagnostic'
        port_root = '/root/worktrees/sgl-kernel-xpu-up/pr/muse-' + component + '-xe35'
        declarations = metadata.additions(info['branch'], 'include/sgl_kernel_ops.h')
        signatures = re.findall(
            r'((?:void|std::tuple<torch::Tensor, torch::Tensor>)\s+)(\w+)\((.*?)\);',
            declarations,
            flags=re.DOTALL,
        )
        cmake = f'''cmake_minimum_required(VERSION 3.19.2)
project(muse_native_port_diagnostic LANGUAGES CXX)
set(SYCL_LINK_LIBRARIES_KEYWORD PRIVATE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE Release)
set(SGL_OPS_XPU_ROOT "{port_root}")
set(DPCPP_SYCL_TARGET cri)
set(PYTHON3_EXECUTABLE /opt/gfx-deps/venv/bin/python)
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(COMMAND ${{PYTHON3_EXECUTABLE}} -c "import torch; print(torch.utils.cmake_prefix_path)" OUTPUT_VARIABLE TORCH_PY_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
list(APPEND CMAKE_PREFIX_PATH ${{TORCH_PY_PREFIX}}/Torch)
find_package(Torch REQUIRED)
list(APPEND CMAKE_MODULE_PATH "{port_root}/cmake/Modules")
include("{port_root}/cmake/SYCL.cmake")
include("{port_root}/cmake/BuildFlags.cmake")
add_compile_definitions(SYCL_INTEL_TARGET=35 __SYCL_TARGET_INTEL_GPU_CRI__)
include_directories(${{SGL_OPS_XPU_ROOT}}/include ${{SGL_OPS_XPU_ROOT}}/src ${{TORCH_XPU_OPS_INCLUDE_DIRS}} ${{ATen_XPU_INCLUDE_DIRS}} ${{TORCH_INCLUDE_DIRS}} ${{SYCL_INCLUDE_DIRS}})
'''
        for old_file, new_file in info['files'].items():
            if not old_file.endswith('.cpp'):
                continue
            original = metadata.git_text('show', info['branch'] + ':' + old_file)
            wrapper = '#include <torch/all.h>\n\n#include "sgl_kernel_export.h"\n\nnamespace at::native::xpu {\n'
            definitions = []
            for return_type, name, arguments in signatures:
                if not re.search(r'\b' + re.escape(name) + r'\s*\(', original):
                    continue
                port_name = info['names'][name]
                wrapper += f'SGL_KERNEL_EXPORT {return_type}{port_name}({arguments});\n'
                forwarded = ', '.join(
                    re.search(r'(\w+)\s*$', argument.strip()).group(1)
                    for argument in arguments.split(',')
                )
                call = f'{port_name}({forwarded})'
                statement = call if return_type.strip() == 'void' else 'return ' + call
                definitions.append(
                    f'SGL_KERNEL_EXPORT {return_type}{name}({arguments}) {{\n'
                    + '  ' + statement + ';\n}\n'
                )
            wrapper += '\n' + '\n'.join(definitions) + '\n}\n'
            target = Path(new_file).stem
            bridge = target + '_source_api.cpp'
            add_file(output / 'xe35' / bridge, wrapper)
            cmake += f'''
sycl_add_library(sgl-ops-sycl-{target} ${{SYCL_OFFLINE_COMPILER_FLAGS}} ${{SYCL_DEVICE_LINK_FLAGS}} SHARED SYCL_SOURCES "{port_root}/{new_file}" "${{CMAKE_CURRENT_SOURCE_DIR}}/xe35/{bridge}")
target_compile_options(sgl-ops-sycl-{target} PRIVATE ${{TORCH_XPU_OPS_FLAGS}})
target_link_libraries(sgl-ops-sycl-{target} PRIVATE ${{TORCH_LIBRARIES}} c10 torch torch_cpu ${{SYCL_LIBRARY}})
set_target_properties(sgl-ops-sycl-{target} PROPERTIES INSTALL_RPATH "$ORIGIN" BUILD_WITH_INSTALL_RPATH TRUE)
'''
        add_file(output / 'CMakeLists.txt', cmake)
        for source_file in info['files']:
            if source_file.startswith('tests/') and source_file != 'tests/test_memory_benchmark_scope.py':
                add_file(output / source_file, metadata.git_text('show', info['branch'] + ':' + source_file))


if __name__ == '__main__':
    main()
