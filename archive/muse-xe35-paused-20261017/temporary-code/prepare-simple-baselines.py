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
        raise RuntimeError(f'refusing to replace existing baseline: {filename}')
    patch = (
        '*** Begin Patch\n'
        + f'*** Add File: {filename}\n'
        + ''.join('+' + line + '\n' for line in content.splitlines())
        + '*** End Patch\n'
    )
    subprocess.run(['apply_patch'], input=patch, text=True, check=True)


def main():
    source_worktrees = {
        'memory': 'muse-spark-memory-main-20261014',
        'position': 'muse-spark-position-main-20261014',
        'logits-vocab': 'muse-spark-logits-vocab-main-20261014',
    }
    for component, info in metadata.SPECS.items():
        output = ROOT / component / 'source-baseline'
        declarations = metadata.additions(info['branch'], 'include/sgl_kernel_ops.h')
        signatures = re.findall(
            r'((?:void|std::tuple<torch::Tensor, torch::Tensor>)\s+)(\w+)\((.*?)\);',
            declarations,
            flags=re.DOTALL,
        )
        targets = []
        for old_file, new_file in info['files'].items():
            if not old_file.endswith('.cpp'):
                continue
            original = metadata.git_text('show', info['branch'] + ':' + old_file)
            filename = Path(old_file).name
            add_file(output / 'xe35' / filename, original)
            wrapper = '#include <torch/all.h>\n\nnamespace at::native::xpu {\n'
            definitions = []
            for return_type, name, arguments in signatures:
                if not re.search(r'\b' + re.escape(name) + r'\s*\(', original):
                    continue
                wrapper += f'{return_type}{name}({arguments});\n'
                forwarded = ', '.join(
                    re.search(r'(\w+)\s*$', argument.strip()).group(1)
                    for argument in arguments.split(',')
                )
                call = f'{name}({forwarded})'
                statement = call if return_type.strip() == 'void' else 'return ' + call
                definitions.append(
                    f'{return_type}{info["names"][name]}({arguments}) {{\n'
                    + '  ' + statement + ';\n}\n'
                )
            wrapper += '\n' + '\n'.join(definitions) + '\n}\n'
            bridge = filename.replace('.cpp', '_api_bridge.cpp')
            add_file(output / 'xe35' / bridge, wrapper)
            targets.append((Path(new_file).stem, filename, bridge))
        source_root = '/root/worktrees/sgl-kernel-xpu/' + source_worktrees[component]
        cmake = f'''cmake_minimum_required(VERSION 3.19.2)
project(muse_source_baseline_xe35 LANGUAGES CXX)
set(SYCL_LINK_LIBRARIES_KEYWORD PRIVATE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE Release)
set(SGL_OPS_XPU_ROOT "{source_root}")
set(DPCPP_SYCL_TARGET cri)
set(PYTHON3_EXECUTABLE /opt/gfx-deps/venv/bin/python)
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(COMMAND ${{PYTHON3_EXECUTABLE}} -c "import torch; print(torch.utils.cmake_prefix_path)" OUTPUT_VARIABLE TORCH_PY_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
list(APPEND CMAKE_PREFIX_PATH ${{TORCH_PY_PREFIX}}/Torch)
find_package(Torch REQUIRED)
list(APPEND CMAKE_MODULE_PATH "{source_root}/cmake/Modules")
include("{source_root}/cmake/SYCL.cmake")
include("{source_root}/cmake/BuildFlags.cmake")
add_compile_definitions(SYCL_INTEL_TARGET=35 __SYCL_TARGET_INTEL_GPU_CRI__)
include_directories(${{SGL_OPS_XPU_ROOT}}/src/sycl ${{TORCH_XPU_OPS_INCLUDE_DIRS}} ${{ATen_XPU_INCLUDE_DIRS}} ${{TORCH_INCLUDE_DIRS}} ${{SYCL_INCLUDE_DIRS}})
'''
        for target, filename, bridge in targets:
            cmake += f'''
sycl_add_library(sgl-ops-sycl-{target} ${{SYCL_OFFLINE_COMPILER_FLAGS}} ${{SYCL_DEVICE_LINK_FLAGS}} SHARED SYCL_SOURCES "${{CMAKE_CURRENT_SOURCE_DIR}}/xe35/{filename}" "${{CMAKE_CURRENT_SOURCE_DIR}}/xe35/{bridge}")
target_compile_options(sgl-ops-sycl-{target} PRIVATE ${{TORCH_XPU_OPS_FLAGS}})
target_link_libraries(sgl-ops-sycl-{target} PRIVATE ${{TORCH_LIBRARIES}} c10 torch torch_cpu ${{SYCL_LIBRARY}})
set_target_properties(sgl-ops-sycl-{target} PROPERTIES INSTALL_RPATH "$ORIGIN" BUILD_WITH_INSTALL_RPATH TRUE)
'''
        add_file(output / 'CMakeLists.txt', cmake)


if __name__ == '__main__':
    main()
