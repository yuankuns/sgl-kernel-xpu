import re
import subprocess
from pathlib import Path


ROOT = Path('/home/syk/worktrees/sgl-kernel-xpu-up')
SOURCE = Path('/home/syk/sgl-kernel-xpu')
BASE = 'ab3a26c43a38e6aea7d73cc1349e4b5e210b772b'
SPECS = {
    'memory': {
        'branch': 'muse_memory',
        'names': {
            'write_req_to_token_pool': 'write_req_to_token_pool_xe35',
            'paged_alloc_extend': 'paged_alloc_extend_xe35',
            'paged_alloc_extend_and_write': 'paged_alloc_extend_and_write_xe35',
            'paged_alloc_decode': 'paged_alloc_decode_xe35',
        },
        'files': {
            'src/sycl/PagedAlloc.cpp': 'src/sycl/xe35/PagedAlloc.cpp',
            'src/sycl/WriteReqToTokenPool.cpp': 'src/sycl/xe35/WriteReqToTokenPool.cpp',
            'tests/test_paged_alloc.py': 'tests/test_paged_alloc.py',
            'tests/test_write_req_to_token_pool.py': 'tests/test_write_req_to_token_pool.py',
            'tests/test_memory_benchmark_scope.py': 'tests/test_memory_benchmark_scope.py',
            'benchmark/bench_memory_compare.py': 'benchmark/bench_memory_compare.py',
        },
    },
    'position': {
        'branch': 'muse_position',
        'names': {
            'muse_compute_position': 'compute_position_xe35',
            'muse_compute_position_with_hint': 'compute_position_with_hint_xe35',
            'muse_compute_position_impl': 'compute_position_impl_xe35',
            'MuseComputePositionKernel': 'ComputePositionKernel_xe35',
        },
        'files': {
            'src/sycl/ComputePosition.cpp': 'src/sycl/xe35/ComputePosition.cpp',
            'tests/test_compute_position.py': 'tests/test_compute_position.py',
            'benchmark/bench_position_occupancy.py': 'benchmark/bench_position_occupancy.py',
        },
    },
    'logits-vocab': {
        'branch': 'muse_logits_vocab',
        'names': {
            'vocab_mask': 'vocab_mask_xe35',
            'muse_vocab_parallel_embedding': 'vocab_parallel_embedding_xe35',
            'softcap_logits': 'softcap_logits_xe35',
        },
        'files': {
            'src/sycl/SoftcapLogits.cpp': 'src/sycl/xe35/SoftcapLogits.cpp',
            'src/sycl/VocabMask.cpp': 'src/sycl/xe35/VocabMask.cpp',
            'tests/test_softcap_logits.py': 'tests/test_softcap_logits.py',
            'tests/test_vocab_mask.py': 'tests/test_vocab_mask.py',
            'benchmark/bench_logits_vocab_sweep.py': 'benchmark/bench_logits_vocab_sweep.py',
        },
    },
}


def git_text(*arguments):
    return subprocess.check_output(['git', '-C', str(SOURCE), *arguments], text=True)


def renamed(text, names):
    pattern = r'\b(' + '|'.join(map(re.escape, names)) + r')\b'
    return re.sub(pattern, lambda match: names[match.group()], text)


def additions(branch, filename):
    patch = git_text('diff', '--unified=0', BASE, branch, '--', filename)
    return '\n'.join(
        line[1:]
        for line in patch.splitlines()
        if line.startswith('+') and not line.startswith('+++')
    ) + '\n'


def update_patch(filename, original, replacement):
    return (
        f'*** Update File: {filename}\n@@\n'
        + ''.join('-' + line + '\n' for line in original.splitlines())
        + ''.join('+' + line + '\n' for line in replacement.splitlines())
    )


def main():
    for component, spec in SPECS.items():
        target = ROOT / 'pr' / f'muse-{component}-xe35'
        patch_parts = ['*** Begin Patch\n']
        for source_file, target_file in spec['files'].items():
            content = renamed(git_text('show', spec['branch'] + ':' + source_file), spec['names'])
            if source_file.endswith('.cpp'):
                header_name = component.replace('-', '_') + '_ops.h'
                content = content.replace(
                    '#include <torch/all.h>',
                    '#include <torch/all.h>\n\n#include "sgl_kernel/xe35/' + header_name + '"',
                )
                content = content.replace('"SYCLHelpers.h"', '"../SYCLHelpers.h"')
                content = content.replace('"MemoryAccess.h"', '"../MemoryAccess.h"')
            if source_file.endswith('bench_memory_compare.py'):
                content = content.replace('import importlib.util\n', 'import importlib\n')
                start = content.index('    source = memory_setup.memory_source\n')
                end = content.index('    from sglang.kernels.ops.memory.allocator import (\n', start)
                content = content[:start] + content[end:]
                content = content.replace(
                    '    parser.add_argument(\n        "--memory-source", type=Path, help="SGLang memory kernel source directory"\n    )\n',
                    '',
                )
                content = content.replace('str(args.memory_source or "installed sglang")', '"installed sglang"')
            target_path = target / target_file
            if target_path.exists():
                if target_path.read_text() != content:
                    raise RuntimeError(f'will not overwrite existing port: {target_path}')
            else:
                patch_parts.append(f'*** Add File: {target_path}\n')
                patch_parts.append(''.join('+' + line + '\n' for line in content.splitlines()))
        header = target / 'include/sgl_kernel_ops.h'
        original = header.read_text()
        declarations = renamed(additions(spec['branch'], 'include/sgl_kernel_ops.h'), spec['names'])
        declarations = declarations.replace('void ', 'SGL_KERNEL_EXPORT void ')
        declarations = declarations.replace('std::tuple<', 'SGL_KERNEL_EXPORT std::tuple<')
        name = component.replace('-', '_') + '_ops.h'
        architecture_header = target / 'include/sgl_kernel/xe35' / name
        architecture_content = '#pragma once\n\n#include <torch/all.h>\n\n#include "sgl_kernel_export.h"\n\nnamespace at::native::xpu {\n' + declarations + '\n}\n'
        patch_parts.append(f'*** Add File: {architecture_header}\n')
        patch_parts.append(''.join('+' + line + '\n' for line in architecture_content.splitlines()))
        anchor = '#include <optional>\n'
        if original.count(anchor) != 1:
            raise RuntimeError('unexpected header insertion point')
        replacement = original.replace(anchor, '#if SYCL_INTEL_TARGET == 35\n#include "sgl_kernel/xe35/' + name + '"\n#endif\n\n' + anchor)
        patch_parts.append(update_patch(header, original, replacement))
        registration = target / 'src/torch_extension_sycl.cc'
        original = registration.read_text()
        definitions = renamed(additions(spec['branch'], 'src/torch_extension_sycl.cc'), spec['names'])
        anchor = '  m.def("topk_softmax('
        if original.count(anchor) != 1:
            raise RuntimeError('unexpected registration insertion point')
        replacement = original.replace(anchor, '#if SYCL_INTEL_TARGET == 35\n' + definitions + '#endif\n\n' + anchor)
        patch_parts.append(update_patch(registration, original, replacement))
        patch_parts.append('*** End Patch\n')
        subprocess.run(['apply_patch'], input=''.join(patch_parts), text=True, check=True)
        print(component, git_text('rev-parse', spec['branch']).strip(), target, flush=True)


if __name__ == '__main__':
    main()
