import re
import subprocess
from pathlib import Path


ROOT = Path('/home/syk/worktrees/sgl-kernel-xpu-up/pr')


def move_patch(source, target, original, replacement):
    return (
        f'*** Update File: {source}\n*** Move to: {target}\n@@\n'
        + ''.join('-' + line + '\n' for line in original.splitlines())
        + ''.join('+' + line + '\n' for line in replacement.splitlines())
    )


def update_patch(filename, original, replacement):
    return (
        f'*** Update File: {filename}\n@@\n'
        + ''.join('-' + line + '\n' for line in original.splitlines())
        + ''.join('+' + line + '\n' for line in replacement.splitlines())
    )


def main():
    for component in ('memory', 'position', 'logits-vocab'):
        worktree = ROOT / f'muse-{component}-xe35'
        parts = ['*** Begin Patch\n']
        files = list((worktree / 'tests').glob('*_xe35.py'))
        files += list((worktree / 'benchmark').glob('*_xe35.py'))
        files += list((worktree / 'src/sycl/xe35').glob('*_xe35.cpp'))
        for filename in files:
            target = filename.with_name(filename.name.replace('_xe35.', '.'))
            if target.exists():
                raise RuntimeError(f'target already exists: {target}')
            original = filename.read_text()
            replacement = original.replace('_xe35.py', '.py')
            replacement = replacement.replace('"SYCLHelpers.h"', '"../SYCLHelpers.h"')
            replacement = replacement.replace('"MemoryAccess.h"', '"../MemoryAccess.h"')
            parts.append(move_patch(filename, target, original, replacement))
        generic = worktree / 'include/sgl_kernel_ops.h'
        original = generic.read_text()
        match = re.search(
            r'#if SYCL_INTEL_TARGET == 35\n(.*?)#endif\n(?=std::tuple<torch::Tensor, torch::Tensor> fused_scale_residual_norm_scale_shift)',
            original,
            re.DOTALL,
        )
        if not match:
            raise RuntimeError('missing isolated declaration block')
        name = component.replace('-', '_') + '_ops.h'
        header = worktree / 'include/sgl_kernel/xe35' / name
        declarations = match.group(1)
        content = '#pragma once\n\n#include <torch/all.h>\n\nnamespace at::native::xpu {\n' + declarations + '\n}\n'
        parts.append(f'*** Add File: {header}\n')
        parts.append(''.join('+' + line + '\n' for line in content.splitlines()))
        replacement = original[:match.start()] + original[match.end():]
        include = f'#if SYCL_INTEL_TARGET == 35\n#include "sgl_kernel/xe35/{name}"\n#endif\n\n'
        replacement = replacement.replace('#include <optional>\n', include + '#include <optional>\n', 1)
        parts.append(update_patch(generic, original, replacement))
        parts.append('*** End Patch\n')
        subprocess.run(['apply_patch'], input=''.join(parts), text=True, check=True)


if __name__ == '__main__':
    main()
