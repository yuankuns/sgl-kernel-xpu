import ast
import re
import subprocess
from pathlib import Path


def git_text(root, ref, name):
    return subprocess.check_output(["git", "-C", str(root), "show", f"{ref}:{name}"], text=True)


def kernel_tokens(text):
    text = re.sub(r"/\*.*?\*/|//[^\n]*", "", text, flags=re.S)
    text = re.sub(r"^\s*#include[^\n]*", "", text, flags=re.M)
    text = text.replace("_xe35", "")
    return re.findall(r"\w+|[^\w\s]", text)


def main():
    target = Path("/home/syk/worktrees/sgl-kernel-xpu-up/pr/muse-residual-norm-xe35")
    source = Path("/home/syk/worktrees/sgl-kernel-xpu/muse-spark-residual-norm-main-20261014")
    source_kernel = git_text(source, "ec29f7a", "src/sycl/FusedNormResidual.cpp")
    port_kernel = (target / "src/sycl/xe35/FusedNormResidual.cpp").read_text()
    assert kernel_tokens(source_kernel) == kernel_tokens(port_kernel)
    assert '#include "sgl_kernel/xe35/residual_norm_ops.h"' in port_kernel
    assert '#include "../MemoryAccess.h"' in port_kernel
    assert '#include "../SYCLHelpers.h"' in port_kernel
    print("PASS: exact source kernel tokens modulo architecture identifiers/includes/comments.")
    original = ast.parse(git_text(target, "833c952", "tests/test_norm.py"))
    current = ast.parse((target / "tests/test_norm.py").read_text())
    original_functions = {node.name: ast.dump(node) for node in original.body if isinstance(node, ast.FunctionDef)}
    current_functions = {node.name: ast.dump(node) for node in current.body if isinstance(node, ast.FunctionDef)}
    assert all(current_functions[name] == body for name, body in original_functions.items())
    print(f"PASS: {len(original_functions)} old norm functions/cases preserved.")
    special_tests = [node for node in current.body if isinstance(node, ast.FunctionDef)
                     and node.name.startswith("test_") and node.name.endswith("_xe35")]
    for node in special_tests:
        assert any("is_residual_norm_available_xe35" in ast.unparse(decorator)
                   for decorator in node.decorator_list), node.name
    guard = next(node for node in current.body if isinstance(node, ast.FunctionDef)
                 and node.name == "is_residual_norm_available_xe35")
    assert isinstance(guard.body[0], ast.If)
    assert ast.unparse(guard.body[0].test) == "not torch.xpu.is_available()"
    assert ast.unparse(guard.body[0].body[0]) == "return False"
    assert "is_xe3_arch() and hasattr" in ast.unparse(guard.body[-1])
    benchmark = ast.parse((target / "benchmark/bench_fused_norm_residual.py").read_text())
    entry = next(node for node in benchmark.body if isinstance(node, ast.FunctionDef) and node.name == "main_xe35")
    availability = next(index for index, node in enumerate(entry.body) if isinstance(node, ast.If)
                        and ast.unparse(node.test) == "not torch.xpu.is_available()")
    architecture = next(index for index, node in enumerate(entry.body) if isinstance(node, ast.If)
                        and ast.unparse(node.test) == "not is_xe3_arch()")
    assert availability < architecture
    utils = ast.parse((target / "python/sgl_kernel/utils.py").read_text())
    arch_helper = next(node for node in utils.body if isinstance(node, ast.FunctionDef) and node.name == "is_xe3_arch")
    assert ast.unparse(arch_helper.body[-1]) == "return major == 3 and minor == 5"
    print(f"PASS: {len(special_tests)} dedicated cases + benchmark enforce XPU availability short-circuit and exact 3.5; no whole-file skip.")
    for name in ("src/sycl/RMSNorm.cpp", "src/sycl/MemoryAccess.h", "src/sycl/SYCLHelpers.h",
                 "python/sgl_kernel/elementwise.py", "src/CMakeLists.txt", "CMakeLists.txt"):
        assert (target / name).read_text() == git_text(target, "833c952", name), name
    print("PASS: upstream generic RMS/API/helpers and CMake unchanged.")
    header = (target / "include/sgl_kernel/xe35/residual_norm_ops.h").read_text()
    assert '#include "sgl_kernel_export.h"' in header
    assert "SGL_KERNEL_EXPORT void fused_post_norm_residual_pre_norm_xe35(" in header
    assert "#if SYCL_INTEL_TARGET == 35" in header
    generic = (target / "include/sgl_kernel_ops.h").read_text()
    assert '#if SYCL_INTEL_TARGET == 35\n#include "sgl_kernel/xe35/residual_norm_ops.h"\n#endif' in generic
    print("PASS: dedicated public ABI exported, cpp includes dedicated header, guard35 declarations.")
    names = subprocess.check_output(["git", "-C", str(target), "ls-files", "--others", "--exclude-standard"], text=True).splitlines()
    names += subprocess.check_output(["git", "-C", str(target), "diff", "--name-only"], text=True).splitlines()
    for name in names:
        if name.endswith(".py"):
            assert not Path(name).stem.endswith("_xe35"), name
        if name.startswith("src/sycl/xe35/") and name.endswith((".cpp", ".h")):
            assert not Path(name).stem.endswith("_xe35"), name
    print("PASS: Python and xe35-directory native filenames have no redundant xe35 suffix.")


if __name__ == "__main__":
    main()
