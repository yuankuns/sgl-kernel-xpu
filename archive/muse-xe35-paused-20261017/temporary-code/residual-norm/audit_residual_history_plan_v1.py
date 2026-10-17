import ast
import copy
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parent
SOURCE = Path("/home/syk/worktrees/sgl-kernel-xpu/muse-spark-residual-norm-main-20261014")
PORT = ROOT.parent.parent / "pr/muse-residual-norm-xe35"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def matrix_recipe(path, function):
    tree = ast.parse(path.read_text())
    body = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function)
    loop = next(node for node in body.body if isinstance(node, ast.For) and isinstance(node.iter, ast.Call)
                and isinstance(node.iter.func, ast.Attribute) and node.iter.func.attr == "product")
    recipe = loop.body[:7]
    normalized = ast.dump(ast.Module(body=recipe, type_ignores=[]), include_attributes=False)
    assert ast.unparse(recipe[0]) == "torch.manual_seed(42)"
    assert [node.targets[0].id for node in recipe[1:]] == ["branch", "stream", "post", "pre", "alpha", "beta"]
    sequence = [ast.literal_eval(node) for node in loop.iter.args]
    return dict(path=str(path), file_sha256=digest(path), function=function,
                recipe_AST_sha256=hashlib.sha256(normalized.encode()).hexdigest(),
                recipe_source=ast.unparse(ast.Module(body=recipe, type_ignores=[])),
                matrix_axes=sequence, per_case_seed42=True, RNG_device="xpu",
                cleanup_source=ast.unparse(ast.Module(body=loop.body[-6:], type_ignores=[]))), normalized


class Normalize(ast.NodeTransformer):
    def visit_Name(self, node):
        node.id = node.id.removesuffix("_xe35")
        return node

    def visit_Attribute(self, node):
        self.generic_visit(node)
        node.attr = node.attr.removesuffix("_xe35")
        return node

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        node.name = node.name.removesuffix("_xe35")
        return node


def function_AST(path, name):
    tree = ast.parse(path.read_text())
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    return ast.dump(Normalize().visit(copy.deepcopy(node)), include_attributes=False)


def tests(path, suffix):
    tree = ast.parse(path.read_text())
    results = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
            continue
        if suffix and not node.name.endswith(suffix):
            continue
        if not suffix and not ("fused_norm_residual" in node.name or "round_bfloat16_reference" in node.name):
            continue
        axes = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute) and decorator.func.attr == "parametrize":
                axes.append(dict(name=ast.literal_eval(decorator.args[0]), values=ast.literal_eval(decorator.args[1])))
        results.append(dict(name=node.name, line=node.lineno, axes=axes,
                            static_case_count=math.prod(len(axis["values"]) for axis in axes)))
    return dict(path=str(path), sha256=digest(path), cases=results,
                static_total=sum(record["static_case_count"] for record in results),
                actual_pytest_collection_run=False)


def main():
    source_recipe, source_AST = matrix_recipe(ROOT / "source_compiler_accuracy.py", "run")
    after_recipe, after_AST = matrix_recipe(ROOT / "after_pre_wave_accuracy_v4.py", "accuracy")
    assert source_AST == after_AST
    assert source_recipe["matrix_axes"] == after_recipe["matrix_axes"] == [[8, 32768], ["none", "weight", "input_scale"], [False, True], [False, True]]
    rows = []
    total_input = total_oracle = total_actual = total_finite = total_difference = 0
    for index, (count, mode, rounded, wide) in enumerate(itertools.product(*source_recipe["matrix_axes"]), 1):
        elements = count * 8192
        weights = 8192 * (10 + (2 if mode != "none" else 0))
        input_bytes = elements * 6 + weights
        oracle_bytes = elements * 10
        actual_two_roles_bytes = elements * (12 + (8 if wide else 0))
        input_elements = 2 * elements + 8192 * (3 + (1 if mode != "none" else 0))
        finite_bytes = 2 * (input_elements + 3 * elements + 2 * (2 + int(wide)) * elements)
        difference_bytes = elements * (11 if wide else 6)
        rows.append(dict(index=index, case=f"{count}_{mode}_{int(rounded)}_{int(wide)}", rows=count,
                         hidden=8192, post_mode=mode, round_pre=rounded, widen=wide,
                         input_bytes=input_bytes, oracle_bytes=oracle_bytes,
                         actual_two_roles_bytes=actual_two_roles_bytes,
                         full_finite_masks_bytes=finite_bytes, full_difference_masks_bytes=difference_bytes))
        total_input += input_bytes
        total_oracle += oracle_bytes
        total_actual += actual_two_roles_bytes
        total_finite += finite_bytes
        total_difference += difference_bytes
    source_tests = SOURCE / "tests/test_fused_norm_residual.py"
    after_tests = PORT / "tests/test_norm.py"
    pure_reference = {name: function_AST(source_tests, name) == function_AST(after_tests, name + "_xe35")
                      for name in ("round_bfloat16", "reference")}
    assert all(pure_reference.values())
    source_scope = tests(source_tests, "")
    after_scope = tests(after_tests, "_xe35")
    assert source_scope["static_total"] == 69 and after_scope["static_total"] == 73
    memory = {line.split(":")[0]: int(line.split()[1]) * 1024
              for line in Path("/proc/meminfo").read_text().splitlines()
              if line.split(":")[0] in ("MemTotal", "MemAvailable", "MemFree")}
    disk = os.statvfs(ROOT)
    historical = json.loads((ROOT / "after-pre-wave-accuracy-v4/AFTER-cases.json").read_text())
    assert len(historical) == 17
    budget = dict(input_bytes=total_input, oracle_bytes=total_oracle,
                  two_role_actual_bytes=total_actual, finite_masks_bytes=total_finite,
                  difference_masks_bytes=total_difference,
                  total_tensor_payload_bytes=total_input + total_oracle + total_actual + total_finite + total_difference,
                  artifact_cap_bytes=256 * 2**30, filesystem_headroom_bytes=100 * 2**30,
                  worker_RSS_cap_bytes=16 * 2**30, descendant_tree_RSS_cap_bytes=24 * 2**30,
                  separate_original_SOURCE_pytest_RSS_cap_bytes=64 * 2**30,
                  separate_pytest_tree_RSS_cap_bytes=80 * 2**30)
    result = dict(scope="CPU AST/resource plan only; no Torch import, fixture generation, GPU, pytest or compilation",
                  epoch=time.time(), SOURCE_actual_matrix_recipe=source_recipe,
                  AFTER_actual_matrix_recipe=after_recipe, actual_matrix_recipe_AST_exact_equal=True,
                  original_SOURCE_benchmark=dict(path=str(SOURCE / "benchmark/bench_fused_norm_residual.py"),
                      sha256=digest(SOURCE / "benchmark/bench_fused_norm_residual.py"),
                      seed42_before_rows_loop=True, differs_from_actual_matrix_per_case_seed=True),
                  case_order=rows, failure_case_ordinal=18, reference_math_AST_equal_after_suffix_normalization=pure_reference,
                  SOURCE_precision_tests=source_scope, AFTER_precision_tests=after_scope,
                  historical_saved_cases=17, historical_original_inputs_saved=False,
                  historical_case_manifest_sha256=digest(ROOT / "after-pre-wave-accuracy-v4/AFTER-cases.json"),
                  historical_manifest_has_full_output_tensor_SHA=True,
                  allocator_history_exact_replay_possible=False, budget=budget,
                  disk=dict(total_bytes=disk.f_blocks * disk.f_frsize, available_bytes=disk.f_bavail * disk.f_frsize),
                  memory=memory, new_GPU_grant_created=False, GPU_started=False)
    (ROOT / "RESIDUAL_HISTORY24_CPU_AUDIT_v1.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(CPU_AST_audit_actual0=True, actual_matrix_RNG_AST_equal=True,
          reference_AST_equal=pure_reference, precision_test_counts=[source_scope["static_total"], after_scope["static_total"]],
          tensor_payload_GiB=budget["total_tensor_payload_bytes"] / 2**30,
          disk_available_GiB=result["disk"]["available_bytes"] / 2**30,
          memory_available_GiB=memory["MemAvailable"] / 2**30)), flush=True)


if __name__ == "__main__":
    main()
