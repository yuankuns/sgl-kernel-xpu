import ast
import hashlib
import json
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    previous = json.loads((ROOT / "RESIDUAL_MODEL_CONFIG_PROVENANCE_v1.json").read_text())
    files = previous["authoritative_files"]
    for record in files.values():
        assert digest(Path(record["path"])) == record["sha256"]
    adapter_path = Path(files["model_adapter"]["path"])
    adapter = ast.parse(adapter_path.read_text())
    binding = next(node for node in ast.walk(adapter) if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "_SYCL_FUSED_NORM_RESIDUAL_OP" for target in node.targets)
                   and isinstance(node.value, ast.Attribute))
    assert ast.unparse(binding.value) == "torch.ops.sgl_kernel.fused_post_norm_residual_pre_norm"
    layer_path = Path(files["layer_math"]["path"])
    layer = ast.parse(layer_path.read_text())
    functions = {}
    for tree, path, names in ((adapter, adapter_path, ("_sycl_supported", "_ensure_cached_weights", "forward_sycl")),
                             (layer, layer_path, ("_get_effective_weight", "_coefficients", "precompute_coefficients"))):
        for name in names:
            function = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)
            functions[name] = dict(path=str(path), line=function.lineno,
                AST_sha256=hashlib.sha256(ast.dump(function, include_attributes=False).encode()).hexdigest(),
                source=ast.get_source_segment(path.read_text(), function))
    math = functions["_coefficients"]["source"]
    assert "torch.sigmoid(w)" in math and "torch.sigmoid(-w)" in math and "torch.clamp(alpha_sq, min=self.eps)" in math
    output = dict(epoch=time.time(), audit="CPU stdlib read-only SHA/AST, no Torch/framework imports or execution",
        authoritative_files=files, authoritative_files_rehashed=True,
        config_parameters=previous["config_parameters"], effective_weight_semantics=previous["effective_weight_semantics"],
        residual_gate_coefficients=previous["residual_gate_coefficients"],
        router_widen_semantics=previous["router_widen_semantics"],
        actual_model_boundary_roles=previous["actual_model_boundary_roles"], functions=functions,
        binding=dict(path=str(adapter_path), line=binding.lineno, op=ast.unparse(binding.value),
            genuine_AFTER_required_op="torch.ops.sgl_kernel.fused_post_norm_residual_pre_norm_xe35",
            binding_gap_verified=True, fixed=False),
        mathematical_bridge_proposal="Only future framework dispatcher change: short circuit XPU availability, real capability3.5, choose genuine suffixed AFTER op on CRI; preserve non-CRI old/fallback routes and all weight/gate/cache/eps/mode plumbing. No alias/fake registration.",
        future_validation_not_scheduled=dict(shapes=[8, 12288, 20480], hidden=8192,
            serving="B8/ISL4096/OSL16/TP1/EP1", boundary_roles=3,
            compare_effective_gamma_gain_cast_and_correlated_alpha_beta=True,
            real_framework_genuine_package_required=True, checkpoint_raw_provenance_required_for_claim=True),
        original69_SOURCE_and73_AFTER_pytest_not_model_adapter_tests=True,
        old_NaN_explained=False, model_path_passed=False, checkpoint_raw_values_audited=False,
        framework_PR_math_unchanged=True, new_GPU_or_compile=False)
    (ROOT / "RESIDUAL_MODEL_BRIDGE_CPU_AUDIT_v2.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(dict(actual_CPU_audit=0, all_authoritative_files_rehashed=True,
        binding_gap=True, model_path_passed=False)), flush=True)


if __name__ == "__main__":
    main()
