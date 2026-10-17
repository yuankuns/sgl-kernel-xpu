import ast
import hashlib
import json
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "normal-precision-pytest-v1"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    result = {}
    identities = {}
    for role, count in (("SOURCE", 69), ("AFTER", 73)):
        gate = json.loads((OUTPUT / f"CPU-{role}.json").read_text())
        path = Path(gate["test_file"].replace("/root/worktrees/", "/home/syk/worktrees/", 1))
        assert digest(path) == gate["test_sha256"]
        assert gate["count"] == count and not gate["GPU_initialized"]
        assert (OUTPUT / f"CPU-{role}.exit").read_text().strip() == "0"
        tree = ast.parse(path.read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name.replace("_xe35", "") == "test_fused_norm_residual")
        assertions = []
        for node in ast.walk(function):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "assert_close":
                values = {keyword.arg: ast.literal_eval(keyword.value) for keyword in node.keywords
                          if keyword.arg in ("rtol", "atol")}
                assertions.append(values)
        assert len(assertions) == 5
        assert assertions.count(dict(rtol=5e-3, atol=5e-3)) == 1
        assert assertions.count(dict(rtol=1e-5, atol=1e-5)) == 1
        assert assertions.count(dict(rtol=0, atol=0)) == 3
        identities[role] = [node.split("::")[-1].replace("_xe35", "") for node in gate["nodeids"]]
        result[role] = dict(test_file=str(path), SHA256=digest(path), actual_collected=count,
            nodeids=gate["nodeids"], primary_test_assertions=assertions,
            precision_original_and_unmodified=True, all_elements_chunked_512_not_sampling=True,
            kernel_performance_function_not_called=True, collection_not_GPU_precision=True)
    assert set(identities["SOURCE"]) <= set(identities["AFTER"])
    additions = sorted(set(identities["AFTER"]) - set(identities["SOURCE"]))
    assert len(additions) == 4
    result.update(epoch=time.time(), mechanical_suffix_all69_preserved=True, AFTER_four_original_additions=additions,
        B8_only=True, no_source_OR_PR_modification=True, new_GPU_or_compiler_started=False)
    (ROOT / "RESIDUAL_PRECISION_SCOPE_CPU_AUDIT_v1.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(actual_scope_audit_exit=0, SOURCE=69, AFTER=73, norm=5e-3,
        stream=1e-5, widen_exact=True, mechanical_suffix_preserved=True)), flush=True)


if __name__ == "__main__":
    main()
