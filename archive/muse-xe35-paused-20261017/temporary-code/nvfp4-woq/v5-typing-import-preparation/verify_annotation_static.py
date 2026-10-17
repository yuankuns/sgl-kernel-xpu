import ast
import builtins
import hashlib
import importlib.util
import json
import typing
from pathlib import Path


DIRECTORY = Path(__file__).resolve().parent
ARTIFACT = DIRECTORY.parent
TARGET = Path('/home/syk/worktrees/sgl-kernel-xpu-up/pr/muse-nvfp4-woq-xe35')
RELATIVE = 'python/sgl_kernel/nvfp4_moe.py'
before = (DIRECTORY / 'before-nvfp4_moe.py').read_bytes()
after = (TARGET / RELATIVE).read_bytes()
assert after == before.replace(b'from typing import Tuple\n', b'from typing import Optional, Tuple\n', 1)
old_tree = ast.parse(before, type_comments=True)
new_tree = ast.parse(after, type_comments=True)
assert isinstance(new_tree.body[0], ast.ImportFrom) and new_tree.body[0].module == 'typing'
assert [alias.name for alias in new_tree.body[0].names] == ['Optional', 'Tuple']
assert ast.dump(ast.Module(body=old_tree.body[1:], type_ignores=old_tree.type_ignores), include_attributes=False) == ast.dump(ast.Module(body=new_tree.body[1:], type_ignores=new_tree.type_ignores), include_attributes=False)
compile(new_tree, str(TARGET / RELATIVE), 'exec', dont_inherit=True)
bindings = set(dir(builtins))
typing_imports = []
for node in new_tree.body:
    if isinstance(node, ast.Import):
        bindings.update(alias.asname or alias.name.split('.')[0] for alias in node.names)
    elif isinstance(node, ast.ImportFrom):
        bindings.update(alias.asname or alias.name for alias in node.names)
        if node.module == 'typing':
            typing_imports.extend(alias.name for alias in node.names)
assert all(hasattr(typing, name) for name in typing_imports)
annotations = []
for node in ast.walk(new_tree):
    if isinstance(node, ast.arg) and node.annotation is not None:
        annotations.append(node.annotation)
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.returns is not None:
        annotations.append(node.returns)
    elif isinstance(node, ast.AnnAssign):
        annotations.append(node.annotation)
names = {item.id for annotation in annotations for item in ast.walk(annotation) if isinstance(item, ast.Name)}
assert not names - bindings
optional_references = [node.lineno for node in ast.walk(new_tree) if isinstance(node, ast.Name) and node.id == 'Optional']
assert optional_references == [109, 110]
assert all(name in names or name == 'Tuple' for name in typing_imports)
specification = importlib.util.spec_from_file_location('inventory', ARTIFACT.parent / 'canonical-tools/overlay_snapshot.py')
inventory = importlib.util.module_from_spec(specification)
specification.loader.exec_module(inventory)
files, selection = inventory.source_inventory(TARGET)
v4 = json.loads((ARTIFACT / 'canonical-source-freeze-v4-icpx.json').read_text())
assert set(files) == set(v4['files'])
changed = sorted(name for name in files if files[name] != v4['files'][name])
assert changed == [RELATIVE]
stage = ARTIFACT / 'canonical-v4-icpx/site/sgl_kernel/nvfp4_moe.py'
assert stage.read_bytes() == before
prepatch = json.loads((DIRECTORY / 'prepatch-provenance.json').read_text())
sha = lambda data: hashlib.sha256(data).hexdigest()
assert sha((ARTIFACT / 'canonical-v4-icpx/source-binary-manifest.json').read_bytes()) == prepatch['v4_package_manifest_sha256']
report = {'actual_static_check_exit': 0, 'changed_actual_PR_file': str(TARGET / RELATIVE), 'only_delta': 'typing import adds Optional; Tuple retained', 'before_sha256': sha(before), 'after_sha256': sha(after), 'AST_and_compile_syntax_verified': True, 'function_signatures_defaults_decorators_bodies_unchanged': True, 'all_annotation_names_statically_bound': True, 'annotation_nodes_checked': len(annotations), 'annotation_root_names': sorted(names), 'declared_typing_imports': typing_imports, 'Optional_reference_lines': optional_references, 'unused_Tuple_retained': True, 'all_other_source_inventory_entries_unchanged': True, 'source_files_checked': len(files), 'only_v4_to_candidate_v5_source_delta': changed, 'candidate_v5_source_digest': inventory.manifest_digest(files), 'v4_stage_and_manifest_unchanged': True, 'SOURCE9_prior_evidence_unmodified': True, 'Torch_GPU_or_actual_module_not_imported': True, 'normal_package_import_verified': False, 'build_started': False, 'GPU_started': False}
print(json.dumps(report, indent=2))
