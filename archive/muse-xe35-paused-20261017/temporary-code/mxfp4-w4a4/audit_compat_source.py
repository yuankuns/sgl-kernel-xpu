import ast
import hashlib
import json
import pathlib
import re
import subprocess


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def asm_strings(text):
    blocks = re.findall(r"asm\s*\((.*?)\s*:\s*\"=rw\"", text, re.DOTALL)
    return ["".join(ast.literal_eval(literal) for literal in re.findall(r'"(?:\\.|[^"\\])*"', block)) for block in blocks]


def main():
    artifact = pathlib.Path(__file__).resolve().parent
    target = pathlib.Path('/home/syk/worktrees/sgl-kernel-xpu-up/pr/muse-mxfp4-w4a4-xe35')
    original_pin = pathlib.Path('/home/syk/worktrees/cutlass_internal/muse-w4a4-e87870b')
    public_pin = artifact.parent / 'deps/sycl-tla-7b56'
    helper = target / 'src/sycl/xe35/moe/collective/moe_bf16_reorder.hpp'
    epilogue = target / 'src/sycl/xe35/moe/collective/moe_array_epilogue.hpp'
    original_arch = original_pin / 'include/cute/arch/reorder_xe.hpp'
    original_algorithm = original_pin / 'include/cute/algorithm/reorder.hpp'
    original_bf16 = original_arch.read_text().split('struct Xe_Reorder<ReorderKind::UU, float, bfloat16_t>', 1)[1]
    assert asm_strings(original_bf16)[:2] == asm_strings(helper.read_text())
    assert len(asm_strings(helper.read_text())) == 2
    assert epilogue.read_text().count('::sgl_kernel::vllm_grouped_gemm_xe35::reorder_moe_output_xe35(') == 2
    assert '#if defined(SYCL_INTEL_TARGET) && SYCL_INTEL_TARGET == 35' in helper.read_text()
    assert 'cute::reorder(source, destination);' in helper.read_text()
    expected_pins = [(original_pin, 'e87870bb7aacf83bfb3be9da685f148316b8df23'), (public_pin, '7b56b936b87f799f80dde60c2fd8d11c1128cebb')]
    for pin, expected in expected_pins:
        assert subprocess.check_output(['git', '-C', str(pin), 'rev-parse', 'HEAD'], text=True).strip() == expected
        assert not subprocess.check_output(['git', '-C', str(pin), 'status', '--porcelain'], text=True).strip()
    base = subprocess.check_output(['git', '-C', str(target), 'rev-parse', 'HEAD'], text=True).strip()
    unchanged = ['src/sycl/TripleOps.cpp', 'src/sycl/SYCLHelpers.h', 'src/sycl/MemoryAccess.h', 'src/sycl/Utils.h', 'src/sycl/per_token_group_quant_fp4.cpp', 'src/sycl/TopKSigMoid.cpp']
    for name in unchanged:
        assert (target / name).read_bytes() == subprocess.check_output(['git', '-C', str(target), 'show', f'{base}:{name}'])
    report = {'kind': 'CPU static source-preserving dependency adaptation audit, not GPU or full-build acceptance', 'original_bf16_stage_and_pack_asm_byte_identical': True, 'output_epilogue_call_sites': 2, 'original_pin': expected_pins[0][1], 'public_pin': expected_pins[1][1], 'unchanged_shared_BMG_paths': unchanged, 'hashes': {str(path): sha256(path) for path in (helper, epilogue, original_arch, original_algorithm, public_pin / 'include/cute/arch/reorder_xe.hpp', public_pin / 'include/cute/algorithm/reorder.hpp')}}
    (artifact / 'compat-source-preservation.json').write_text(json.dumps(report, indent=2) + '\n')
    changed = subprocess.check_output(['git', '-C', str(target), 'diff', '--name-only'], text=True).splitlines()
    untracked = subprocess.check_output(['git', '-C', str(target), 'ls-files', '--others', '--exclude-standard'], text=True).splitlines()
    files = sorted(set(changed + untracked))
    assert len(files) == 25, files
    manifest = json.loads((artifact / 'port-source-provenance.json').read_text())
    manifest['source_hashes'] = {name: sha256(target / name) for name in files}
    manifest['dependency_adaptation_audit'] = 'compat-source-preservation.json'
    (artifact / 'port-source-provenance.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (artifact / 'changed-files.txt').write_text('\n'.join(files) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
