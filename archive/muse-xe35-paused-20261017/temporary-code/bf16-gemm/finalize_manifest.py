import hashlib
import json
import os
import subprocess
from pathlib import Path


def digest_xe35(path):
    checksum = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def main_xe35():
    root = Path(__file__).resolve().parent
    module = root.name
    version = os.environ.get('CANONICAL_VERSION', 'canonical-v1')
    if Path(version).name != version:
        raise RuntimeError('Canonical version must be a directory basename')
    tree = Path('/home/syk/worktrees/sgl-kernel-xpu-up/pr') / ('muse-bf16-gemm-xe35' if module == 'bf16-gemm' else 'muse-moe-permutation-xe35')
    source_manifest = json.loads((root / 'port-source-manifest.json').read_text())
    source_manifest['files'] = {name: digest_xe35(tree / name) for name in source_manifest['files']}
    (root / 'port-source-manifest.json').write_text(json.dumps(source_manifest, indent=2) + '\n')
    exits = {path.name: int(path.read_text().strip()) for path in root.glob('*.exit')}
    summaries = {path.name: json.loads(path.read_text()) for path in root.glob('*-summary.json') if 'BAAB' in path.name}
    diagnostic_summaries = {name: value for name, value in summaries.items() if name.startswith('kernel-diagnostic')}
    official_summaries = {name: value for name, value in summaries.items() if name.startswith(version + '-') and value.get('verification_kind') == 'official integration'}
    performance_summaries = {name: value for name, value in official_summaries.items() if not value.get('accuracy_only', False)}
    accepted_performance = bool(performance_summaries) and all(value.get('performance_accepted') and value.get('gpu') == 4 and not value['needs_retest'] for value in performance_summaries.values())
    correctness = bool(official_summaries) and all(value['accuracy_passed'] for value in official_summaries.values())
    canonical_stage = root / version
    canonical_exit_path = canonical_stage / 'default-all-build.exit'
    full_build = canonical_exit_path.exists() and canonical_exit_path.read_text().strip() == '0' and (canonical_stage / 'source-binary-manifest.json').exists()
    pytest_pass = exits.get(version + '-BAAB-pytest.exit') == 0
    compiler_policy_hold = (root / 'icpx-migration-hold.json').exists()
    decode_pass = full_build and correctness and accepted_performance and pytest_pass and not compiler_policy_hold
    manifest = {'module': module, 'tree': str(tree), 'upstream_base': source_manifest['upstream_base'],
                'source_ref': source_manifest['source_ref'], 'gpu': 4 if module == 'bf16-gemm' else 10,
                'scope': {'batch': 8, 'isl': 4096, 'osl': 16, 'tp': 1, 'ep': 1},
                'full_default_build_passed': full_build, 'accuracy_passed': correctness,
                'full_default_build_status': 'passed' if full_build else 'canonical_pending_or_failed',
                'canonical_stage': str(canonical_stage),
                'original_full_build_actual_exit': exits.get('full-build.exit'),
                'performance_gpu': 4, 'accuracy_gpu': 10,
                'historical_gpu10_performance_non_acceptance': module == 'moe-permutation',
                'performance_passed': accepted_performance, 'pytest_passed': pytest_pass,
                'final_acceptance_eligible': not compiler_policy_hold,
                'compiler_policy_hold': compiler_policy_hold,
                'compiler_policy_status': 'Awaiting coordinator all-icpx package and compiler verification' if compiler_policy_hold else 'No migration hold',
                'kernel_diagnostic_accuracy_passed': bool(diagnostic_summaries) and all(value['accuracy_passed'] for value in diagnostic_summaries.values()),
                'kernel_diagnostic_performance_within_5pct': module == 'bf16-gemm' and bool(diagnostic_summaries) and all(not value['needs_retest'] for value in diagnostic_summaries.values()),
                'diagnostic_is_not_official_acceptance': True,
                'decode_acceptance_passed': decode_pass, 'module_acceptance_passed': decode_pass and module == 'bf16-gemm',
                'prefill_limitation': 'Source small-permutation pairs<=256; B8 prefill unsupported; no prefill device-time acceptance' if module == 'moe-permutation' else None,
                'exit_statuses': exits, 'performance_summaries': summaries,
                'source_provenance_sha256': digest_xe35(root / 'source-provenance.json'),
                'port_source_manifest_sha256': digest_xe35(root / 'port-source-manifest.json'),
                'build_settings': json.loads(sorted(root.glob('build-attempt-*-settings.json'))[-1].read_text()),
                'build_history': [json.loads(path.read_text()) for path in sorted(root.glob('build-attempt-*-settings.json'))],
                'modified_paths': subprocess.check_output(['git', '-C', str(tree), 'diff', '--name-only']).decode().splitlines()
                                  + subprocess.check_output(['git', '-C', str(tree), 'ls-files', '--others', '--exclude-standard']).decode().splitlines(),
                'committed': False, 'pushed': False, 'pr_created': False,
                'artifact_hashes': {path.name: digest_xe35(path) for path in root.iterdir() if path.is_file() and path.suffix in ['.log', '.command', '.csv', '.json', '.exit'] and path.name != 'manifest.json'}}
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    status_path = root / 'worker-status.json'
    if status_path.exists():
        worker_status = json.loads(status_path.read_text())
        worker_status['old_full_build_exit'] = exits.get('full-build.exit')
        worker_status['full_build_exit'] = int(canonical_exit_path.read_text()) if canonical_exit_path.exists() else None
        worker_status['canonical_stage'] = str(canonical_stage)
        runtime_ended = 'BAAB-retest.exit' in exits or ('BAAB.exit' in exits and exits.get('BAAB-analysis.exit') != 3)
        if canonical_stage.exists() and not canonical_exit_path.exists():
            worker_status['validation_state'] = 'canonical_build_pending_old_waiters_cancelled'
            worker_status['gpu_acceptance_processes_ended'] = True
        elif not full_build and 'full-build.exit' in exits:
            worker_status['validation_state'] = 'full_build_failed_no_gpu_acceptance'
            worker_status['gpu_acceptance_processes_ended'] = True
        elif runtime_ended:
            worker_status['validation_state'] = 'runtime_ended_review_manifest'
            worker_status['gpu_acceptance_processes_ended'] = True
        elif full_build:
            worker_status['validation_state'] = 'runtime_pending_or_running'
        else:
            worker_status['validation_state'] = 'waiting_for_full_default_build'
        status_path.write_text(json.dumps(worker_status, indent=2) + '\n')
    draft_path = root / 'PR_DRAFT.md'
    old = draft_path.read_text()
    content = old.split('\n## 本次验收状态\n')[0]
    if '\n## 命名修正与安全重启\n' in old and '\n## 命名修正与安全重启\n' not in content:
        content += '\n## 命名修正与安全重启\n' + old.split('\n## 命名修正与安全重启\n', 1)[1].split('\n## 本次验收状态\n')[0]
    content = content.replace('（草稿，尚未验收）', '（中文草稿，验收状态见末节）')
    content = content.replace('（待新 port 验证）', '（本次结果见末节）')
    content = content.replace('默认完整构建正在执行；尚无 port 验收通过结论。', '默认完整构建与测试的实际 exit 及结论见末节及 manifest.json。')
    content = content.replace('Port 默认完整构建执行中；decode port Accuracy/性能、default full build 均尚未宣布通过。', 'Port 完整构建及 decode Accuracy/性能的实际状态见末节及 manifest.json。')
    content = content.replace('- 尚未启动 diagnostic BAAB，等待 main 的 GPU4–11 八卡 barrier；默认 full build 保持原进程，未重复启动。',
                              '- Diagnostic BAAB 已按 main GPU4–11 八卡 barrier 启动；实际 exit、结果与限制见原始日志及末节。默认 full build 保持原进程，未重复启动。')
    status = 'decode 通过' if decode_pass else '未通过完整 decode 验收（失败或待完成）'
    content += f'\n## 本次验收状态\n\n- 下列验收 gate 仅指 official integration，提前 kernel diagnostic 不代替完整 package。\n- 状态：{status}。\n- 默认 full build：{full_build}；source-vs-port accuracy：{correctness}；性能 <=5% gate：{accepted_performance}；pytest：{pytest_pass}。\n'
    if module == 'moe-permutation':
        content += '- Prefill 仍不支持，因此整个 decode+prefill 验收不完整，不能声明整体通过。\n'
    content += '- 构建、测试及 bench 原始 exit、日志、所有 samples、库/源码哈希见 `manifest.json` 与关联原始文件。\n'
    for summary_name, summary in summaries.items():
        kind = ('早期GPU10卡 timings non-acceptance；仅历史诊断' if module == 'moe-permutation' else '提前 kernel diagnostic，非 official integration') if summary_name.startswith('kernel-diagnostic') else 'official integration'
        content += f'\n### {summary_name}（{kind}）\n\n| case | path | source ms | port ms | 变化 | source 漂移 | accuracy |\n|---|---|---:|---:|---:|---:|---|\n'
        for row in summary['rows']:
            content += f'| {row["case"]} | {row["path"]} | {row["source_ms"]:.6f} | {row["port_ms"]:.6f} | {row["change_pct"]:+.2f}% | {row["source_drift_pct"]:+.2f}% | {row["accuracy"]} |\n'
    if any(value['needs_retest'] for value in summaries.values()):
        content += '\n- 已发现 >5% 变化/漂移，必须保留复测与差异分析；仅一次结果、或仍超阈值的复测结果不构成性能通过。\n'
    patch = '*** Begin Patch\n*** Update File: ' + str(draft_path) + '\n@@\n'
    patch += ''.join('-' + line + '\n' for line in old.splitlines())
    patch += ''.join('+' + line + '\n' for line in content.splitlines()) + '*** End Patch\n'
    subprocess.run(['apply_patch', patch], check=True)
    print(json.dumps({key: manifest[key] for key in ['module', 'full_default_build_passed', 'accuracy_passed', 'performance_passed', 'decode_acceptance_passed', 'module_acceptance_passed']}, indent=2))


if __name__ == '__main__':
    main_xe35()
