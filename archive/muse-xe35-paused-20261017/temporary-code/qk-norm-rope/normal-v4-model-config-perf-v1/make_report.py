import json
from pathlib import Path

RUN = Path(__file__).resolve().parent


def main():
    final = json.loads((RUN / 'analysis-final.json').read_text())
    audit = json.loads((RUN / 'final-evidence-audit.json').read_text())
    assert (RUN / 'final-evidence-audit.exit').read_text().strip() == '0' and audit['actual_exit'] == 0
    provenance = json.loads((RUN / 'provenance.json').read_text())
    continued = json.loads((RUN / 'actual-CONT-proof.json').read_text())
    lines = ['# Genuine normal v4 optimized SOURCE→PORT GPU6 性能', '',
        '这是移植保真/无退化比较，不是preopt→optimized的优化前后加速比。稳定GPU6独立授权，不是八卡并发。', '',
        f"supervisor68311/host3129065/container234687；CONT epoch{continued['epoch']}，同starttick8042282，无重启。全部{audit['phase_count']}个phase、{audit['measured_event_call_count']}次device-event样本；supervisor/analysis/auditor均actual0，物理GPU6锁已释放，逻辑lease不自动转W4。", '',
        'SOURCE155 / AFTER712 mapped native SHA逐phase核验。模型operator adapter→generic dispatcher真实正常package路径；不使用diagABI/shim/load_library/fake注册。', '',
        '每phase每case 21samples×32calls、warmup5、BAAB三轮；超过5%回归或漂移的case独立有界追加三轮。每call在event外恢复完整原输入，每phase先全QKV/tail对照已审SOURCE uint16 sanity。event时间包含实际adapter提交的position cast/native GPU工作和可能可见的host提交空隙，不是纯kernel latency，也不是E2E latency；不能与旧graph timing直接作速度比。', '',
        '完整strict5e-3 CPUref/全QKV/tail证据在 ../normal-v4-model-config-accuracy-v2/final-evidence-audit.json（17cases，独立auditor95498exit0）。旧T8/T32768独立代表精度在 ../normal-v4-accuracy-only-v1/。本性能阶段使用SHA完全一致输入，未更改数学、容差、kernel或编译。', '',
        '## 初始三轮：actual-config semantic synthetic', '',
        'Q128/K16/D64，18432列QKVO，tail144=V16+O_gate128。decode8 positions4096..4110；derived prefill waves12288/20480。不是raw model replay，不声明已验证所有production调用。', '',
        '| case | SOURCE µs median±MAD | PORT µs median±MAD | PORT变化 | BAAB最大漂移 | 最终分类 |',
        '|---|---:|---:|---:|---:|---|']
    for group in ('actual-config-semantic-synthetic', 'legacy-aggregate-representative'):
        if group == 'legacy-aggregate-representative':
            lines.extend(['', '## 初始三轮：旧独立代表栏', '',
                'T8 positions0..7为旧合成decode-sized代表；T32768只是B8 aggregate-prefill代表，不宣称实际单launch。', '',
                '| case | SOURCE µs median±MAD | PORT µs median±MAD | PORT变化 | BAAB最大漂移 | 最终分类 |',
                '|---|---:|---:|---:|---:|---|'])
        for name, case in final['cases'].items():
            if case['group'] != group:
                continue
            first = case['initial']
            lines.append(f"| {name} | {first['SOURCE']['median_us']:.6f}±{first['SOURCE']['MAD_us']:.6f} | {first['PORT']['median_us']:.6f}±{first['PORT']['MAD_us']:.6f} | {first['PORT_regression_percent']:+.3f}% | {first['maximum_within_BAAB_absolute_drift_percent']:.3f}% | {case['acceptance_status']} |")
    lines.extend(['', '## 有界复测（原始三轮不覆盖）', '',
        '| case | SOURCE µs median±MAD | PORT µs median±MAD | PORT变化 | BAAB最大漂移 | 首末轮漂移 SOURCE / PORT |',
        '|---|---:|---:|---:|---:|---:|'])
    for name, case in final['cases'].items():
        retry = case['bounded_retest']
        if retry:
            drift = retry['first_last_round_drift_percent']
            lines.append(f"| {name} | {retry['SOURCE']['median_us']:.6f}±{retry['SOURCE']['MAD_us']:.6f} | {retry['PORT']['median_us']:.6f}±{retry['PORT']['MAD_us']:.6f} | {retry['PORT_regression_percent']:+.3f}% | {retry['maximum_within_BAAB_absolute_drift_percent']:.3f}% | {drift['source']:+.3f}% / {drift['port']:+.3f}% |")
    if not final['bounded_retest_cases']:
        lines.append('| 无case触发复测 | — | — | — | — | — |')
    lines.extend(['', 'analysis-final.json保存所有初始/复测轮raw samples、median/MAD、两次provider漂移和首末轮漂移。analysis/audit exit0仅代表执行/证据通过；regression_confirmed或drift_requires_qualification不应写成无条件性能PASS。', '',
        f"无条件全case性能无退化：{final['all_cases_unqualified_performance_nonregression']}。", '',
        '## 独立优化前后问题', '',
        '原preopt ab3a26c的新icpx原native严格T32768仍3508违规/phaseexit1，完整证据保留 ../source-accuracy-v1/fresh-preopt/accuracy-t32768.json/.exit。该失败baseline未修数学，不作为正确性匹配加速比分母。历史preopt→optimized raw timing仅诊断，见PR_DRAFT独立表；本轮未计时preopt，不能用SOURCE→PORT冒充优化前后。', '',
        '## Provenance 与实际命令', '',
        f"- AFTER current full manifest: `{provenance['canonical_manifest_sha256']}`",
        f"- IntelLLVM compiler policy: `{provenance['compiler_policy_sha256']}`",
        f"- Frozen PR source digest: `{provenance['source_digest']}`",
        f"- SOURCE original feed315 QK native: `{provenance['source_provider']['source_qk_library_sha256']}`",
        f"- SOURCE complete original package manifest: `{provenance['source_provider']['source_manifest_sha256']}`",
        f"- New frozen driver: `{provenance['driver_sha256']}`"])
    for name, checksum in provenance['port_native_SHA'].items():
        lines.append(f'- AFTER {name}: `{checksum}`')
    lines.extend(['', 'SOURCE仅原QK native fresh icpx targeted rebuild；其完整原package其它native保留历史构建，不声称SOURCE default-all icpx。AFTER是main真实默认全部功能持久CMake图增量构建/当前图fresh install，不是有限PORT，也不声称每PR独立clean。AFTER public CUTLASS7b56，SOURCE遵守原声明e878，不换数学/deps伪造基线。', '',
        '```bash', f'bash {RUN}/launch.sh',
        f'PYTHONDONTWRITEBYTECODE=1 taskset -c 32-47 python3 {RUN}/prepare_main_release.py',
        'SGLANG_MUSE_CONTAINER=codex-muse-bf16-review-20261015 /home/syk/agent-skills/cri-po-gpu-container/scripts/enter-sglang-syk.sh \\',
        f'  --workdir {str(RUN).replace("/home/syk/", "/root/", 1)} -- bash -c \'export PYTHONDONTWRITEBYTECODE=1; exec taskset -c 32-47 python3 continue_same_suite.py\'',
        f'PYTHONDONTWRITEBYTECODE=1 taskset -c 32-47 python3 {RUN}/audit.py', '```', '',
        '全部原始命令/PID/phase stdoutJSON/实际exits和样本在phases/*/supervisor.*与caseJSON；grant/新CONT release/暂停证明均保留，不编辑旧grant或live runner。final-evidence-audit.json记录全部raw SHA与独立统计重算，drain-check.json记录物理锁释放/无自有live进程。'])
    (RUN / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    print('RESULTS.md written from independently audited completed evidence')


if __name__ == '__main__':
    main()
