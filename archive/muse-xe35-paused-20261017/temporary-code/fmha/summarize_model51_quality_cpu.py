import hashlib
import json
from pathlib import Path
import time


ART = Path(__file__).resolve().parent
ROOT = ART / 'model-config51-v1'


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            checksum.update(block)
    return checksum.hexdigest()


def main():
    assert (ART / 'model51-numerical-quality-CPU-v1.exit').read_text().strip() == '0'
    quality_path = ART / 'MODEL51_NUMERICAL_QUALITY_CPU_v1.json'
    quality = json.loads(quality_path.read_text())
    assert quality['GPU_initialized'] is False and len(quality['cases']) == 51
    plan = json.loads((ROOT / 'plan.json').read_text())
    assert digest(ROOT / 'plan.json') == quality['plan_sha256']
    assert digest(ART / 'launch-v3-icpx/model_semantic_accuracy.py') == plan['harness_sha256']
    assert digest(ART / 'launch-v3-icpx/source_only_accuracy.py') == quality['original_reference_helper_sha256']
    for name, record in plan['fixture_files'].items():
        assert digest(ROOT / name) == record['sha256']
    for provider in ('source', 'port'):
        raw_path = ROOT / f'{provider}-accuracy.json'
        assert digest(raw_path) == quality[f'raw_{provider}_sha256']
        raw = json.loads(raw_path.read_text())
        for case in raw['cases']:
            assert digest(ROOT / f'{provider}-full-outputs' / f'{case["name"]}.pt') == case['saved_full_output_file_sha256']
    freeze = json.loads((ART / 'canonical-source-freeze-v4-icpx.json').read_text())
    assert len(freeze['files']) == 660
    for relative, record in freeze['files'].items():
        assert digest(Path(freeze['source']) / relative) == record['sha256'], relative
    pids = [3202306, 3202370, 3136247, 3136268, 3136851, 3136274, 3144585, 3144692, 3144591]
    live = [pid for pid in pids if Path(f'/proc/{pid}').exists()]
    assert not live
    lock = Path('/tmp/intel-gpu-5.lock')
    locks = [line for line in Path('/proc/locks').read_text().splitlines() if line.split()[5].endswith(':' + str(lock.stat().st_ino))]
    assert not locks
    totals = quality['totals']
    source = totals['source_to_CPU_oracle']
    port = totals['port_to_same_CPU_oracle']
    paired = totals['SOURCE_to_PORT']
    assert source['elements'] == port['elements'] == 808255488
    assert source['violations_atol_plus_rtol'] == port['violations_atol_plus_rtol'] == 0
    worse_maxabs = [case['name'] for case in quality['cases'] if case['port_to_same_CPU_oracle']['max_abs'] > case['source_to_CPU_oracle']['max_abs']]
    lines = ['# model51完整CPU质量量化终态（非optimized PORT验收）', '',
        '唯一session77862实际exit0；CPU16–31/8threads，GPU未初始化。原FP32参考helper逐case完整重算；AST仅插observer，去掉observer与原函数AST相同。同一oracle同时比较两个已保存完整BF16输出。没有重跑GPU/改raw/数学。', '',
        '**真实路径为SOURCE optimized51 → normal AFTER generic51。** 不把当前generic输出合格冒称PORT optimized正确/模型优化覆盖，也不从within_tol推出等价或无局部精度下降。', '',
        '| 指标 | SOURCE→同CPU oracle | PORT generic→同CPU oracle |', '|---|---:|---:|',
        f'| 完整元素 | {source["elements"]} | {port["elements"]} |',
        f'| maxabs | {source["max_abs"]:.17g} | {port["max_abs"]:.17g} |',
        f'| 全元素加权RMS | {source["RMS_error"]:.17g} | {port["RMS_error"]:.17g} |',
        f'| strict005 violations | {source["violations_atol_plus_rtol"]} | {port["violations_atol_plus_rtol"]} |',
        f'| nonfinite | {source["nonfinite"]} | {port["nonfinite"]} |', '',
        f'PORT总加权RMS变化 {totals["weighted_RMS_port_delta_percent"]:+.12g}%；逐case RMS 3例升、48例降。24例PORT maxabs更大，不能声称逐例/最坏误差均不下降。三例升RMS均active3 prefill（+0.000119625961%、+0.000119625961%、+0.000186882664%）；active5三例RMS略降，decode45例RMS降约2.10305%到3.84802%。这是同一合成fixture数值表面，不推断真实model task-quality或所有输入。', '',
        f'SOURCE→PORT maxabs={paired["max_abs"]:.17g}，RMS={paired["RMS_error"]:.17g}；BF16 bit差异{totals["BF16_bitwise_mismatched_elements"]}元素，wholebitwise0/51；max ordered BF16 ULP={totals["max_BF16_ULP"]}，sumULP={sum(case["sum_BF16_ULP"] for case in quality["cases"])}。ULP按单调BF16整数顺序，负数倒序，±0距0但bytes仍独立计数；近零跨符号可有大ULP，不能直接解释为容差违反或精度等价。', '',
        f'更靠近oracle元素：PORT {totals["port_closer_to_oracle_elements"]}；SOURCE {totals["source_closer_to_oracle_elements"]}；abs-error相同 {totals["equal_absolute_error_elements"]}。', '',
        '## 全量ULP分布', '', '| BF16 ULP距离 | 元素数 |', '|---|---:|']
    for label, count in totals['BF16_ULP_histogram'].items():
        lines.append(f'| {label} | {count} |')
    lines += ['', '## 逐case完整质量统计', '',
        'violations列为SOURCE/PORT各自对同oracle。所有query/request/head/channel均检查，不采样。', '',
        '| case | SOURCE maxabs | SOURCE RMS | PORT maxabs | PORT RMS | RMS变化% | violations S/P | paired maxabs | paired RMS | bitdiff count | max ULP |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for case in quality['cases']:
        source_case = case['source_to_CPU_oracle']
        port_case = case['port_to_same_CPU_oracle']
        pair_case = case['SOURCE_to_PORT']
        lines.append(f'| {case["name"]} | {source_case["max_abs"]:.9g} | {source_case["RMS_error"]:.9g} | {port_case["max_abs"]:.9g} | {port_case["RMS_error"]:.9g} | {case["port_RMS_delta_percent"]:+.9g} | {source_case["violations_atol_plus_rtol"]}/{port_case["violations_atol_plus_rtol"]} | {pair_case["max_abs"]:.9g} | {pair_case["RMS_error"]:.9g} | {case["BF16_bitwise_mismatched_elements"]} | {case["max_BF16_ULP"]} |')
    lines += ['', '## Provenance与drain', '',
        f'原完整质量JSON SHA256 `{digest(quality_path)}`；原log/exit/handle `model51-numerical-quality-CPU-v1.*` 全部保留。',
        'normal raw SOURCE/PORT、102完整output、两immutable fixtures、plan/harness/reference helper与冻结actual PR660文件均CPU重新SHA核对不变；supervisor/runtime全drain，GPU5物理锁无持有者，逻辑FM lease保留，无performance/NV grant。']
    (ART / 'MODEL51_NUMERICAL_QUALITY_CPU_v1.md').write_text('\n'.join(lines) + '\n')
    report = {'epoch': time.time(), 'CPU_only': True, 'actual_quality_exit': 0,
        'quality_json_sha256': digest(quality_path), 'quality_summary_md_sha256': digest(ART / 'MODEL51_NUMERICAL_QUALITY_CPU_v1.md'),
        'full51_same_oracle_elements': 808255488, 'strict005_SOURCE_PORT_violations': [0, 0],
        'cases_PORT_RMS_higher': len(totals['cases_port_RMS_greater']),
        'cases_PORT_RMS_lower': len(totals['cases_port_RMS_lower']),
        'cases_PORT_maxabs_higher': worse_maxabs,
        'normal_raw_102_full_outputs_2_fixtures_plan_harness_reference_SHA_unchanged': True,
        'frozen_actual_PR660_SHA_unchanged': True, 'source_digest': freeze['source_digest'],
        'actual_PR_writes': 0, 'checked_runtime_supervisor_PIDs': pids, 'live_checked_PIDs': live,
        'GPU5_physical_lock_holders': locks, 'logical_FM_GPU5_lease_retained': True,
        'new_GPU_compile_performance_NV_handoff': False,
        'current_route': 'SOURCE optimized6prefill/45decode; AFTER generic51; not optimized PORT acceptance'}
    (ART / 'MODEL51_QUALITY_PROPOSAL_CPU_POSTVERIFY_v1.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
