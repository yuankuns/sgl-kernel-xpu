import hashlib
import json
import os
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent
target = root.parents[1] / 'pr/muse-qk-norm-rope-xe35'


def read_json_xe35(name, default):
    path = root / name
    return json.loads(path.read_text()) if path.exists() else default


def exit_xe35(name):
    path = root / f'{name}.exit'
    return path.read_text().strip() if path.exists() else '尚未完成'


accuracy = read_json_xe35('accuracy.json', {})
timing = read_json_xe35('summary.json', {})
provenance = read_json_xe35('provenance.json', {})
source = read_json_xe35('source-verification.json', {})
commands = read_json_xe35('commands.json', [])
diagnostic_provenance = read_json_xe35('diagnostic/provenance.json', {})
diagnostic_run = 'diagnostic-retry1' if (root / 'diagnostic-retry1').exists() else 'diagnostic'
diagnostic_accuracy = read_json_xe35(f'{diagnostic_run}/accuracy.json', {})
diagnostic_timing = read_json_xe35(f'{diagnostic_run}/summary.json', {})
eightcard_run = 'diagnostic-eightcard-20261017'
eightcard_info = read_json_xe35(f'{eightcard_run}/run-info.json', {})
eightcard_accuracy = read_json_xe35(f'{eightcard_run}/accuracy.json', {})
eightcard_timing = read_json_xe35(f'{eightcard_run}/summary.json', {})
port_manifest = read_json_xe35('port/site-manifest.json', [])
handoff = read_json_xe35('full-main-handoff.json', {})
release_policy = read_json_xe35('official-release-policy.json', {})
ready = read_json_xe35('official-source-ready.json', {})
canonical_version = os.environ.get('QK_CANONICAL_VERSION', 'canonical-v2')
assert canonical_version == 'canonical-v2', 'Official v1 release paused'
coordinator_ready = read_json_xe35(f'{canonical_version}/source-snapshot.json', {})
official_run = canonical_version + '/' + os.environ.get('QK_REPORT_OFFICIAL_RUN', 'official-v2')
formal = read_json_xe35(f'{official_run}/formal-checks.json', {})
if (root / official_run / 'provenance.json').exists():
    provenance = read_json_xe35(f'{official_run}/provenance.json', {})
    accuracy = read_json_xe35(f'{official_run}/accuracy.json', {})
    timing = read_json_xe35(f'{official_run}/summary.json', {})
    commands = read_json_xe35(f'{official_run}/commands.json', [])
performance_regression = any(record.get('retest', record)['change_pct'] > 5 for record in timing.values())
passed = formal.get('accuracy_verified', False) and formal.get('default_all_command_verified', False) and formal.get('all_runtime_mapped_dso_verified', False) and exit_xe35(f'{official_run}/supervisor') == '0' and not performance_regression
status = '默认全构建与已分配 GPU 验证通过' if passed else '尚未完成验收（不可声明 P0 PASS）'
if handoff and not passed:
    status = 'own per-PR full 已安全停止；等待 main canonical 真实 default-all 与 fresh install，尚未 official 验收'
if release_policy and not release_policy['official_gpu_launch_authorized']:
    passed = False
    status = '等待独立icpx C++ full/fresh新pkg和new GPU6 grant；GNUhost+icxSYCL v2仅诊断，禁止official GPU6放行'
text = f'''# QK norm + RoPE Xe35 upstream port

**状态：{status}。**

最新icpx要求覆盖下方v2计划：main queue STOP，root manifest权限不处理；冻结PR未改，当前没有GPU6/waiter/compiler进程。v2 GNUhost+icxSYCL CPU default-full/install只保留诊断，旧v2启动命令不得执行，不能称新compiler要求下的official PASS。`official-release-policy.json`关闭放行。等待main独立icpx C++ genuinefull/currentfresh正常包、新GPU6 grant；source-port和原preopt独立额外授权保留，失败baseline不得修/冒称正确性匹配加速。

最新 full 去重调度以 full-main-handoff.json 为准：固定 canonical snapshot、持久 CMake tree 真实 default-all 增量以及当前图 fresh install；不 copy objects/旧 libraries，不宣称每 PR clean build。Own handle2096 已真实退出130，完整55进程子树停止，stop审计0；没有残留 own compiler/Ninja/official waiter。原日志/退出码、commands/config/dependency text logs/resource snapshot 均保留 full-j8-v2-stop-evidence/manifest.json。Main pool56 jobs/CPU0–127，reserve8诊断compiler；GPU6 lease、runtime CPU32–47不变。禁止自行恢复本 lane full；等 main canonical正常新package后再做 GPU6 原严格精度与 BAAB 正式复验。以下 own full 命令和进度仅为历史记录，不是恢复授权。

最新性能接受仅GPU4–7，GPU8–11只作精度，GPU0–3保留。QK GPU6当前own进程全结束/锁无持有者，lease继续归QK；没有基于GPU8–11性能回归改kernel或dispatch，保留SOURCE feed315算法/precision/launch。W4A4未来GPU6切换需main显式grant；见gpu6-current-status.json。

## Canonical v2 官方入口准备（放行暂停）

原QK v1冻结SHA：`{ready.get('source_digest', '未冻结')}`（历史，非v2）。Main v2实际source SHA：`{coordinator_ready.get('source_digest', 'PENDING main freeze v2')}`。`official-source-ready.json` 使用 main overlay_snapshot 完全相同 source_inventory/file_record/manifest_digest，记录639个文件、size/mode/hash及base833c952，不以未提交 worktree 的 HEAD 冒充实际修改 SHA。只冻结 target，未修改 main canonical helpers/live scripts，未启动额外 full/GPU watcher。

官方 AFTER 仅接受 `canonical-v2/site` 与 `canonical-v2/source-binary-manifest.json`（stage_install.py payload），来自 default-all 增量和当前 native graph fresh install，不解压旧 wheel、不复制 diagnostic ABI bridge/native/library/object、不使用共享 pip/shim/load_library。`canonical_check.py` 先验证 snapshot 等于main v2 freeze、仅允许main-owned src/sycl/Device.cpp相对原QK冻结的变化、public7b56/default features、真实完整命令/log/exit、完整 package 文件 hash、generic链接Xe35以及两套 public native T symbols；缺包 fail-closed，不称 official PASS。

`launch_xe35.sh` 先B8拒绝、再CPU preflight，之后等待 main 独立未来 official barrier；仅GPU6/CPU32–47。新结果在 `canonical-v2/official-<tag>/`；normal package import后每个子进程通过 `runtime_check.py` 验证全体实际映射的 libsgl-ops/common_ops 路径与 manifest SHA，再执行原完整 reference/tests/BAAB，计时热路径无新增代码。SOURCE依旧feed315已校验完整历史包；PORT必须真正新包。arch必须3.5、无XPU先短路、six tests实际PASS不能全skip。

CPU自检 `canonical-preflight-audit.exit={exit_xe35('canonical-preflight-audit')}`；当前 missing-package负例 `canonical-missing-package.exit={exit_xe35('canonical-missing-package')}` 是预期拒绝，不是构建失败。容器 ownership首次预检exit1保留，改为本进程GIT_CONFIG safe.directory（不写global）后的 `canonical-container-preflight-v2.exit={exit_xe35('canonical-container-preflight-v2')}` 证明host/container同一READY SHA，无GPU工作。Main canonical full句柄live23436由用户通知；本worker不监控/修改该live shell。

## 最新放行暂停与v2协议

BF normal-v1真实query_device报Unsupported；shared src/sycl/Device.cpp仅guard __INTEL_LLVM_COMPILER，而GNUhost仅提供显式_VERSION别名。Main owns全部11共享Device局部releaseguard修复和freeze v2，本worker不改actualPR、不fakearch、不设global __INTEL_LLVM_COMPILER、不改数学。CPU queue parent2701478由main STOP防后继，FM v1允许自然完成；本worker不操作这些进程。QK实际AFTER仅canonical-v2 genuine default-all/current fresh install，v1 checker/launcher放行拒绝。预opt额外授权仍有效但须新版包、新main GPU6 grant；现在没有启动old waiter/GPU。

官方release JSON必须含gpu=6、coordinator_release=true、canonical_version=canonical-v2、source_digest等于main v2 freeze、未来start_epoch；默认文件gpu6-official-canonical-v2-barrier.json。拒绝旧version/旧digest/旧epoch，以及无版本MUSE_SYNC_START_EPOCH替代。原official-source-ready.json保留历史，不自动重写；checker消费main canonical-v2/source-snapshot.json（或显式--ready-manifest），fail-closed直到main真实freeze/pkg。GPU实际arch继续normalpackage query_device严格3.5，不替换查询结果。

## 范围与来源

- Official target：`{target}`，分支 `pr/muse-qk-norm-rope-xe35`，tracking `origin/main`，base `833c95208dccd2af5628c121222f91186a5b2de7`。官方主 checkout `/home/syk/sgl-kernel-xpu-up` 未修改。
- Personal source：`/home/syk/worktrees/sgl-kernel-xpu/muse-spark-qk-norm-rope-main-20261014`，分支 `muse_qk_norm_rope`，SOURCE `feed315712ea91e6becbb983f3a4d56ac0cc3b96`，source base `ab3a26c` 仅作为来源，不作为此次性能 baseline。
- B8 / ISL4096 / OSL16 / TP1 / EP1；生产 BF16，T8/T32768，Q128/K16/tail144/D64。没有其他 serving batch 测试；FP16 仅验证已支持类型，不能解释为生产精度修改。
- 无 agent spawn、commit、push、PR。只修改 target 和本 artifact。
- 最新调度：QK 从 GPU2 改为 GPU6，CPU lane2（32–47）不变。已启动 GPU2 suite 仅自然结束；不再新启动 GPU0–3。GPU6 必须等待 main 4–11 八卡 barrier。历史 GPU2 数据保留，不冒充 GPU6 结果。

## 实现与兼容性

- 移植 source immutable FP32 frequency table：CPU FP64 negative-exponent pow 后一次舍入至 FP32，再保持原 YaRN 算法；移植 source optimized kernel 与原 launch 选择，不做新的优化。
- 新 CPP/optimized/frequency headers 全部位于 `src/sycl/xe35/`，使用 generic basenames；Python 新文件不带 Xe35 后缀，使用 `bench_muse_qk_norm_rope.py`、`validate_qk_norm_rope_b8.py`、`muse_scope.py`，避免覆盖既有 generic benchmark。必要 helper/registered op/Python entry 的 `_xe35` 隔离保留。新声明位于 `include/sgl_kernel/xe35/fused_qk_norm_rope.h`，generic header 仅 guard include。现有 generic API 签名不变；`SYCL_INTEL_TARGET == 35` guard 实际调用新实现，BMG 原实现保持，CRI FP32 仍走原 generic 路径。
- CRI-only CMake 分类与 generic→Xe35 DSO 显式链接；正常完整 package import 注册 op，无 shim、显式 load_library、共享 pip 安装或热路径数据搬运。
- `algorithm-equivalence.json` 验证三个 kernel 源码除机械命名/export/include 路径外与 feed315 完全一致，所有 quoted includes 可解析；共享 MemoryAccess/SYCLHelpers/Utils/Norm headers 未搬动或修改。原 `tests/test_fused_qk_norm_rope.py` 扩展新 cases，旧 cases 字节级保留；official GPU6 命令只 `-k` 选择新增 B8 cases。`bmg-static-audit.json` 验证去掉新增 guard/include 后 generic CPP 与 base 字节一致，existing Python APIs 不变。BMG runtime 未验证。
- Host `launch_xe35.sh` 在 wrapper/GPU 前拒绝其他 batch 或 ISL/OSL/TP/EP 覆盖、任何 CLI 覆盖。Python bench/validator 在 torch import 前同样检查，20 项 CPU-only 拒绝验证见 `scope-audit.json`；host 拒绝 exit{exit_xe35('scope-reject')}。
- 仅两组新增 `_xe35` GPU tests 标记 `pytest.mark.arch("xe35")`，沿用 shared tests/_xpu.py：无 XPU 先短路，不查询 current_device/query_device，识别条件精确 major=3/minor=5。原 generic/shared QK tests 与 BMG fallback 保持，不做整文件 skip。新 bench/validator 沿用 benchmark/_bench.py 精确架构 probe，在分配/actual op 前明确拒绝非CRI；B8 preflight 仍先于 torch import。arch-guard-audit.json 是纯 CPU/offline control-flow 审核，不是假注册、shim 或 GPU acceptance；最新 guard GPU执行由后续 official 包验证，未重启 full build。

## 构建、包隔离与资源

容器 `codex-muse-bf16-review-20261015`，wrapper 显式 `/root/worktrees/sgl-kernel-xpu-up/pr/muse-qk-norm-rope-xe35` workdir。所有编译经容器内 `bash .../build-lane.sh 2`，CPU32–47；外层 cap8 jobs；初始 heavy jobs2，最新用户授权 host约854GiB available 后扩展实际8 jobs，内层 MAX_JOBS=1 限制 per-job parallel-link。按协调指令 per-file fatal monitor 配置8GiB，未关闭 monitor 或 feature。

默认 USE_MOE/FMHA/GDN/MLA/QSA=ON、USE_MLA_SPARSE_FUSED/SYCL_JIT=OFF。真实默认全功能 wheel build，不把 targeted build 声称为 full。最终依赖只读共享 public clone `.../deps/sycl-tla-7b56`，HEAD `7b56b936b87f799f80dde60c2fd8d11c1128cebb` 且 clean，不以内部 pin 替代。

首次尝试 exit{exit_xe35('build-initial')}：op 与 source namespace 同名，已仅机械改名并消除 upstream macro 重名，日志 `build-initial.log`。首次依赖 `/root/inkling-artifacts/sycl-tla` 也检查为 clean 7b56。最终当前构建 `build-full-j8-v2.log` / `build-full-j8-v2.exit`（完成后同步 canonical `build-full.exit`）：exit{exit_xe35('build-full')}。

SOURCE 完整隔离包 `qk-before-after-20261017/after/site` 的 {source.get('source_site_files_verified', '未验证')} 个文件重新校验 SHA256；{source.get('source_inputs_git_bytes_verified', '未验证')} 个模块 source/API 文件匹配当前 git bytes。历史 SOURCE 的 strict accuracy T8/T32768 为零违规，可复用证据，但不替代 fresh PORT 验证。SOURCE 全包存在非 QK `TripleOps.cpp` 的 frozen-source 来源差异；本模块没有调用该辅助实现，不宣称 SOURCE 整包全部源码等于 feed315。

PORT 官方路径为 `canonical-v2/site` 的当前graph fresh install；旧 `prepare.py`/wheel/`port/site` 仅历史，不能作为新AFTER。官方 supervisor exit{exit_xe35(f'{official_run}/supervisor')}；各子命令退出码见 `{official_run}/commands.json` 与各 `.exit`。

Filename 修正前确认自己的 supervisor 安全停止、编译子进程全部退出，再做11项 rename/测试合并与 lane2 重配；`rename-write-set.json` 记录实际写集。旧 terminal handle 81818 已结束，初始 j2 pipeline handle33762（host pipeline1540730，supervisor1540796）已按新调度安全终止，真实exit130；当前唯一 j8-v2 pipeline handle2096（host pipeline2214608，supervisor host2214675/container82607）。旧停止后的 launcher 记录 exit127：此前运行中编辑 launcher 造成 shell 续读偏移，旧 full log 尾部被覆盖，保留 `build-pre-rename-stop.log/.exit`，不把该尝试声称成功或假造丢失日志；新 launcher 稳定后不再 live 编辑，继续真实 default 全构建。

## 严格精度

两侧独立 CPU FP32 全量 RMSNorm/rotation reference，CPU FP64 frequency 舍入至 FP32。生产 `atol=rtol=5e-3`，逐元素 `abs(got-ref) <= 0.005 + 0.005*abs(ref)`，不放宽。检查所有 Q/K、有限值、tail uint16 bits 完全一致；SOURCE/PORT 相同 seed 输入与输出全量 SHA256 必须相同。FP16 独立 validator 与生命周期测试保留1e-3门槛。

| 实现 | T | 元素数 | 违规 | Max abs | Mean abs | Tail bits | 结论 |
|---|---:|---:|---:|---:|---:|---|---|
'''
for label in ('source', 'port'):
    for tokens in (8, 32768):
        record = accuracy.get(f'{label}-t{tokens}')
        if record:
            text += f"| {label.upper()} | {tokens} | {record['elements']} | {record['violations']} | {record['max_abs']:.10g} | {record['mean_abs']:.10g} | {record['tail_equal']} | PASS |\n"
        else:
            text += f'| {label.upper()} | {tokens} | — | — | — | — | — | 未验证 |\n'
text += '''
`accuracy.json`、`source/accuracy-t*.log` 与 `port/accuracy-t*.log` 保存精度统计、全量 input/output hash、实际映射库/hash。`port/pytest.log` 包含 direct Xe35、generic dispatch bitwise 一致、首次 graph capture、并发 streams/table 生命周期，生产 T8/T32768 与 FP16 类型支持。测试/FP16 命令与退出码见 commands。

## 真正优化前→优化后（独立证据，GPU6正式复验PENDING）

以下与本PR的optimized SOURCE→PORT移植对比严格分开。真实preopt provider为 `qk-before-after-20261017/before/site`，QK源码base `ab3a26c43a38e6aea7d73cc1349e4b5e210b772b`，原ABI `sgl_kernel.fused_qk_norm_rope`，QK DSO SHA256 `57749179e38893c1710c0f382603f46524e2552e567355fa78e10a156e4f0d99`。真实postopt为feed315的 `after/site`，DSO `d21098cae486c3082e26c0e9f420391b645aebdb768c1871281793012369a96d`。两套215文件完整正常package已CPU重新验hash，base23/after25输入逐项匹配git bytes；BEFORE package使用相同完整SOURCE package/common_ops/Python，仅QK DSO为hash验证的preopt，**不是official ab3a整包fresh full build**。证据目录 `/home/syk/worktrees/sgl-kernel-xpu/review-validation-artifacts/qk-before-after-20261017` 的provenance/runtime/commands/accuracy/summary/raw JSON及本目录preopt-evidence-audit.json。

历史GPU7同卡B8/ISL4096/OSL16/TP1/EP1、BF16 Q128/K16/tail144/D64，独立CPU full reference atol=rtol=5e-3，tail bits全部一致；20实际命令日志/退出码和16个BAAB图测/336 raw samples重新验hash。统计为每侧4个process p50的中位数：

| T | preopt→postopt违规数 | preopt→postopt µs | 历史变化 | 精度结论 |
|---|---:|---:|---:|---|
| 8 | 0→0 | 2.942703125→2.430015625 | −17.4223%，1.21098x | 两侧历史full reference通过 |
| 32768 | 3508→0 | 3967.080078125→1688.379703125 | 原始诊断时延，不报告正确性匹配的加速比 | preopt exit1，postopt exit0；preopt不满足生产精度，不能称合格加速 |

**GPU6正常官方新包preopt→postopt正式复验仍PENDING**，不得用optimized feed315 SOURCE→PORT表顶替。当前 `canonical-v2/site`/manifest尚不可用；即使新包完成，generic BF16在Xe35上guard实际转发optimized实现，不是preopt；Float32 fallback不能改变生产BF16来充baseline。保留的 `fused_inplace_qknorm_rope` 等上游API只是候选不同路径，尚未证明同QKV布局/cache频率/完整reference/精度语义等价，不能冒认ab3a preopt。official833c952原QK CPP SHA `963193678355ac58a8ab11fec1a884d0a4c59ca80e8022e133136ecf611aa9dd` 与ab3a `cca56a59ea22eb4b130d5b5947eb2a83b085b39abb2b09586f93e5cf87bfe85e` 不同，不能混称same baseline。

后续如main授权：在GPU6独立fresh barrier、同卡锁下以原BEFORE正常package作明确历史preopt provider，AFTER只用canonical正常新package，不替换/复制AFTER native、不创建fake op；完整reference保留BEFORE失败、AFTER必须严格0违规，再BAAB两轮32×21。预计额外4–6分钟GPU lease（历史20个子进程166.86秒 + shape cooling40秒 + initial5秒 + 导入/hash审核余量，排除等待包/构建/barrier）；>5%退化复测再加约1–2分钟。旧GPU7 launcher仅证据，禁止启动。这轮只CPU核查/文档，不launchGPU，不改变冻结source。

未来额外GPU6复验已经用户授权，前提仍是actual canonical正常新pkg及main新official barrier，**现在不启GPU**。授权 `preopt-authorization.json`；未运行 `run_preopt.py` 在同一个GPU6锁内、独立optimized SOURCE→PORT suite之后执行，独立输出 `official-<tag>/preopt-to-port/`。`runtime_check.py` 对preopt numerical AssertionError保留原traceback和真实exit1，同时输出normal package/全映射DSO audit；SOURCE/PORT原失败处理不放宽。PORT仍严格零违规；preopt失败不修、不改变频率/精度，不要求错误baseline全output bitwise等于PORT，仅同input hash与tail bits。失败形状summary无speedup/change_pct字段，只保留raw timing/违规/actual exit，不冒称正确性匹配加速。GPU6转W4必须两个suite全部drain、锁释放并main grant。

当前benchmark provider审核：专用 `bench_muse_qk_norm_rope.py` 只调用当前正常包generic op，没有preopt selector；stock `bench_fused_qk_norm_rope.py` 的 `sglang` 同样取决于正常包dispatch，`torch` 是reference不是历史SYCL preopt，其默认eps=1e-6/base=10000也不同于B8生产验证eps=1e-5/base=500000。`native_split`/`cache_fused` 是独立Q/K和预生成cos/sin cache路径，不能不经完整production reference就称preopt。历史真正的before/after来自两套isolated full-package的相同sglang provider、不同且hash验证的QK DSO，而非provider字符串本身。

## optimized SOURCE→PORT 性能（仅移植等价性，非优化前后）

新 official 验收分配 GPU6，同一个 `--gpu-lock --gpu 6` wrapper 覆盖整套测试；首次冷却5秒，进程结束同步后/形状间至少2秒。每形状 SOURCE→PORT→PORT→SOURCE（BAAB）两轮；每进程 XPUGraph repeats32 / samples21 / warmup5，XPU event 设备时间除以32。21 raw samples、p50/MAD/p20/p80、实际加载库 hash 均保留在各 graph JSON/log 中。主表为每侧四个独立进程 p50 的中位数；不是旧 base→optimized 对比，也不是 serving E2E 验证。

| 形状（生产 BF16） | SOURCE before µs | PORT after µs | PORT 延迟变化 | source/port speedup |
|---|---:|---:|---:|---:|
'''
for tokens in (8, 32768):
    record = timing.get(str(tokens))
    if record:
        text += f"| T{tokens} | {record['source_us']:.3f} | {record['port_us']:.3f} | {record['change_pct']:+.2f}% | {record['source_us']/record['port_us']:.4f}× |\n"
    else:
        text += f'| T{tokens} | 未测 | 未测 | — | — |\n'
for tokens, record in timing.items():
    text += f"\nT{tokens} SOURCE 四进程 p50：`{record['source_process_p50_us']}`；PORT：`{record['port_process_p50_us']}`。\n"
    paired = [before / after for before, after in zip(record['source_process_p50_us'], record['port_process_p50_us'])]
    text += f'四组 paired speedup：`{paired}`。\n'
    if 'retest' in record:
        retest = record['retest']
        text += f"\n初测超过5%退化，因此同 GPU6、同 binary/shape 再执行 BAAB 两轮（r3/r4）而没有修改 kernel。复测 SOURCE `{retest['source_process_p50_us']}`，PORT `{retest['port_process_p50_us']}`，中位数 `{retest['source_us']:.3f}`→`{retest['port_us']:.3f}` µs，变化 `{retest['change_pct']:+.2f}%`。"
        if retest['change_pct'] > 5:
            text += '退化仍存在；机械源码等价不代表编译后性能相同，尚需进一步定位，不宣称性能验收通过。\n'
        else:
            text += '复测没有重复>5%退化，初测归为未重复的进程/设备波动，不将其作为确定性算法退化。\n'
text += '\n## 优先执行的实际 PORT-kernel 诊断（不是 official 整包 PASS）\n\n'
text += '新 public native entry 已使用 SGL_KERNEL_EXPORT；visibility-audit.json / visibility-fresh_port.log / visibility-native_abi_forward.log 记录 nm -D -C 的实际 global T 符号，未只凭源码宏判断。官方 generic DSO 在整包链接后再核对；目前该 DSO 尚未生成。\n\n'
text += '完整 SOURCE 原 sgl_kernel 包正常 import；原 common_ops 注册及 Python 文件全部保持原字节。仅将原 QK DSO 替换为 C++ 原 ABI forward，并加入 target 实际 fresh 编译的 Xe35 kernel DSO。forward 无注册、无 shim、无 load_library、无 kernel 副本或精度转换；ELF DT_NEEDED 指向实际 Xe35 DSO，RUNPATH=$ORIGIN。214 个 SOURCE 包文件不变，仅一个库替换和一个库新增。\n\n'
text += f"诊断 native build exit{exit_xe35('diagnostic/build')}；prepare exit{exit_xe35('diagnostic/prepare')}；GPU2 supervisor exit{exit_xe35('diagnostic/supervisor')}。官方当前 full pipeline handle `2096`（旧 `33762`已退出130），诊断 ABI build handle `47229`（已结束），GPU2 诊断 handle `81447`。初次 prepare 因 SOURCE common_ops 使用 CPython 文件名而非 abi3 文件名退出1，原日志/退出码保留为 `diagnostic/prepare-initial.*`，修正为检查实际完整包扩展名后成功。\n\n"
if diagnostic_run != 'diagnostic':
    text += f"首次 GPU 诊断 exit1：PORT T32768 在 `qkv.cpu()`（未调用 QK kernel）报 UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY；SOURCE T32768 同配置已通过。所有失败 raw logs 保留原 `diagnostic/`。随后 GPU2 独立状态查询可用内存约120GiB、allocator active=0，未改 kernel/精度/输入/包，只在 fresh 进程与5秒 cooling 后同卡重跑完整诊断，handle `86189`，目录 `{diagnostic_run}/`，exit{exit_xe35(f'{diagnostic_run}/supervisor')}。下表仅引用该复测实际完成记录；尚不能仅凭 free-memory 查询断定瞬时 OOM 的根因。\n\n"
text += '| 独立 full CPU reference | max abs | mean abs | violations | tail bitwise |\n|---|---:|---:|---:|---|\n'
for label in ('source', 'port'):
    for tokens in (8, 32768):
        record = diagnostic_accuracy.get(f'{label}-t{tokens}')
        if record:
            text += f"| {label.upper()} T{tokens} BF16 | {record['max_abs']:.9g} | {record['mean_abs']:.9g} | {record['violations']} | {record['tail_equal']} |\n"
        else:
            text += f'| {label.upper()} T{tokens} BF16 | 未验证 | — | — | — |\n'
text += '\n| 诊断 BAAB 两轮 | SOURCE µs | PORT µs | 变化 |\n|---|---:|---:|---:|\n'
diagnostic_details = []
for tokens in (8, 32768):
    record = diagnostic_timing.get(str(tokens))
    if record:
        text += f"| T{tokens} | {record['source_us']:.3f} | {record['port_us']:.3f} | {record['change_pct']:+.2f}% |\n"
        diagnostic_details.append(f"\nT{tokens} SOURCE process p50 `{record['source_process_p50_us']}`；PORT `{record['port_process_p50_us']}`。\n")
        if 'retest' in record:
            diagnostic_details.append(f"\n>5%触发未改变代码的额外同卡 BAAB 两轮：`{json.dumps(record['retest'])}`。仍>5%不视为性能验收通过；需对编译选项/设备状态解释，不新增 kernel 优化。\n")
    else:
        text += f'| T{tokens} | 未测 | 未测 | — |\n'
text += ''.join(diagnostic_details)
text += f"\n诊断 SOURCE QK SHA256 `{diagnostic_provenance.get('source_qk_library_sha256', '未验证')}`；实际 PORT kernel `{diagnostic_provenance.get('port_qk_library_sha256', '未验证')}`；native forward `{diagnostic_provenance.get('native_forward_sha256', '未验证')}`。source/port manifest、ELF 在 `diagnostic/`，当前命令、raw samples、logs/exit 在 `{diagnostic_run}/`。该证据只说明实际 kernel 通过原完整 SOURCE 包的 ABI 路径，不证明 official wheel、新 op/Python entry、official lifetime tests 或默认 full build 已通过。\n"
text += f'''\n同步八卡复测仅由协调方安排，本 lane 只启动 GPU6（CPU lane2）：

```bash
MUSE_SYNC_START_EPOCH=<共同host_epoch> QK_DIAGNOSTIC_RUN=diagnostic-eightcard-<唯一tag> bash {root}/launch-diagnostic_xe35.sh
```

启动前 host 拒绝其他 serving batch；共同 epoch 后取得 GPU6 flock，并5秒 cooling。未给 epoch 时等 main barrier JSON（默认 gpu6-barrier.json，或 MUSE_BARRIER_FILE），需要 gpu=6、coordinator_release=true、start_epoch 字段。每进程 XPU synchronize 后间隔2秒，两形状均为生产 BF16、graph repeats32/samples21，BAAB 两轮。独立 run 目录避免覆写原 raw logs；不能自行创建 release barrier。
'''
text += '\nGPU2 diagnostic-retry1 suite 已自然结束 exit0，own wrapper/子进程退出且 fuser 无持有者，释放证据 gpu2-release.json；此后所有 active launcher 固定 GPU6。\n'
if eightcard_info:
    text += f"\n## Main 4–11 同步轮次的 GPU6 实测\n\nmain 发布 gpu6-barrier.json 后，handle16686 从 CPU-only waiter 转为 GPU6 wrapper；实际 ZE_AFFINITY_MASK={eightcard_info['gpu_affinity']}，CPU affinity32–47，运行目录 `{eightcard_run}`，supervisor exit{exit_xe35(f'{eightcard_run}/supervisor')}。仅该卡 SOURCE/PORT 同卡数据由本 worker 验证，不冒充 official full-package pass。\n\n"
    text += '| GPU6 full CPU reference | violations | tail bitwise | SOURCE/PORT output hash |\n|---|---:|---|---|\n'
    for tokens in (8, 32768):
        source_record = eightcard_accuracy.get(f'source-t{tokens}')
        port_record = eightcard_accuracy.get(f'port-t{tokens}')
        if source_record and port_record:
            text += f"| T{tokens} BF16 | {source_record['violations']}/{port_record['violations']} | {source_record['tail_equal']}/{port_record['tail_equal']} | {source_record['output_sha256'] == port_record['output_sha256']} |\n"
        else:
            text += f'| T{tokens} BF16 | 未完成 | — | — |\n'
    text += '\n| GPU6 BAAB 两轮 | SOURCE µs | PORT µs | 变化 |\n|---|---:|---:|---:|\n'
    for tokens in (8, 32768):
        record = eightcard_timing.get(str(tokens))
        if record:
            text += f"| T{tokens} | {record['source_us']:.6f} | {record['port_us']:.6f} | {record['change_pct']:+.4f}% |\n"
        else:
            text += f'| T{tokens} | 未完成 | 未完成 | — |\n'
    for tokens, record in eightcard_timing.items():
        text += f"\nT{tokens} SOURCE process p50 `{record['source_process_p50_us']}`；PORT `{record['port_process_p50_us']}`。\n"
        if 'retest' in record:
            text += f"\n>5%同卡额外 BAAB 两轮复测：`{json.dumps(record['retest'])}`，不修改 kernel/精度，仍>5%不通过性能接受。\n"
    text += f"\nGPU6 精度/hash、21 samples 原始 graph JSON、log/exit、commands.json、run-info.json、barrier epoch 均保留。CPU-only integrity audit exit{exit_xe35(f'{eightcard_run}/integrity')}。Official full wheel 使用独立 gpu6-official-barrier.json，不复用已经释放的 diagnostic barrier。\n"
    release = read_json_xe35('gpu6-diagnostic-release.json', {})
    if release:
        text += '\nGPU6 diagnostic wrapper/子进程已全部结束，flock 释放证据 gpu6-diagnostic-release.json（fuser 无持有者）；仍等默认 full build 与独立 official barrier，不将 native diagnostic 当作官方整包验收。\n'
else:
    text += '\nGPU6 CPU-only diagnostic barrier waiter handle16686 已启动，run目录 diagnostic-eightcard-20261017，等待 gpu6-barrier.json；当前未持 GPU6 lock，main 不要重复启动同 run launcher。Official full wheel 阶段使用独立 gpu6-official-barrier.json，不复用已释放的 diagnostic barrier。\n'
text += '\n原 j2 停止证据 build-j2-stop.json/log/exit：核对自身 cwd/cmd/PID 后 SIGTERM supervisor，runner graceful SIGINT 子树，确认所有 compiler/Ninja 结束。原日志 build-full-j2.log 与 build-full.log SHA256相同，真实退出130保存 build-full-j2.exit。新稳定 versioned shell 启动前完成全部写入，未修改 live shell；原tree增量恢复8jobs，--reset-history仅因明确资源授权改变而重置旧parallelism history，不clean build、不关闭features。资源快照 build-j8-v2-runtime.json / resource.log / exit0，CPU32–47，MAX_JOBS=1，8GiB/file fatal monitor，public7b56不变；当前 handle2096，host pipeline2214608 / supervisor2214675，container82607。\n'
text += '\n## Hash 与退出码\n\n| 项目 | SHA256 / exit |\n|---|---|\n'
text += f"| SOURCE QK library | `{source.get('source_qk_binary_sha256', '未验证')}` |\n"
for record in port_manifest:
    if 'FusedQKNormRope' in record['path'] or '/common_ops.' in record['path']:
        text += f"| `{record['path']}` | `{record['sha256']}` |\n"
if provenance:
    text += f"| canonical source-binary manifest | `{provenance.get('canonical_manifest_sha256', '历史非canonical')}` |\n"
for name in ('source-verification', 'scope-reject', 'format-check', 'build-initial', 'build-full', 'prepare', 'supervisor'):
    text += f'| {name} | exit{exit_xe35(name)} |\n'
text += f'''
`provenance.json` 保存 source/port 全输入 hash 与 wheel hash，两个 site manifest 保存包文件 hash，`SHA256SUMS` 包括代码/日志/raw JSON。各执行命令、退出码、log hash 在 `commands.json`。仅文件/静态审计成功不代表 full build 或 GPU 测试通过。

## 历史执行命令（当前禁止恢复 own full）

在 host `{root}`：

```bash
bash build-full-j8-v2_xe35.sh
python3 verify_source.py
python3 audit_port.py
python3 prepare.py
bash launch_xe35.sh
python3 report.py
```

完整 wrapper、容器 env、build-lane、workdir、CMake 资源配置均在脚本内。prepare 拒绝覆盖现有 stage，复用 stage 前应重新核对其 manifest，不使用 shared pip 安装。运行必须先取得协调方 GPU6 分配并等待 main 4–11 barrier；GPU0–3 已交还用户，严禁重新启动。

## 当前官方复验命令（仅main包与fresh barrier均ready后）

```bash
PYTHONDONTWRITEBYTECODE=1 python3 canonical_check.py --canonical-version canonical-v2
QK_CANONICAL_VERSION=canonical-v2 QK_OFFICIAL_RUN=official-v2 MUSE_BARRIER_FILE=<main独立fresh-GPU6-official-JSON> bash launch_xe35.sh
QK_REPORT_OFFICIAL_RUN=official-v2 python3 report.py
```

第一次命令只校验，不复制任何包。launcher拒绝重复run和旧epoch，保持原SOURCE-to-PORT严格BF16 full accuracy、tail bits、T8/T32768 BAAB两轮/repeats32/samples21；>5%同GPU6追加两轮并保留raw samples。旧full/prepare历史命令不得恢复。本次无新GPU supervisor handle；GPU6锁空闲但lease保留，W4A4仅main grant切换。

## 未能验证/限制

- BMG runtime、非生产 FP8/FP32 Xe35 数值覆盖、serving E2E 未测试；BMG 仅静态保留审核，生产只 BF16。
- 历史 SOURCE 包复用已校验 hash/模块 git bytes，但非 QK TripleOps frozen-source 差异如上，不能宣称 SOURCE 整包从 feed315 本轮 fresh build。
'''
if not passed:
    text += '- 默认全功能 PORT build、normal import/precision/performance 仅以已记录 exit0 为准；无完成日志的项尚未验证，当前不是 P0 接受结果。\n'
if performance_regression:
    text += '- >5%性能退化复测仍存在，未作新的 kernel 优化，需要对编译/设备状态进一步定位，不宣称性能验收通过。\n'
if passed:
    text += '- GPU wrapper 已结束（supervisor.exit=0），相应子进程退出后 flock 已释放；外部 lease 仍需协调方重新分配。\n'
changed = subprocess.check_output(['git', '-C', str(target), 'ls-files', '--modified', '--others', '--exclude-standard'], text=True).splitlines()
text += '\n## 修改路径\n\n' + ''.join(f'- `{relative}`\n' for relative in changed)
text += f'\nArtifact：`{root}`。\n'
old = root / 'PR_DRAFT.md'
patch = '*** Begin Patch\n*** Delete File: ' + str(old) + '\n*** Add File: ' + str(old) + '\n' + ''.join('+' + line + '\n' for line in text.splitlines()) + '*** End Patch\n'
subprocess.run(['apply_patch', patch], check=True)
manifest = []
for path in sorted(root.rglob('*')):
    if path.is_file() and path.name != 'SHA256SUMS' and not path.is_relative_to(root / 'port/site') and not path.is_relative_to(root / 'diagnostic/port/site') and not path.is_relative_to(root / 'black-cache'):
        manifest.append(f'{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root)}')
(root / 'SHA256SUMS').write_text('\n'.join(manifest) + '\n')
print(status)
