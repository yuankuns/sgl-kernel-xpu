import argparse
from collections import Counter
import json
import os
from pathlib import Path
import runpy
import statistics
import sys
import time
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parent
ART = ROOT.parent
sys.path.insert(0, str(ART / 'v5-model51-gpu10-preparation'))
import precision_v5 as precision


PHASES = {'source-port': 'fmha-v5-gpu5-source-port-all89-three-BAAB',
          'preopt-source': 'fmha-v5-gpu5-true-preopt-source-all89-three-BAAB'}
SLOTS = {'B0': 'after', 'A1': 'before', 'A2': 'before', 'B3': 'after'}
ROLES = {'source-port': {'before': 'source', 'after': 'port'},
         'preopt-source': {'before': 'preopt', 'after': 'source'}}


def grant_guard(args):
    precision.scope_guard()
    if args.grant is None or not args.grant.is_file():
        raise RuntimeError('fresh main GPU5 performance grant required; this driver never issues or launches a grant')
    grant = json.loads(args.grant.read_text())
    assert grant['phase'] == PHASES[args.comparison] and grant['gpu'] == 5
    assert grant['coordinator_release'] and not grant['closed'] and not grant['runtime_reuse_permitted']
    assert grant['workload'] == precision.WORKLOAD and grant['runtime_cpus'] == '16-31'
    assert grant['epoch'] >= grant['issued_epoch'] + 120 and time.time() >= grant['epoch']
    assert grant['source_digest'] == precision.EXPECTED_DIGEST and grant['canonical_version'] == precision.VERSION
    assert grant['driver_sha256'] == precision.digest(Path(__file__)) and grant['rounds'] == 3
    assert grant['slot_order'] == ['B0', 'A1', 'A2', 'B3'] and grant['cases_per_provider_per_slot'] == 89
    assert grant['warmup'] == 5 and grant['iterations'] == 21 and grant['graph_replays'] == 8
    assert grant['finalseal_sha256'] == precision.digest(ROOT / 'GPU10_V5_PRECISION_FINAL_SEAL.json')
    assert grant['baseline_audit_sha256'] == precision.digest(ROOT / 'ORIGINAL_SOURCE_PREOPT_CURRENT_NATIVE_CPU_AUDIT.json')
    assert grant['model51_plan_sha256'] == precision.digest(ART / 'model-config51-v1/plan.json')
    assert grant['compiler_policy_sha256'] == precision.digest(ART / precision.VERSION / 'compiler-policy-verified.json')
    assert grant['normal_CPU_gate_sha256'] == precision.digest(ART / 'v5-model51-gpu10-preparation/normal-v5-CPU-gate-v2.json')
    for role in ('source', 'preopt') if args.comparison == 'preopt-source' else ('source',):
        assert grant[f'{role}_package_manifest_sha256'] == precision.digest(ART / 'matched-native-icpx-v1' / role / 'normal-package-manifest.json')
    assert grant['NV_handoff_authorized'] is False and grant['main_stable_pool_quiet_release'] is True
    assert args.role == ROLES[args.comparison][SLOTS[args.slot]] and args.round in (1, 2, 3)
    if args.comparison == 'preopt-source':
        preopt_gate = ROOT / 'preopt-model51-accuracy.json'
        assert grant['preopt51_accuracy_sha256'] == precision.digest(preopt_gate)
        assert (ROOT / 'preopt-model51-accuracy.exit').read_text().strip() == '0'
        proof = json.loads(preopt_gate.read_text())
        assert proof['all51_full_reference_PASS'] and len(proof['cases']) == 51
        assert proof['source_head'] == 'ab3a26c43a38e6aea7d73cc1349e4b5e210b772b'
    return grant


def observe_capture(torch, role, precision_module, guard_module, entry_text, forward_rows, operator_counts):
    from torch.utils._python_dispatch import TorchDispatchMode
    port_observer = precision_module.native_route_observer(torch, guard_module, entry_text, forward_rows) if role == 'port' else None

    class ObserveRealOperators(TorchDispatchMode):
        def __torch_dispatch__(self, function, types, args=(), kwargs=None):
            keywords = {} if kwargs is None else kwargs
            operator_counts[str(function)] += 1
            if function == torch.ops.sgl_kernel.fwd.default and role != 'port':
                arguments = {item.name: item.default_value for item in function._schema.arguments}
                arguments.update(dict(zip((item.name for item in function._schema.arguments), args)))
                arguments.update(keywords)
                query, key, table = arguments['q'], arguments['k'], arguments['page_table']
                paged64bf16 = table is not None and table.dim() == 2 and key.dim() == 4 and key.shape[1:] == (64, 16, 64)
                paged64bf16 = paged64bf16 and query.dim() == 3 and query.shape[1:] == (128, 64) and query.dtype == key.dtype == torch.bfloat16
                assert paged64bf16
                phase = 'decode' if arguments['max_seqlen_q'] == 1 else 'prefill'
                row = {'operator': str(function), 'original_provider': role, 'phase': phase,
                    'actual_maxK': arguments['max_seqlen_k'], 'Q_shape': list(query.shape),
                    'page_table_shape': list(table.shape), 'paged64BF16_actual_metadata': True,
                    'native_operator_completed': False,
                    'boundary': 'original root has no generic branch; original source/ELF dispatch plus genuine current mapped SHA, not a hardware kernel trace'}
                forward_rows.append(row)
                result = function(*args, **keywords)
                row['native_operator_completed'] = True
                return result
            return function(*args, **keywords)

    return ObserveRealOperators(), port_observer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--comparison', choices=PHASES, default='source-port')
    parser.add_argument('--role', choices=('source', 'port', 'preopt'), required=True)
    parser.add_argument('--round', type=int, default=1)
    parser.add_argument('--slot', choices=SLOTS, default='B0')
    parser.add_argument('--grant', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    grant = grant_guard(args)
    assert not args.output.exists()
    assert os.environ.get('ZE_AFFINITY_MASK') == '5' and os.sched_getaffinity(0) == set(range(16, 32))
    original, guards, canonical_preflight, local_path, verify_loaded, _, tensor_digest = precision.shared_imports()
    manifest, manifest_path, port_package, required_port = canonical_preflight()
    assert precision.digest(manifest_path) == grant['canonical_manifest_sha256']
    assert manifest['source']['source_digest'] == precision.EXPECTED_DIGEST
    from matched_baseline_provenance import package_preflight, provenance
    if args.role == 'port':
        package = port_package
        hashes = {name: record['sha256'] for name, record in manifest['binaries'].items()}
        required = set(hashes)
        compiler = {'whole_normal_package_icpx': True, 'compiler_policy_sha256': precision.digest(ART / precision.VERSION / 'compiler-policy-verified.json')}
        binding = precision.verify_native_binding(package)
    else:
        package, hashes = package_preflight(args.role)
        compiler = provenance(args.role)
        imported = json.loads((ART / 'matched-native-icpx-v1' / args.role / 'normal-import-verified.json').read_text())
        required = {Path(name).name for name in imported['loaded_normal_package_libraries_sha256']}
        binding = {'original_three_math_native': compiler['native_kernels'], 'whole_package_icpx': False}
        expected_head = '08955d90fad52a40779705f259569db93e570a22' if args.role == 'source' else 'ab3a26c43a38e6aea7d73cc1349e4b5e210b772b'
        assert compiler['source_head'] == expected_head
        assert compiler['package_manifest_sha256'] == grant[f'{args.role}_package_manifest_sha256']
    import torch
    import sgl_kernel
    assert Path(sgl_kernel.__file__).resolve().parent == package.resolve()
    verify_loaded(package, hashes, required)
    from sgl_kernel.flash_attn import flash_attn_with_kvcache
    from sgl_kernel.utils import is_xe3_arch
    assert is_xe3_arch()
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision('highest')
    framework = original.import_framework(torch, False)
    assert framework[0].forward_extend.__globals__['flash_attn_with_kvcache'] is flash_attn_with_kvcache
    assert framework[0].forward_decode.__globals__['flash_attn_with_kvcache'] is flash_attn_with_kvcache
    plan = original.verify_plan()
    bundles = {name: torch.load(original.ROOT / name, map_location='cpu', weights_only=True) for name in plan['fixture_files']}
    candidate = json.loads((ART / 'v5-metadata-bridge-preparation/candidate-source-freeze-v5.json').read_text())
    entry = local_path(candidate['destination']) / 'src/sycl/xe35/fmha/entry.cpp'
    assert precision.digest(entry) == manifest['source']['files']['src/sycl/xe35/fmha/entry.cpp']['sha256']
    entry_text = entry.read_text()
    bench = runpy.run_path(str(local_path('/home/syk/worktrees/sgl-kernel-xpu-up/pr/muse-fmha-xe35/benchmark/bench_muse_fmha.py')))
    _, representatives = bench['arguments_xe35'](['--guard-only', '--mode', 'bench'])
    assert len(representatives) == 38
    source_model_path = original.ROOT / 'source-accuracy.json'
    source_rep_path = ART / 'source-only-v1-fresh-optimized.json'
    if args.role == 'preopt':
        source_model_path = ROOT / 'preopt-model51-accuracy.json'
        source_rep_path = ART / 'source-only-v1-fresh-preopt.json'
    golden = {family: {case['name']: case for case in json.loads(path.read_text())['cases']}
              for family, path in [('model51', source_model_path), ('representative38', source_rep_path)]}
    work = [('model51', SimpleNamespace(**case), case) for case in plan['cases']]
    work += [('representative38', case, None) for case in representatives]
    assert len(work) == 89
    result = {'comparison': args.comparison, 'role': args.role, 'round': args.round, 'slot': args.slot,
        'workload': precision.WORKLOAD, 'canonical_manifest_sha256': precision.digest(manifest_path),
        'source_digest': precision.EXPECTED_DIGEST, 'driver_sha256': precision.digest(Path(__file__)),
        'grant_sha256': precision.digest(args.grant), 'normal_package': str(package),
        'finalseal_sha256': grant['finalseal_sha256'], 'baseline_audit_sha256': grant['baseline_audit_sha256'],
        'model51_plan_sha256': grant['model51_plan_sha256'], 'compiler_policy_sha256': grant['compiler_policy_sha256'],
        'compiler_scope_and_confound': compiler, 'native_binding': binding,
        'timing': 'XPU device events around 8-launch graph replay, 21 raw samples; no host wall-time',
        'semantic51_timing_scope': 'same original accuracy adapter including synthetic active-KV gather; inclusive adapter graph, not isolated FMHA-only or complete model',
        'cases': []}
    output_dir = args.output.parent / (args.output.stem + '-full-outputs')
    output_dir.mkdir(exist_ok=False)
    graph_dir = args.output.parent / (args.output.stem + '-graphs')
    graph_dir.mkdir(exist_ok=False)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    with torch.inference_mode():
        for family, case, model in work:
            torch.xpu.synchronize()
            time.sleep(2)
            holder = {}
            if family == 'model51':
                tensors = original.fixture(model, bundles[model['bundle']], torch, 'xpu')
                input_hashes = [tensor_digest(tensor, torch) for tensor in tensors]

                def launch():
                    holder['output'] = original.backend_call(model, tensors, torch, framework)
            else:
                prepared = bench['tensors_xe35'](case, torch)
                tensors, holder['output'] = prepared[:-1], prepared[-1]
                input_hashes = [tensor_digest(tensor, torch) for tensor in tensors]

                def launch():
                    bench['launch_xe35'](case, prepared, flash_attn_with_kvcache)
            baseline = golden[family][case.name]
            assert input_hashes == baseline['input_sha256']

            def check_full_output():
                actual = holder['output'].detach().cpu().contiguous()
                assert tensor_digest(actual, torch) == baseline['output_sha256'], 'whole output differs from full-reference-verified own golden; stop, no timing acceptance'
                return actual

            launch()
            torch.xpu.synchronize()
            check_full_output()
            for _ in range(5):
                launch()
            torch.xpu.synchronize()
            forward_rows, operators = [], Counter()
            observer, port_observer = observe_capture(torch, args.role, precision, guards, entry_text, forward_rows, operators)
            graph = torch.xpu.XPUGraph()
            graph.enable_debug_mode()
            with observer:
                if port_observer is not None:
                    with port_observer:
                        with torch.xpu.graph(graph):
                            for _ in range(8):
                                launch()
                else:
                    with torch.xpu.graph(graph):
                        for _ in range(8):
                            launch()
            torch.xpu.synchronize()
            graph_path = graph_dir / f'{family}-{case.name}.dot'
            assert graph_path.suffix == '.dot'
            graph.debug_dump(str(graph_path))
            assert graph_path.is_file() and graph_path.stat().st_size > 0
            assert len(forward_rows) == 8 and all(row['native_operator_completed'] for row in forward_rows)
            if args.role == 'port':
                assert all(row['guard_evaluation_from_actual_metadata']['optimized'] for row in forward_rows)
                if family == 'model51':
                    assert all(row['actual_fwd_maxK_before_private_resolver'] == 0 for row in forward_rows)
            record = {'family': family, 'name': case.name, 'input_sha256': input_hashes,
                'capture_actual_registered_fwd_nodes': forward_rows,
                'capture_registered_operator_counts': dict(operators),
                'actual_graph_debug_dump': str(graph_path), 'actual_graph_debug_dump_sha256': precision.digest(graph_path),
                'capture_proof_boundary': 'actual XPU graph .dot plus real registered operator capture observations and ELF/mapped binding; not a device-event profiler kernel trace',
                'wholebits_before_timing_PASS': True, 'samples_ms': []}
            result['cases'].append(record)
            args.output.write_text(json.dumps(result, indent=2) + '\n')
            for _ in range(5):
                graph.replay()
            torch.xpu.synchronize()
            check_full_output()
            for _ in range(21):
                start = torch.xpu.Event(enable_timing=True)
                end = torch.xpu.Event(enable_timing=True)
                start.record()
                graph.replay()
                end.record()
                torch.xpu.synchronize()
                record['samples_ms'].append(start.elapsed_time(end) / 8)
            actual = check_full_output()
            path = output_dir / f'{family}-{case.name}.pt'
            torch.save(actual, path)
            record.update({'median_ms': statistics.median(record['samples_ms']),
                'wholebits_after_timing_PASS': True, 'output_sha256': tensor_digest(actual, torch),
                'saved_output_file': str(path),
                'saved_output_file_sha256': precision.digest(path), 'warmup': 5, 'graph_replays': 8, 'iterations': 21})
            args.output.write_text(json.dumps(result, indent=2) + '\n')
            del graph, tensors, holder, observer, port_observer
    verify_loaded(package, hashes, required)
    result['all89_input_SHA_capture_nodes_wholebits_samples_PASS'] = len(result['cases']) == 89
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, AssertionError, FileNotFoundError) as error:
        print(f'performance gate/precision/capture failed: {error}', file=sys.stderr)
        raise SystemExit(2)
