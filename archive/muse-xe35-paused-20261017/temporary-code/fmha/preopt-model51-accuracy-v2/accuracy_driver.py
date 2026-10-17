import json
import time
import traceback
from types import SimpleNamespace
from contracts import ROOT, ART, WORKLOAD, BASE, arguments, grant_guard, provider, observe_registered, digest, write_json


def main():
    args = arguments()
    if args.guard_only:
        print('B8/4096/16/TP1EP1/BF16 guard0; no Torch or GPU import')
        return
    if not args.run:
        raise RuntimeError('No runtime authorized by preparation; explicit --run and fresh main grant required')
    grant = grant_guard(args.grant)
    output_dir = ROOT / 'accuracy-run-v1'
    output_dir.mkdir(exist_ok=False)
    result = {'workload': WORKLOAD, 'source_head': BASE, 'phase': grant['phase'], 'grant_sha256': digest(args.grant),
              'cross_algorithm_bitwise_required': False, 'no_timing_warmup_events_graphs': True, 'cases': []}
    try:
        torch, semantic, reference, framework, plan, bundles, proof = provider()
        from sgl_kernel.utils import is_xe3_arch
        assert is_xe3_arch()
        result['provider'] = proof
        result['plan_sha256'] = digest(ART / 'model-config51-v1/plan.json')
        torch.set_num_threads(8)
        torch.set_float32_matmul_precision('highest')
        cpu = json.loads((ROOT / 'CPU_PREPARATION.json').read_text())
        with torch.inference_mode():
            for case_index, case in enumerate(plan['cases']):
                torch.xpu.synchronize()
                time.sleep(2)
                tensors = semantic.fixture(case, bundles[case['bundle']], torch, 'xpu')
                record = {'name': case['name'], 'input_sha256': [reference.tensor_digest(tensor, torch) for tensor in tensors], 'registered_calls': [], 'accuracy': 'PENDING'}
                result['cases'].append(record)
                assert record['input_sha256'] == cpu['cases'][case_index]['input_sha256']
                observer, boundary_stop = observe_registered(torch, record['registered_calls'])
                with observer:
                    output = semantic.backend_call(case, tensors, torch, framework)
                torch.xpu.synchronize()
                actual = output.cpu().contiguous()
                assert actual.dtype == torch.bfloat16 and list(actual.shape) == case['output_shape']
                assert len(record['registered_calls']) == 1 and record['registered_calls'][0]['native_operator_completed']
                output_file = output_dir / (case['name'] + '.pt')
                torch.save(actual, output_file)
                record.update({'output_file': str(output_file.relative_to(ROOT)), 'saved_full_output_file_sha256': digest(output_file),
                               'output_tensor_sha256': reference.tensor_digest(actual, torch), 'production_dtype': str(actual.dtype), 'output_elements': actual.numel()})
                write_json(output_dir / 'preopt-model51-accuracy.json', result)
                record.update(reference.full_cpu_reference(SimpleNamespace(**case), (*tensors, output), torch))
                assert record['reference_PASS']
                assert record['checked_elements'] == case['requests'] * case['query_length'] * 128 * 64
                record['accuracy'] = 'PASS'
                write_json(output_dir / 'preopt-model51-accuracy.json', result)
                print(json.dumps(record), flush=True)
                del tensors, output, actual
        assert len(result['cases']) == 51 and sum(case['checked_elements'] for case in result['cases']) == 808255488
        result['all51_full_reference_PASS'] = True
        result['actual_exit'] = 0
        write_json(output_dir / 'preopt-model51-accuracy.json', result)
        (output_dir / 'preopt-model51-accuracy.exit').write_text('0\n')
    except Exception:
        if result['cases']:
            result['cases'][-1]['accuracy'] = 'FAIL'
        result['first_failure_traceback'] = traceback.format_exc()
        result['all51_full_reference_PASS'] = False
        result['actual_exit'] = 1
        write_json(output_dir / 'preopt-model51-accuracy.json', result)
        (output_dir / 'preopt-model51-accuracy.exit').write_text('1\n')
        raise


if __name__ == '__main__':
    main()
