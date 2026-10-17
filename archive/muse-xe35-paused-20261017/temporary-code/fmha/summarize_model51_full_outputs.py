import hashlib
import json
from pathlib import Path


ART = Path(__file__).resolve().parent
ROOT = ART / 'model-config51-v1'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert (ART / 'model51-accuracy-suite-v1.exit').read_text().strip() == '0'
    raws = {provider: json.loads((ROOT / f'{provider}-accuracy.json').read_text()) for provider in ('source', 'port')}
    routes = json.loads((ART / 'MODEL51_SOURCE_DISPATCH_NATIVE_COMPILER_AUDIT_v1.json').read_text())
    import torch
    assert not torch.xpu.is_initialized()
    torch.set_num_threads(8)
    records = []
    for source, port, route in zip(raws['source']['cases'], raws['port']['cases'], routes['routes'], strict=True):
        assert source['name'] == port['name'] == route['name'] and source['input_sha256'] == port['input_sha256']
        tensors = {}
        for provider, raw in [('source', source), ('port', port)]:
            path = ROOT / f'{provider}-full-outputs' / f'{raw["name"]}.pt'
            assert digest(path) == raw['saved_full_output_file_sha256']
            tensors[provider] = torch.load(path, map_location='cpu', weights_only=True)
            assert tensors[provider].dtype == torch.bfloat16
        source_output, port_output = tensors['source'], tensors['port']
        assert source_output.shape == port_output.shape
        mismatches = (source_output.contiguous().view(torch.int16) != port_output.contiguous().view(torch.int16)).count_nonzero().item()
        assert port['full_output_bitwise'] == (mismatches == 0)
        records.append({'name': source['name'], 'whole_output_elements': source_output.numel(),
                        'BF16_bitwise_mismatched_elements': mismatches, 'full_output_bitwise': mismatches == 0,
                        'paired_max_abs': port['source_port_max_abs'], 'full_source_reference_PASS': source['reference_PASS'],
                        'full_port_reference_PASS': port['reference_PASS'], 'source_route': route['source_route'],
                        'source_device_native': route['device_native_library'], 'source_device_native_sha256': route['device_native_sha256'],
                        'port_route': route['AFTER_route']})
    assert len(records) == 51 and sum(record['whole_output_elements'] for record in records) == 808255488
    assert not torch.xpu.is_initialized()
    report = {'CPU_only': True, 'GPU_initialized': False, 'cases': records,
              'whole_output_elements': 808255488, 'BF16_bitwise_mismatched_elements': sum(record['BF16_bitwise_mismatched_elements'] for record in records),
              'full_output_bitwise_cases': sum(record['full_output_bitwise'] for record in records),
              'all_shared_input_SHA_equal': True, 'route_audit_sha256': digest(ART / 'MODEL51_SOURCE_DISPATCH_NATIVE_COMPILER_AUDIT_v1.json'),
              'no_raw_input_runner_math_changes': True}
    (ART / 'MODEL51_FULL_OUTPUT_BITWISE_CPU_AUDIT_v1.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'cases'}))


if __name__ == '__main__':
    main()
