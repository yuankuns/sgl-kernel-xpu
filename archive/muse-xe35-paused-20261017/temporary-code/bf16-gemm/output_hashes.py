import argparse
import hashlib
import json
from pathlib import Path


def tensor_digest(tensor):
    tensor = tensor.contiguous()
    checksum = hashlib.sha256()
    checksum.update(str(tensor.dtype).encode())
    checksum.update(json.dumps(list(tensor.shape)).encode())
    checksum.update(memoryview(tensor.view(torch.uint8).numpy()))
    return checksum.hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument('round')
args = parser.parse_args()
import torch

root = Path(__file__).resolve().parent
port_label = 'diagnostic-port' if args.round.startswith('kernel-diagnostic') else 'port'
labels = ['source', port_label, port_label, 'source']
records = []
for index, label in enumerate(labels):
    metadata = json.loads((root / f'{args.round}-{index}-{label}.json').read_text())
    cases = []
    for case in metadata['cases']:
        outputs = torch.load(root / case['outputs'], map_location='cpu', weights_only=True)
        items = outputs.items() if isinstance(outputs, dict) else enumerate(outputs)
        cases.append({'case': case.get('case', f'M8-N{case.get("n")}-K8192'),
                      'tensor_sha256': {str(name): tensor_digest(tensor) for name, tensor in items}})
    records.append({'run': metadata['run'], 'gpu_mask': metadata['gpu_mask'], 'cases': cases})
identical = all(record['cases'] == records[0]['cases'] for record in records[1:])
if root.name == 'moe-permutation':
    assert identical, 'MoE output tensor bytes must be bitwise identical across BAAB'
result = {'round': args.round, 'verification_kind': 'tensor-byte checksum supplement, not a separate accuracy test',
          'hash_format': 'SHA256(dtype UTF8 + JSON shape UTF8 + contiguous raw tensor bytes)',
          'archive_file_checksums_are_separate': True, 'all_output_tensor_bytes_identical': identical,
          'runs': records}
(root / (args.round + '-tensor-hashes.json')).write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'round': args.round, 'all_output_tensor_bytes_identical': identical}))
