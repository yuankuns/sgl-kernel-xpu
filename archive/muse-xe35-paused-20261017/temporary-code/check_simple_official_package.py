import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('component', choices=('memory', 'position', 'logits-vocab'))
    parser.add_argument('label', choices=('before', 'after'))
    args = parser.parse_args()
    artifact = ROOT / args.component
    stage = artifact / args.label / 'site'
    manifest_name = 'source-binary-manifest.json' if args.label == 'after' else 'baseline-source-binary-manifest.json'
    manifest = json.loads((artifact / manifest_name).read_text())
    expected = manifest['stage']
    import sgl_kernel
    import torch

    if Path(sgl_kernel.__file__).resolve() != (stage / 'sgl_kernel/__init__.py').resolve():
        raise RuntimeError('Wrong normal package import')
    names = {
        'memory': ('paged_alloc_extend', 'paged_alloc_extend_and_write', 'paged_alloc_decode', 'write_req_to_token_pool'),
        'position': ('compute_position', 'compute_position_with_hint'),
        'logits-vocab': ('softcap_logits', 'vocab_mask', 'vocab_parallel_embedding'),
    }[args.component]
    schemas = {name + '_xe35': str(getattr(torch.ops.sgl_kernel, name + '_xe35').default._schema) for name in names}
    mapped = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if 'libsgl-ops-sycl-' in line or '/common_ops' in line})
    if not any('/common_ops' in filename for filename in mapped):
        raise RuntimeError('No real official registration extension mapped')
    libraries = {}
    for filename in mapped:
        path = Path(filename).resolve()
        try:
            relative = str(path.relative_to(stage.resolve()))
        except ValueError as error:
            raise RuntimeError(f'Native library loaded outside official stage: {path}') from error
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected.get(relative) != digest:
            raise RuntimeError(f'Mapped binary does not match manifest: {path}')
        libraries[relative] = digest
    print(json.dumps({'component': args.component, 'label': args.label, 'package': sgl_kernel.__file__, 'torch': torch.__version__, 'schemas': schemas, 'mapped_libraries': libraries, 'xpu_initialized': torch.xpu.is_initialized()}))


if __name__ == '__main__':
    main()
