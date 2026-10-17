import collections
import ctypes
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

sys.dont_write_bytecode = True
RUN = Path(__file__).resolve().parent
ARTIFACT = RUN.parent
sys.path.insert(0, str(ARTIFACT / 'normal-v4-model-config-perf-v1'))
from run_perf import digest, local_path, verify


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def sections(raw):
    assert raw[:6] == b'\x7fELF\x02\x01'
    header = struct.unpack_from('<16sHHIQQQIHHHHHH', raw)
    entries = [struct.unpack_from('<IIQQQQIIQQ', raw, header[6] + index * header[11]) for index in range(header[12])]
    names = entries[header[13]]
    strings = raw[names[4]:names[4] + names[5]]
    result = {}
    for entry in entries:
        name = strings[entry[0]:].split(b'\0', 1)[0].decode()
        result[name] = {'offset': entry[4], 'size': entry[5], 'raw': raw[entry[4]:entry[4] + entry[5]]}
    return header[2], result


def decompress_frames(payload):
    library = ctypes.CDLL('libzstd.so.1')
    library.ZSTD_findFrameCompressedSize.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    library.ZSTD_findFrameCompressedSize.restype = ctypes.c_size_t
    library.ZSTD_getFrameContentSize.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    library.ZSTD_getFrameContentSize.restype = ctypes.c_uint64
    library.ZSTD_decompress.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
    library.ZSTD_decompress.restype = ctypes.c_size_t
    library.ZSTD_isError.argtypes = [ctypes.c_size_t]
    library.ZSTD_isError.restype = ctypes.c_uint
    offset = 0
    images = []
    while offset < len(payload):
        if payload[offset] == 0:
            offset += 1
            continue
        assert payload[offset:offset + 4] == b'\x28\xb5\x2f\xfd', f'Unrecognized read-only payload offset{offset}'
        source = ctypes.create_string_buffer(payload[offset:])
        size = library.ZSTD_findFrameCompressedSize(source, len(payload) - offset)
        assert not library.ZSTD_isError(size)
        expanded = library.ZSTD_getFrameContentSize(source, size)
        assert 0 < expanded <= 16 * 1024 * 1024, 'Bounded CPU extraction only'
        destination = ctypes.create_string_buffer(expanded)
        actual = library.ZSTD_decompress(destination, expanded, source, size)
        assert not library.ZSTD_isError(actual) and actual == expanded
        images.append({'offset': offset, 'compressed_size': size, 'raw': destination.raw})
        offset += size
    return images


def extract(role, path):
    before = digest(path)
    directory = RUN / role
    directory.mkdir(exist_ok=False)
    machine, table = sections(path.read_bytes())
    summary = {'native_path': str(path), 'native_sha256': before, 'host_ELF_machine': machine,
        'sections': {name: {'size': item['size'], 'sha256': sha(item['raw'])} for name, item in table.items() if name.startswith('__CLANG_OFFLOAD') or name in ('.text', '.tgtimg', '.tgtsym', '.rodata')}, 'images': []}
    for mode, arguments in (('sections', ['readelf', '-SW', str(path)]), ('dynamic', ['readelf', '-dW', str(path)]),
        ('exports', ['nm', '-D', '-C', str(path)])):
        output = subprocess.run(arguments, capture_output=True, text=True)
        (directory / f'{mode}.log').write_text(output.stdout + output.stderr)
        (directory / f'{mode}.exit').write_text(f'{output.returncode}\n')
        (directory / f'{mode}.command.json').write_text(json.dumps(arguments) + '\n')
        assert output.returncode == 0
    for name, item in table.items():
        if not name.startswith('__CLANG_OFFLOAD_BUNDLE__sycl-spir64_gen'):
            continue
        for index, frame in enumerate(decompress_frames(item['raw'])):
            image = frame['raw']
            device_machine, device_sections = sections(image)
            target = directory / f'image-{index:02d}'
            target.mkdir(exist_ok=False)
            row = {'index': index, 'payload_offset': frame['offset'], 'compressed_size': frame['compressed_size'],
                'device_ELF_machine': device_machine, 'decompressed_image_sha256': sha(image), 'kernels': []}
            for section, content in device_sections.items():
                if section == '.ze_info':
                    (target / 'ze_info.yaml').write_bytes(content['raw'])
                    row['ze_info_sha256'] = sha(content['raw'])
                if section.startswith('.text.'):
                    kernel_index = len(row['kernels'])
                    output = target / f'kernel-{kernel_index:03d}.isa.bin'
                    output.write_bytes(content['raw'])
                    row['kernels'].append({'section_name': section, 'bytes': content['size'],
                        'ISA_sha256': sha(content['raw']), 'raw_extracted_path': str(output)})
            assert row['kernels'] and 'ze_info_sha256' in row
            summary['images'].append(row)
    assert summary['images'] and digest(path) == before
    (directory / 'metadata.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main():
    assert set(os.sched_getaffinity(0)) == set(range(32, 48))
    value = verify()
    source = local_path(value['source_provider']['source_site']) / 'sgl_kernel/libsgl-ops-sycl-FusedQKNormRope.so'
    port = local_path(value['port_site']) / 'sgl_kernel/libsgl-ops-sycl-FusedQKNormRope_xe35.so'
    assert digest(source) == value['source_provider']['source_qk_library_sha256']
    assert digest(port) == value['port_native_SHA'][port.name]
    results = {role: extract(role, path) for role, path in (('source', source), ('port', port))}
    counters = {role: collections.Counter(kernel['ISA_sha256'] for image in item['images'] for kernel in image['kernels']) for role, item in results.items()}
    matching = counters['source'] & counters['port']
    output = {'actual_exit': 0, 'CPU_affinity': sorted(os.sched_getaffinity(0)),
        'read_only_existing_native_ELF_zstd_AOT_metadata_and_instruction_bytes': True,
        'GPU_initialized_or_run': False, 'compiler_or_build_launched': False,
        'SOURCE_native_sha256': digest(source), 'PORT_native_sha256': digest(port),
        'image_counts': {role: len(item['images']) for role, item in results.items()},
        'kernel_instruction_section_counts': {role: sum(counter.values()) for role, counter in counters.items()},
        'exact_instruction_SHA_multiset_equal': counters['source'] == counters['port'],
        'exact_matching_instruction_sections': sum(matching.values()),
        'SOURCE_only_instruction_SHA': dict(counters['source'] - counters['port']),
        'PORT_only_instruction_SHA': dict(counters['port'] - counters['source']),
        'kernel_identity_and_metadata_mapping_proven': False,
        'same_ISA_does_not_prove_same_host_enqueue_grid_runtime_or_performance': True}
    (RUN / 'extraction-summary.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
