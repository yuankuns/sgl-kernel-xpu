import collections
import ctypes
import hashlib
import json
import os
import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / 'mxfp4-w4a4/aot-regression-main-cpu-v1'
LIBRARY = 'libsgl-ops-sycl-moe_grouped_mm_mxfp4_w4a4.so'


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
        result[name] = raw[entry[4]:entry[4] + entry[5]]
    return header[2], result


def frames(payload):
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
    expanded_total = 0
    while offset < len(payload):
        if payload[offset] == 0:
            offset += 1
            continue
        assert payload[offset:offset + 4] == b'\x28\xb5\x2f\xfd'
        source = ctypes.create_string_buffer(payload[offset:])
        size = library.ZSTD_findFrameCompressedSize(source, len(payload) - offset)
        assert not library.ZSTD_isError(size)
        expanded = library.ZSTD_getFrameContentSize(source, size)
        expanded_total += expanded
        assert 0 < expanded <= 64 * 1024 ** 2 and expanded_total <= 512 * 1024 ** 2
        destination = ctypes.create_string_buffer(expanded)
        actual = library.ZSTD_decompress(destination, expanded, source, size)
        assert not library.ZSTD_isError(actual) and actual == expanded
        yield offset, destination.raw
        offset += size


def extract(role, provider):
    package = Path(provider['package'].replace('/home/syk/', '/root/', 1))
    if package.name == '__init__.py':
        package = package.parent
    path = package / LIBRARY
    native = path.read_bytes()
    expected = provider['selected_native_sha256'][LIBRARY]
    assert sha(native) == expected
    directory = OUTPUT / role
    directory.mkdir()
    machine, table = sections(native)
    records = []
    for section_name, payload in table.items():
        if not section_name.startswith('__CLANG_OFFLOAD_BUNDLE__sycl-spir64_gen'):
            continue
        for offset, image in frames(payload):
            device_machine, device_sections = sections(image)
            image_directory = directory / f'image-{len(records):03d}'
            image_directory.mkdir()
            assert '.ze_info' in device_sections
            metadata = device_sections['.ze_info']
            (image_directory / 'ze_info.yaml').write_bytes(metadata)
            kernels = []
            for name, instructions in device_sections.items():
                if not name.startswith('.text.'):
                    continue
                output = image_directory / f'kernel-{len(kernels):03d}.isa.bin'
                output.write_bytes(instructions)
                kernels.append({'section_name': name, 'ISA_bytes': len(instructions),
                                'ISA_sha256': sha(instructions), 'ISA_file': str(output)})
            assert kernels
            records.append({'offset': offset, 'image_sha256': sha(image), 'device_ELF_machine': device_machine,
                            'ze_info_sha256': sha(metadata), 'ze_info_file': str(image_directory / 'ze_info.yaml'),
                            'kernels': kernels})
    assert records and sha(path.read_bytes()) == expected
    report = {'native': str(path), 'native_sha256': expected, 'host_ELF_machine': machine, 'images': records}
    (directory / 'metadata.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    assert set(os.sched_getaffinity(0)) == set(range(112, 128))
    OUTPUT.mkdir(exist_ok=False)
    audit_path = ROOT / 'qk-norm-rope/w4-regression-independent-cpu-v1/audit.json'
    audit = json.loads(audit_path.read_text())
    assert audit['independent_CPU_audit_actual_exit'] == 0
    results = {role: extract(role, audit['providers'][role]) for role in ('source', 'after')}
    counters = {role: collections.Counter(kernel['ISA_sha256'] for image in result['images'] for kernel in image['kernels'])
                for role, result in results.items()}
    report = {'actual_exit': 0, 'CPU_only': True, 'GPU_or_compiler_started': False,
              'input_provider_audit_sha256': sha(audit_path.read_bytes()),
              'native_sha256': {role: result['native_sha256'] for role, result in results.items()},
              'AOT_images': {role: len(result['images']) for role, result in results.items()},
              'kernels': {role: sum(counter.values()) for role, counter in counters.items()},
              'exact_instruction_SHA_multisets_equal': counters['source'] == counters['after'],
              'exact_matching_instruction_sections': sum((counters['source'] & counters['after']).values()),
              'selected_runtime_kernel_grid_or_performance_root_cause_proven': False}
    (OUTPUT / 'SUMMARY.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
