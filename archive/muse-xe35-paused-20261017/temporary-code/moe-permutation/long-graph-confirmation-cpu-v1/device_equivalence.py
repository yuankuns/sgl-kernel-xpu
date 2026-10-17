import ctypes
import ctypes.util
import hashlib
import json
import struct
import subprocess
from pathlib import Path


FOLDER = Path(__file__).resolve().parent
ROOT = FOLDER.parent


def checksum(data):
    return hashlib.sha256(data).hexdigest()


def elf_sections(data):
    if data[:6] != b'\x7fELF\x02\x01':
        raise RuntimeError('Expected genuine little-endian ELF64')
    offset = struct.unpack_from('<Q', data, 40)[0]
    entry_size, count, names_index = struct.unpack_from('<HHH', data, 58)
    headers = [struct.unpack_from('<IIQQQQIIQQ', data, offset + index * entry_size) for index in range(count)]
    names_header = headers[names_index]
    names = data[names_header[4]:names_header[4] + names_header[5]]
    result = {}
    for header in headers:
        name = names[header[0]:].split(b'\0', 1)[0].decode()
        if header[1] != 8:
            result[name] = data[header[4]:header[4] + header[5]]
    return result


def decompress(data):
    library = ctypes.CDLL(ctypes.util.find_library('zstd'))
    library.ZSTD_getFrameContentSize.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    library.ZSTD_getFrameContentSize.restype = ctypes.c_ulonglong
    library.ZSTD_decompress.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
    library.ZSTD_decompress.restype = ctypes.c_size_t
    library.ZSTD_isError.argtypes = [ctypes.c_size_t]
    library.ZSTD_isError.restype = ctypes.c_uint
    buffer = ctypes.create_string_buffer(data)
    size = library.ZSTD_getFrameContentSize(buffer, len(data))
    if size >= 1024 * 1024 * 1024:
        raise RuntimeError('Unexpected compressed native image size')
    output = ctypes.create_string_buffer(size)
    actual = library.ZSTD_decompress(output, size, buffer, len(data))
    if library.ZSTD_isError(actual) or actual != size:
        raise RuntimeError('Actual native image decompression failed')
    return output.raw


def image_record(label, path):
    data = path.read_bytes()
    bundled = elf_sections(data)['__CLANG_OFFLOAD_BUNDLE__sycl-spir64_gen']
    image = decompress(bundled)
    sections = elf_sections(image)
    (FOLDER / (label + '-device.elf')).write_bytes(image)
    bodies = {}
    for name, body in sections.items():
        if not name.startswith('.text.'):
            continue
        symbol = name.removeprefix('.text.')
        demangled = subprocess.run(['c++filt', '-t', symbol], capture_output=True, text=True, check=True).stdout.strip()
        normalized = demangled.replace('_xe35', '')
        if normalized in bodies:
            raise RuntimeError('Duplicate semantic kernel body')
        bodies[normalized] = {'section': name, 'demangled': demangled, 'size': len(body), 'sha256': checksum(body)}
    return {'path': str(path), 'native_sha256': checksum(data), 'compressed_image_sha256': checksum(bundled),
            'device_image_sha256': checksum(image), 'sections': list(sections), 'bodies': bodies,
            'ze_info': sections.get('.ze_info', b'').decode()}


def main():
    source = image_record('SOURCE', ROOT / 'fresh-source-icpx-v1/site/sgl_kernel/libsgl-ops-sycl-Mxfp4SmallPermutation.so')
    after = image_record('AFTER', ROOT / 'canonical-v4-icpx/site/sgl_kernel/libsgl-ops-sycl-mxfp4_small_permutation.so')
    equal = bool(source['bodies']) and source['bodies'].keys() == after['bodies'].keys()
    checks = []
    for name, body in source['bodies'].items():
        other = after['bodies'].get(name)
        identical = other is not None and body['sha256'] == other['sha256'] and body['size'] == other['size']
        checks.append({'semantic_name': name, 'byte_identical': identical, 'SOURCE': body, 'AFTER': other})
        equal = equal and identical
    record = {'SOURCE': source, 'AFTER': after, 'actual_embedded_device_text_checks': checks,
              'embedded_device_machine_bodies_byte_identical': equal, 'GPU_started': False, 'build_started': False,
              'limitation': 'Native image/container/kernel names and host symbols can differ. Body equality alone is not a runtime timing equivalence claim.'}
    (FOLDER / 'device-equivalence.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'device_bodies': len(checks), 'byte_identical': equal, 'GPU_started': False}), flush=True)


if __name__ == '__main__':
    main()
