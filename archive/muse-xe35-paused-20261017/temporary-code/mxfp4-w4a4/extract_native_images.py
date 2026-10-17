import ctypes
import ctypes.util
import hashlib
import json
import pathlib
import struct


def offload_section(path):
    data = path.read_bytes()
    offset = struct.unpack_from('<Q', data, 40)[0]
    entry_size, count, string_index = struct.unpack_from('<HHH', data, 58)
    headers = [struct.unpack_from('<IIQQQQIIQQ', data, offset + entry_size * index) for index in range(count)]
    strings = data[headers[string_index][4]:headers[string_index][4] + headers[string_index][5]]
    for header in headers:
        name = strings[header[0]:].split(b'\0', 1)[0].decode()
        if 'CLANG_OFFLOAD' in name:
            return data[header[4]:header[4] + header[5]]
    raise RuntimeError('Missing offload section: ' + str(path))


def main():
    artifact = pathlib.Path(__file__).resolve().parent
    libraries = {
        'source': artifact / 'source-venv/lib/python3.12/site-packages/sgl_kernel/libsgl-ops-sycl-moe_grouped_mm_mxfp4_w4a4.so',
        'port': artifact / 'diagnostic-native/libmuse-port-mxfp4-w4a4-moe_grouped_mm_mxfp4_w4a4.so',
        'fresh': artifact / 'fresh-source-build/src/libsgl-ops-sycl-moe_grouped_mm_mxfp4_w4a4.so',
    }
    library = ctypes.CDLL(ctypes.util.find_library('zstd'))
    library.ZSTD_getFrameContentSize.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    library.ZSTD_getFrameContentSize.restype = ctypes.c_uint64
    library.ZSTD_findFrameCompressedSize.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    library.ZSTD_findFrameCompressedSize.restype = ctypes.c_size_t
    library.ZSTD_decompress.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
    library.ZSTD_decompress.restype = ctypes.c_size_t
    report = {'kind': 'CPU-only extraction, no GPU initialization; source-to-port precision remains failed', 'libraries': {}}
    blobs = {}
    for implementation, path in libraries.items():
        blob = offload_section(path)
        blobs[implementation] = blob
        record = {'path': str(path), 'native_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'offload_sha256': hashlib.sha256(blob).hexdigest(), 'images': []}
        offset = 0
        while True:
            offset = blob.find(b'\x28\xb5\x2f\xfd', offset)
            if offset < 0:
                break
            compressed = ctypes.create_string_buffer(blob[offset:])
            size = library.ZSTD_getFrameContentSize(compressed, len(blob) - offset)
            if size > 256 * 1024 * 1024:
                raise RuntimeError('Unexpected decompressed image size')
            frame_size = library.ZSTD_findFrameCompressedSize(compressed, len(blob) - offset)
            output = ctypes.create_string_buffer(size)
            actual = library.ZSTD_decompress(output, size, compressed, frame_size)
            if actual != size:
                raise RuntimeError('Zstd image decompression failed')
            contents = output.raw
            filename = artifact / f'diagnostic-native-image-{implementation}-{len(record["images"])}.bin'
            filename.write_bytes(contents)
            record['images'].append({'file': str(filename), 'sha256': hashlib.sha256(contents).hexdigest(), 'size': size, 'magic_hex': contents[:16].hex(), 'elf': contents.startswith(b'\x7fELF')})
            offset += frame_size
        report['libraries'][implementation] = record
    report['historical_source_equals_fresh_source_offload_section'] = blobs['source'] == blobs['fresh']
    (artifact / 'diagnostic-native-images.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
