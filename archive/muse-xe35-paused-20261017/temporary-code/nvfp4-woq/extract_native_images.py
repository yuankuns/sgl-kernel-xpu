import ctypes
import ctypes.util
import hashlib
import json
import pathlib
import struct
import sys


def elf_sections(content):
    assert content[:5] == b'\x7fELF\x02'
    section_offset = struct.unpack_from('<Q', content, 40)[0]
    section_size, section_count, names_index = struct.unpack_from('<HHH', content, 58)
    sections = [struct.unpack_from('<IIQQQQIIQQ', content, section_offset + index * section_size) for index in range(section_count)]
    names_section = sections[names_index]
    names = content[names_section[4]:names_section[4] + names_section[5]]
    return {names[section[0]:].split(b'\0', 1)[0].decode(): section for section in sections}


root = pathlib.Path(__file__).resolve().parent
output = root / (sys.argv[2] if len(sys.argv) > 2 else 'device-images')
output.mkdir(exist_ok=True)
libraries = {
    'source': pathlib.Path('/root' if str(root).startswith('/root/') else '/home/syk') / 'worktrees/sgl-kernel-xpu/muse-spark-nvfp4-woq-main-20261014/build/src/libsgl-ops-sycl-blockwise_moe_nvfp4_woq.so',
    'port': pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else root / 'native-diagnostic/port/site/sgl_kernel/libsgl-ops-sycl-blockwise_moe_nvfp4_woq_port_payload.so',
}
zstd = ctypes.CDLL(ctypes.util.find_library('zstd'))
zstd.ZSTD_findFrameCompressedSize.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
zstd.ZSTD_findFrameCompressedSize.restype = ctypes.c_size_t
zstd.ZSTD_decompress.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t)
zstd.ZSTD_decompress.restype = ctypes.c_size_t
zstd.ZSTD_isError.argtypes = (ctypes.c_size_t,)
zstd.ZSTD_isError.restype = ctypes.c_uint
report = {}
for label, library in libraries.items():
    content = library.read_bytes()
    sections = elf_sections(content)
    image_table = sections['.tgtimg']
    images = []
    for image_index, entry_offset in enumerate(range(image_table[4], image_table[4] + image_table[5], 16)):
        address, decoded_size = struct.unpack_from('<QQ', content, entry_offset)
        enclosing = next(section for section in sections.values() if section[1] == 1 and section[3] <= address < section[3] + section[5])
        file_offset = enclosing[4] + address - enclosing[3]
        compressed = ctypes.create_string_buffer(content[file_offset:enclosing[4] + enclosing[5]])
        compressed_size = zstd.ZSTD_findFrameCompressedSize(compressed, len(compressed) - 1)
        assert not zstd.ZSTD_isError(compressed_size)
        decoded = ctypes.create_string_buffer(decoded_size)
        actual_size = zstd.ZSTD_decompress(decoded, decoded_size, compressed, compressed_size)
        assert not zstd.ZSTD_isError(actual_size) and actual_size == decoded_size
        data = decoded.raw[:actual_size]
        path = output / f'{label}-image{image_index}.bin'
        path.write_bytes(data)
        item = {'image': str(path), 'compressed_size': compressed_size, 'decoded_size': actual_size, 'sha256': hashlib.sha256(data).hexdigest()}
        if data.startswith(b'\x7fELF'):
            native_sections = elf_sections(data)
            metadata = native_sections.get('.ze_info')
            if metadata:
                text = data[metadata[4]:metadata[4] + metadata[5]].decode()
                metadata_path = output / f'{label}-image{image_index}.ze_info.yaml'
                metadata_path.write_text(text)
                item['ze_info'] = str(metadata_path)
        images.append(item)
    report[label] = {'library': str(library), 'sha256': hashlib.sha256(content).hexdigest(), 'images': images}
(output / 'provenance.json').write_text(json.dumps(report, indent=2) + '\n')
if len(sys.argv) == 1:
    (root / 'native-image-provenance.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
