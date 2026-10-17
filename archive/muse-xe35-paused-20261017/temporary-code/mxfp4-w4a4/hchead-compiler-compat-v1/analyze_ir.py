import hashlib
import json
import pathlib
import re
import subprocess


def digest_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def kernel_metadata(text):
    definitions = dict(re.findall(r'^!(\d+) = (.*)$', text, re.MULTILINE))
    result = {}
    for line in text.splitlines():
        if line.startswith('define ') and 'spir_kernel' in line:
            name = re.search(r'@([^ (]+)', line).group(1)
            properties = {}
            for property_name, identifier in re.findall(r'!(reqd_work_group_size|intel_reqd_sub_group_size|reqd_sub_group_size) !(\d+)', line):
                properties[property_name] = definitions[identifier]
            if properties:
                result[name] = properties
    return result


def main():
    sidecar = pathlib.Path(__file__).resolve().parent
    for name in ('plain-host-original', 'sycl-host-only-original', 'plain-host-device-guard', 'original-device-ir', 'guarded-device-ir'):
        expected = '1' if name == 'plain-host-original' else '0'
        if (sidecar / (name + '.exit')).read_text().strip() != expected:
            raise RuntimeError(f'Actual compile result mismatch: {name}')
    original = (sidecar / 'original-device.ll').read_text()
    guarded = (sidecar / 'guarded-device.ll').read_text()
    original_metadata = kernel_metadata(original)
    guarded_metadata = kernel_metadata(guarded)
    if not original_metadata or original_metadata != guarded_metadata:
        raise RuntimeError('Original/guarded actual kernel work-group/sub-group metadata differ')
    for properties in original_metadata.values():
        if 'reqd_work_group_size' not in properties or not any('sub_group_size' in name for name in properties):
            raise RuntimeError('Actual WG/SG property metadata missing')
    def normalize(text):
        source_name = re.search(r'^source_filename = "([^"]+)"$', text, re.MULTILINE).group(1)
        module_names = re.findall(r'"sycl-module-id"="([^"]+)"', text)
        if not module_names or any(name != source_name for name in module_names):
            raise RuntimeError('SYCL module-ID is not exactly the source filename path')
        lines = [line for line in text.splitlines() if not line.startswith(("; ModuleID =", 'source_filename ='))]
        return '\n'.join(lines).replace('"sycl-module-id"="' + source_name + '"', '"sycl-module-id"="HCHead.cpp"')
    normalized_original = normalize(original)
    normalized_guarded = normalize(guarded)
    if normalized_original != normalized_guarded:
        raise RuntimeError('Device IR differs beyond the three source-path identity fields')
    object_sections = {}
    for name in ('sycl-host-only-original', 'plain-host-device-guard'):
        result = subprocess.run(['readelf', '-SW', str(sidecar / (name + '.o'))], capture_output=True, text=True, check=True)
        (sidecar / (name + '.sections.log')).write_text(result.stdout)
        if '.llvm.offloading' in result.stdout or '__CLANG_OFFLOAD_BUNDLE__' in result.stdout:
            raise RuntimeError('Unexpected device payload in host-only object')
        object_sections[name] = {'readelf_exit': 0, 'has_device_offload_section': False}
    original_source = (sidecar / 'HCHead.original.cpp').read_text()
    guarded_source = (sidecar / 'HCHead.device-guard.cpp').read_text()
    expected_source = original_source.replace('  [[sycl::reqd_work_group_size(WG_SIZE)]] [[sycl::reqd_sub_group_size(SG_SIZE)]] void\n', '  #PLACEHOLDER\n', 1)
    actual_source = guarded_source.replace('#ifdef __SYCL_DEVICE_ONLY__\n  [[sycl::reqd_work_group_size(WG_SIZE)]] [[sycl::reqd_sub_group_size(SG_SIZE)]]\n#endif\n  void\n', '  #PLACEHOLDER\n', 1)
    if expected_source != actual_source:
        raise RuntimeError('Guard proposal changed other source or kernel body')
    report = {'kind': 'CPU compile/device LLVM IR proof only; not full/package/GPU acceptance', 'recommended_patch': 'HCHead only: __SYCL_DEVICE_ONLY__ around unchanged one-argument WG and SG attributes; void remains outside guard', 'original_plain_host_compile_exit': 1, 'actual_host_only_original_compile_exit': 0, 'actual_plain_host_guard_compile_exit': 0, 'original_device_ir_compile_exit': 0, 'guarded_device_ir_compile_exit': 0, 'actual_kernel_count_with_wg_sg_metadata': len(original_metadata), 'actual_original_and_guarded_metadata': original_metadata, 'all_wg_sg_metadata_identical': True, 'normalized_entire_device_ir_identical': True, 'normalization_only': ['ModuleID source path', 'source_filename source path', 'sycl-module-id source path, verified identical to source_filename'], 'normalized_device_ir_sha256': digest_text(normalized_original), 'original_device_ir_raw_sha256': digest_text(original), 'guarded_device_ir_raw_sha256': digest_text(guarded), 'kernel_body_and_nd_item_dimension_and_wg_values_unchanged': True, 'host_only_object_sections': object_sections, 'production_device_flags': 'unchanged from original generated HCHead command', 'source_preserving_option': 'local device-only attribute guard avoids global host compilation mode change; host-only language option also compiled successfully without an offload payload', 'no_full_or_gpu': True, 'frozen_pr_writes': 0}
    (sidecar / 'device-ir-equivalence.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
