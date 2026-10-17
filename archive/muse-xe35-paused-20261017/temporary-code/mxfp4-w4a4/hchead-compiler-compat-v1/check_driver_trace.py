import json
import pathlib
import shlex


def main():
    sidecar = pathlib.Path(__file__).resolve().parent
    trace = (sidecar / 'sycl-host-only-driver-trace.log').read_text()
    frontends = [shlex.split(line) for line in trace.splitlines() if '"-cc1"' in line]
    devices = [arguments for arguments in frontends if '-fsycl-is-device' in arguments]
    hosts = [arguments for arguments in frontends if '-fsycl-is-host' in arguments]
    if len(devices) != 1 or len(hosts) != 1:
        raise RuntimeError('Unexpected real host-only driver frontend actions')
    if '-fsyntax-only' not in devices[0] or '-o' in devices[0] or any(flag in devices[0] for flag in ('-emit-obj', '-emit-llvm', '-emit-llvm-bc')):
        raise RuntimeError('Host-only mode unexpectedly produces device code')
    if '-emit-obj' not in hosts[0]:
        raise RuntimeError('Host-only mode did not emit the actual host object')
    report = {'kind': 'CPU dry driver trace plus previous real successful host-only compilation, not full/GPU acceptance', 'actual_host_only_compile_exit': 0, 'device_frontend_present': True, 'device_frontend_action': '-fsyntax-only for integration header/footer; no output device object/IR', 'device_code_generation': False, 'host_frontend_action': '-emit-obj', 'host_object_offload_payload': False, 'root_cause': 'plain IntelLLVM host compilation does not use SYCL attribute language mode; device frontend accepts the original one-dimensional attribute', 'recommended_fix': 'Local device-only attribute guard rather than changing every host compilation mode', 'extra_compile_this_check': False, 'gpu': False, 'frozen_pr_writes': 0}
    (sidecar / 'sycl-host-only-driver-proof.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
