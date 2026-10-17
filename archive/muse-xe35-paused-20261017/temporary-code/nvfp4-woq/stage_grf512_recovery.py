import hashlib
import json
import pathlib
import shutil

root = pathlib.Path(__file__).resolve().parent
assert (root / 'diagnostic-barrier-round.exit').is_file()
assert (root / 'native-grf512-targeted-build-v2.exit').read_text().strip() == '0'
source = root / 'port-compile-audit/src/libsgl-ops-sycl-blockwise_moe_nvfp4_woq_port_payload.so'
destination = root / 'native-diagnostic/port/site/sgl_kernel' / source.name
digest = hashlib.sha256(source.read_bytes()).hexdigest()
expected = (root / 'native-grf512-targeted-build-v2.sha256').read_text().split()[0]
assert digest == expected
record = {'scope': 'actual port-kernel diagnostic, NOT official full package acceptance', 'original_round_exit': int((root / 'diagnostic-barrier-round.exit').read_text().strip()), 'old_sha256': hashlib.sha256(destination.read_bytes()).hexdigest(), 'new_sha256': digest, 'source': str(source), 'destination': str(destination), 'policy': '-cl-intel-512-GRF-per-thread, source-preserved per NVFP4 target', 'compile_jobs': 0}
archive = root / 'native-diagnostic/original-round-payload'
archive.mkdir(exist_ok=True)
shutil.copy2(destination, archive / destination.name)
shutil.copy2(source, destination)
assert hashlib.sha256(destination.read_bytes()).hexdigest() == digest
(root / 'grf512-recovery-stage.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2))
