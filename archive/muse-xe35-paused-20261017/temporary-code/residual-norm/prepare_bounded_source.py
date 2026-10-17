import hashlib
import difflib
import io
import json
from pathlib import Path
import subprocess
import tarfile


ROOT = Path(__file__).resolve().parent
REPO = Path('/home/syk/worktrees/sgl-kernel-xpu/muse-spark-residual-norm-main-20261014')
REF = 'ec29f7a6a14c4f1fac42fb93526236ca03ab894b'
JOB = ROOT / 'source-before-icpx-bounded-v1'
JOB.mkdir(exist_ok=False)
snapshot = JOB / 'source'
snapshot.mkdir()
archive = subprocess.check_output(['git', '--no-optional-locks', '-C', str(REPO), 'archive', REF])
with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
    members = tar.getmembers()
    assert all(member.isdir() or member.isfile() for member in members)
    tar.extractall(snapshot, filter='data')
before = {path.relative_to(snapshot).as_posix(): {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
          'size': path.stat().st_size, 'mode': path.stat().st_mode & 0o777}
          for path in snapshot.rglob('*') if path.is_file()}
flags = snapshot / 'cmake/BuildFlags.cmake'
text = flags.read_text()
old_gate = 'if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")'
old_message = '  message("Not compiling with XPU. Currently only support GCC compiler on Linux as CXX compiler.")'
assert text.count(old_gate) == text.count(old_message) == 1
subprocess.run(['apply_patch', f'*** Begin Patch\n*** Update File: {flags}\n@@\n-{old_gate}\n+if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")\n@@\n-{old_message}\n+  message("Not compiling with XPU. Currently only support GCC or IntelLLVM compiler on Linux as CXX compiler.")\n*** End Patch\n'], check=True)
after = {path.relative_to(snapshot).as_posix(): {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
         'size': path.stat().st_size, 'mode': path.stat().st_mode & 0o777}
         for path in snapshot.rglob('*') if path.is_file()}
assert [name for name in before if before[name] != after[name]] == ['cmake/BuildFlags.cmake']
record = {'source_ref': REF, 'repo': str(REPO), 'archive_sha256': hashlib.sha256(archive).hexdigest(),
          'original_files': before, 'snapshot_files': after, 'only_delta': 'cmake/BuildFlags.cmake (gate + message, 2 lines)',
          'source_digest': hashlib.sha256(json.dumps(after, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
          'original_helpers_and_math_preserved': True, 'original_repo_PR_written': False}
(JOB / 'snapshot-manifest.json').write_text(json.dumps(record, indent=2) + '\n')
(JOB / 'BuildFlags.diff').write_text(''.join(difflib.unified_diff(text.splitlines(keepends=True), flags.read_text().splitlines(keepends=True), fromfile='ec29f7a/cmake/BuildFlags.cmake', tofile='snapshot/cmake/BuildFlags.cmake')))
print(f'Prepared exact {REF} snapshot {len(after)} files; only gate/message changed; no build/GPU launched')
