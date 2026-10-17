import json
import os
from pathlib import Path
import re
import sys


ART = Path(__file__).resolve().parent.parent
CANONICAL_VERSION = os.environ.get('FMHA_CANONICAL_VERSION', 'canonical-v4-icpx')
match = re.fullmatch(r'canonical-v([0-9]+)(?:-icpx)?', CANONICAL_VERSION)
if match is None or int(match.group(1)) < 3:
    raise RuntimeError('An explicit canonical v3-or-newer version with real icpx compiler policy is required; v1/v2 forbidden')
STAGE = ART / CANONICAL_VERSION
FREEZE = ART / f'canonical-source-freeze-{CANONICAL_VERSION.removeprefix("canonical-")}.json'


if __name__ == '__main__':
    if sys.argv[1:] == ['--version-only']:
        print(CANONICAL_VERSION)
    else:
        print(json.dumps({'canonical_version': CANONICAL_VERSION, 'stage': str(STAGE), 'freeze': str(FREEZE),
                          'status': 'path selection only; no package or grant success implied'}))
