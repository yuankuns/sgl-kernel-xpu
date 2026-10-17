import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent / 'canonical-tools'))
from overlay_snapshot import digest_file, inventory
