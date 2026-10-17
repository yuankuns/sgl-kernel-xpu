import argparse
from pathlib import Path

import stage_fresh_source_site as staging


SUPPORT = staging.SUPPORT


def package(artifact, stage, stems):
    status = 1
    try:
        staging.stage_site(artifact, stage, stems)
        status = 0
    finally:
        (stage / 'isolated-site-stage.exit').write_text(str(status) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--container', action='store_true')
    parser.add_argument('--module', required=True, choices=list(SUPPORT.FAMILIES))
    parser.add_argument('--stage-name', default='fresh-source-icpx-native-v2')
    options = parser.parse_args()
    SUPPORT.STAGE_NAME = options.stage_name
    SUPPORT.ENTRY_SCRIPT = Path(__file__).name
    SUPPORT.package = package
    if options.container:
        raise SystemExit(SUPPORT.container(options.module))
    raise SystemExit(SUPPORT.host(options.module))
