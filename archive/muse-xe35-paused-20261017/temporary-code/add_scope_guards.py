import subprocess
from pathlib import Path


ROOT = Path('/home/syk/worktrees/sgl-kernel-xpu-up/pr')


def apply_update(filename, original, replacement):
    patch = (
        '*** Begin Patch\n'
        + f'*** Update File: {filename}\n@@\n'
        + ''.join('-' + line + '\n' for line in original.splitlines())
        + ''.join('+' + line + '\n' for line in replacement.splitlines())
        + '*** End Patch\n'
    )
    subprocess.run(['apply_patch'], input=patch, text=True, check=True)


def main():
    memory = ROOT / 'muse-memory-xe35/benchmark/bench_memory_compare.py'
    original = memory.read_text()
    replacement = original.replace('import json\n', 'import json\nimport os\n')
    replacement = replacement.replace(
        '    if args.batch != 8:\n',
        '    if args.batch != 8 or any(\n'
        '        os.environ.get(name, "8") != "8"\n'
        '        for name in ("BATCHES", "BATCH_SIZE", "SERVING_BATCH")\n'
        '    ):\n',
    )
    apply_update(memory, original, replacement)
    for component, benchmark in (
        ('position', 'bench_position_occupancy.py'),
        ('logits-vocab', 'bench_logits_vocab_sweep.py'),
    ):
        filename = ROOT / f'muse-{component}-xe35/benchmark' / benchmark
        original = filename.read_text()
        guard = '''import os

if __name__ == "__main__":
    scope_parser_xe35 = argparse.ArgumentParser(add_help=False)
    scope_parser_xe35.add_argument("--batch", "--serving-batch", type=int, default=8)
    scope_parser_xe35.add_argument("--isl", type=int, default=4096)
    scope_parser_xe35.add_argument("--osl", type=int, default=16)
    scope_parser_xe35.add_argument("--tp", type=int, default=1)
    scope_parser_xe35.add_argument("--ep", type=int, default=1)
    scope_xe35, _unknown_xe35 = scope_parser_xe35.parse_known_args()
    if (scope_xe35.batch, scope_xe35.isl, scope_xe35.osl, scope_xe35.tp, scope_xe35.ep) != (
        8, 4096, 16, 1, 1
    ) or any(
        os.environ.get(name, "8") != "8"
        for name in ("BATCHES", "BATCH_SIZE", "SERVING_BATCH")
    ):
        scope_parser_xe35.error("Muse-Spark serving scope requires B8/ISL4096/OSL16/TP1/EP1")

'''
        replacement = original.replace('import argparse\n', 'import argparse\n' + guard, 1)
        apply_update(filename, original, replacement)


if __name__ == '__main__':
    main()
