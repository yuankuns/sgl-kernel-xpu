import argparse
import pathlib
import sys

import pytest
import sgl_kernel
import torch


def forbidden_gpu(*args, **kwargs):
    raise AssertionError('Mocked CPU architecture audit must not initialize or use an actual GPU')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--module', choices=('mxfp4-w4a4', 'routing-quant'), required=True)
    parser.add_argument('--arch', choices=('cpu', 'bmg'), required=True)
    args = parser.parse_args()
    torch.xpu.is_available = lambda: args.arch == 'bmg'
    torch.xpu.init = forbidden_gpu
    torch.xpu._lazy_init = forbidden_gpu
    torch.xpu.current_device = forbidden_gpu
    torch.xpu.synchronize = lambda *args, **kwargs: None
    torch.xpu.empty_cache = lambda: None
    torch.xpu.manual_seed_all = lambda seed: None
    sgl_kernel.is_xe3_arch = lambda: False
    target = pathlib.Path('/root/worktrees/sgl-kernel-xpu-up/pr') / ('muse-' + args.module + '-xe35')
    sys.path.insert(0, str(target / 'tests'))
    if args.module == 'mxfp4-w4a4':
        files = ['test_mxfp4_moe_w4a4.py']
        options = ['--run-long-tests']
    else:
        files = ['test_moe_routing.py', 'test_per_token_group_quant_mxfp4_routing.py', 'test_topk_sigmoid_routing.py', 'test_routing_benchmark_scope.py']
        options = []
    raise SystemExit(pytest.main(['-q', '-p', 'no:cacheprovider'] + options + [str(target / 'tests' / name) for name in files]))


if __name__ == '__main__':
    main()
