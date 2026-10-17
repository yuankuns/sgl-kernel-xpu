import json
import os
import time

from providers import preflight, verify_provider


preflight()


def pytest_sessionstart(session):
    report = verify_provider(os.environ['NV_SOURCE_LABEL'], cpu_only=True)
    import torch
    import sgl_kernel

    assert sgl_kernel.is_xe3_arch(), 'Actual architecture must be Xe3.5'
    print(json.dumps(report), flush=True)


def pytest_runtest_teardown(item, nextitem):
    import torch

    torch.xpu.synchronize()
    time.sleep(2)
