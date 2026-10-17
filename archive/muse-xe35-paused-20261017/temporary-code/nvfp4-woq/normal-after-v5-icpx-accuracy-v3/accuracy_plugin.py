import json
import os
import time

from providers import preflight, runtime_grant, verify_provider


preflight()


def pytest_sessionstart(session):
    runtime_grant()
    report = verify_provider(os.environ['NV_ACCURACY_LABEL'], cpu_only=True)
    import sgl_kernel

    assert sgl_kernel.is_xe3_arch(), 'Actual architecture must be Xe3.5'
    print(json.dumps(report), flush=True)


def pytest_runtest_teardown(item, nextitem):
    import torch

    torch.xpu.synchronize()
    time.sleep(2)
