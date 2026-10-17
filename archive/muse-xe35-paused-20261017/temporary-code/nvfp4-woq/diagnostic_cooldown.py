import time

import torch


def pytest_runtest_teardown(item, nextitem):
    torch.xpu.synchronize()
    time.sleep(2)
