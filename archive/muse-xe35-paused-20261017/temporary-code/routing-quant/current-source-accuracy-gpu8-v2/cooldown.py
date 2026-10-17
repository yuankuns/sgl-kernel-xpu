import time


def pytest_runtest_teardown(item, nextitem):
    import torch

    if torch.xpu.is_available():
        torch.xpu.synchronize()
        time.sleep(2)
