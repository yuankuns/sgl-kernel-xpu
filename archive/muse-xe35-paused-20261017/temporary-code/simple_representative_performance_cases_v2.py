import inspect
import os
from dataclasses import dataclass
from pathlib import Path

import benchmark_simple
from simple_verified_graph_v2 import CALLS, digest_file, time_verified_graph, validate_parameters


FROZEN_FACTORY_SHA256 = 'ee8b80880b07fb679db74a1d18c250be412fc9c5faed0a5ca3e6956752befff3'


def verify_frozen_factory():
    path = Path(__file__).with_name('benchmark_simple.py')
    if Path(benchmark_simple.__file__).resolve() != path.resolve() or digest_file(path) != FROZEN_FACTORY_SHA256:
        raise RuntimeError('Exact frozen benchmark_simple factory provider and file SHA required')


@dataclass
class RepresentativeCase:
    name: str
    launch: object
    check: object
    reset: object = None
    check_single: object = None
    before_capture: object = None
    after_capture: object = None
    softcap_state: object = None

    def measure(self, torch, *, case_index, evidence_directory):
        verify_frozen_factory()
        return time_verified_graph(torch, self.launch, case_index=case_index, case_name=self.name, evidence_directory=evidence_directory, reset=self.reset, check=self.check, check_single=self.check_single, before_capture=self.before_capture, after_capture=self.after_capture)


class IndependentSoftcapBuffers:
    def __init__(self, torch, operations, frozen_case):
        self.torch = torch
        self.operation = operations['softcap_logits']
        name, launch, check, reset = frozen_case
        if name != 'softcap-fp32-b8-v202048':
            raise RuntimeError('Original frozen softcap case required')
        inputs = inspect.getclosurevars(reset).nonlocals
        reference = inspect.getclosurevars(check).nonlocals
        self.original = inputs['original']
        self.expected = reference['expected_logits']
        self.cap = inspect.getclosurevars(launch).nonlocals['cap']
        if tuple(self.original.shape) != (8, 202048) or self.original.dtype != torch.float32 or self.original.device.type != 'xpu' or self.cap != 20.0:
            raise RuntimeError('Softcap requires original (8,202048) FP32 XPU input and cap20')
        if tuple(self.expected.shape) != (8, 202048) or self.expected.dtype != torch.float32 or self.expected.device.type != 'cpu':
            raise RuntimeError('Exact frozen full FP32 CPU reference required')
        self.buffers = (inputs['logits'],) + tuple(self.original.clone() for buffer_index in range(CALLS - 1))
        if len({buffer.data_ptr() for buffer in self.buffers} | {self.original.data_ptr()}) != CALLS + 1:
            raise RuntimeError('All128 buffers must have independent original-preserving storage')
        self.capture_calls = None
        self.capture_index = 0

    def before_capture(self, calls):
        if calls not in (1, CALLS) or self.torch.xpu.is_current_stream_capturing():
            raise RuntimeError('Prepare only real single or128-call capture outside capture')
        self.capture_calls = calls
        self.capture_index = 0

    def launch(self):
        if self.torch.xpu.is_current_stream_capturing():
            if self.capture_calls is None or self.capture_index >= self.capture_calls:
                raise RuntimeError('Real capture selector overrun or unprepared capture')
            buffer = self.buffers[self.capture_index]
            self.capture_index += 1
        else:
            buffer = self.buffers[0]
        self.operation(buffer, self.cap)

    def after_capture(self, calls):
        if self.torch.xpu.is_current_stream_capturing() or self.capture_calls != calls or self.capture_index != calls:
            raise RuntimeError('Every independent softcap buffer must be selected once during real capture')
        self.capture_calls = None
        self.capture_index = 0

    def reset(self):
        if self.torch.xpu.is_current_stream_capturing():
            raise RuntimeError('Softcap reset copies must never be captured or timed')
        for buffer in self.buffers:
            buffer.copy_(self.original)

    def check_single(self):
        self.torch.testing.assert_close(self.buffers[0].cpu(), self.expected, atol=3e-5, rtol=3e-5)

    def check(self):
        for buffer in self.buffers:
            self.torch.testing.assert_close(buffer.cpu(), self.expected, atol=3e-5, rtol=3e-5)


def representative_cases(torch, operations, component):
    validate_parameters()
    verify_frozen_factory()
    factories = {'memory': benchmark_simple.memory_cases, 'position': benchmark_simple.position_cases, 'logits-vocab': benchmark_simple.logits_cases}
    if component not in factories:
        raise ValueError('Only frozen memory/position/logits-vocab representative cases supported')
    if os.environ.get('ZE_AFFINITY_MASK') not in ('4', '5', '6', '7'):
        raise RuntimeError('Representative performance factories require a new main GPU4–7 grant before GPU work')
    import sgl_kernel

    if not torch.xpu.is_available() or not sgl_kernel.is_xe3_arch():
        raise RuntimeError('Real Xe35 compute capability3.5 required before factory native calls')
    for frozen_case in factories[component](torch, operations):
        name, launch, check, reset = frozen_case
        if name == 'softcap-fp32-b8-v202048':
            state = IndependentSoftcapBuffers(torch, operations, frozen_case)
            yield RepresentativeCase(name, state.launch, state.check, state.reset, state.check_single, state.before_capture, state.after_capture, state)
        else:
            yield RepresentativeCase(name, launch, check, reset)
