import json
from pathlib import Path
import traceback

import torch


def save_failure(error, phase):
    directory = Path(__file__).resolve().parent
    tensors = {}
    failures = {}
    current = error.__traceback__
    index = 0
    while current is not None:
        for name, value in list(current.tb_frame.f_locals.items()):
            if isinstance(value, torch.Tensor):
                key = f'frame{index}-{name}'
                try:
                    tensors[key] = value.detach().cpu()
                except Exception as failure:
                    failures[key] = str(failure)
        current = current.tb_next
        index += 1
    path = directory / f'first-failure-{phase}-full-local-tensors.pt'
    torch.save(tensors, path)
    (directory / f'first-failure-{phase}.json').write_text(json.dumps({
        'phase': phase, 'full_local_tensor_file': str(path),
        'tensors': {name: {'shape': list(value.shape), 'dtype': str(value.dtype)} for name, value in tensors.items()},
        'unavailable_tensors': failures, 'original_exception': str(error),
        'raw_traceback': ''.join(traceback.format_exception(type(error), error, error.__traceback__)),
        'original_failure_not_mapped_to_success': True}, indent=2) + '\n')
