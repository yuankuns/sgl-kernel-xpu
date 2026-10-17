import weakref


def tensor_metadata(tensor):
    storage = tensor.untyped_storage()
    return {'dtype': str(tensor.dtype), 'device': str(tensor.device), 'shape': list(tensor.shape),
        'stride': list(tensor.stride()), 'storage_offset': tensor.storage_offset(),
        'tensor_data_ptr': tensor.data_ptr(), 'storage_data_ptr': storage.data_ptr(),
        'storage_nbytes': storage.nbytes()}


def make_cast_observer(torch, retain=False):
    from torch.utils._python_dispatch import TorchDispatchMode

    class OriginalCastObserver(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.records = []
            self.weak_outputs = []
            self.retained_outputs = []

        def __torch_dispatch__(self, function, types, args=(), kwargs=None):
            result = function(*args, **(kwargs or {}))
            if function == torch.ops.aten._to_copy.default and args and torch.is_tensor(args[0]) and torch.is_tensor(result):
                original = args[0]
                if original.dtype == torch.int64 and result.dtype == torch.int32 and original.device.type == result.device.type == 'xpu':
                    self.records.append({'original_real_operator': str(function), 'input': tensor_metadata(original),
                        'result': tensor_metadata(result), 'explicit_retention_control': retain})
                    self.weak_outputs.append(weakref.ref(result))
                    if retain:
                        self.retained_outputs.append(result)
            return result

        def python_wrapper_liveness(self):
            return [{'Python_tensor_wrapper_alive': reference() is not None,
                'wrapper_absence_NOT_storage_freed_proof': True} for reference in self.weak_outputs]

    return OriginalCastObserver()
