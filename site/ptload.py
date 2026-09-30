"""Lê um state_dict salvo com torch.save sem precisar do PyTorch (zip + pickle)."""
import collections, pickle, zipfile
import numpy as np

DT = {'FloatStorage': np.float32, 'DoubleStorage': np.float64, 'HalfStorage': np.float16,
      'LongStorage': np.int64, 'IntStorage': np.int32, 'BoolStorage': np.bool_, 'ByteStorage': np.uint8}

def load_pt(path):
    zf = zipfile.ZipFile(path)
    pkl = [n for n in zf.namelist() if n.endswith('data.pkl')][0]
    prefix = pkl[:-len('data.pkl')]

    def rebuild(storage, offset, size, stride, *rest):
        it = storage.itemsize
        return np.lib.stride_tricks.as_strided(storage[offset:], shape=tuple(size),
                                               strides=tuple(s * it for s in stride)).copy()

    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            if mod == 'torch._utils' and name == '_rebuild_tensor_v2':
                return rebuild
            if mod == 'collections' and name == 'OrderedDict':
                return collections.OrderedDict
            if mod == 'torch' and name.endswith('Storage'):
                return name
            if mod == 'torch._utils' and name == '_rebuild_parameter':
                return lambda data, requires_grad, hooks: data
            return super().find_class(mod, name)

        def persistent_load(self, pid):
            _, stype, key, _loc, numel = pid
            stype = stype if isinstance(stype, str) else getattr(stype, '__name__', str(stype))
            dt = DT[stype]
            return np.frombuffer(zf.read(prefix + 'data/' + str(key)), dtype=dt)[:numel]

    return U(zf.open(pkl)).load()
