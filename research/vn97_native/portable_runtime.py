"""PC CPU reference runner for the research PC/Android ONNX contract. No PyTorch."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort


def _hash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class CoreState:
    manifest_sha256: str
    position: int
    fast: np.ndarray
    slow: np.ndarray


class PortableCore:
    def __init__(self, directory, expected_manifest_sha256, threads=1):
        self.root = Path(directory)
        path = self.root / 'manifest.json'
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 65536 or _hash(path) != expected_manifest_sha256:
            raise ValueError('manifest identity mismatch')
        def strict_pairs(pairs):
            out = {}
            for key, value in pairs:
                if key in out: raise ValueError('duplicate manifest key')
                out[key] = value
            return out
        self.manifest = m = json.loads(path.read_text(), object_pairs_hook=strict_pairs)
        if set(m) != {'schema','architecture','weights_id','config','parameters','dtype','tokenizer','state_shape','graphs','files','production_activation_authorized','bundle_id'}:
            raise ValueError('manifest fields mismatch')
        if m['schema'] != 'VN97ASCORT1' or m['dtype'] != 'float32' or m['tokenizer'] != 'utf8-bytes-256' or m['production_activation_authorized'] is not False:
            raise ValueError('unsupported research core')
        body = {k:v for k,v in m.items() if k != 'bundle_id'}
        if hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest() != m['bundle_id']:
            raise ValueError('bundle identity mismatch')
        expected = {'1':'chunk-1.onnx','8':'chunk-8.onnx','32':'chunk-32.onnx'}
        if m['graphs'] != expected or set(m['files']) != set(expected.values()) | {'weights.bin'}:
            raise ValueError('unsupported graph inventory')
        self.shape = tuple(m['state_shape'])
        if len(self.shape) != 3 or any(type(v) is not int for v in self.shape) or not (1 <= self.shape[0] <= 8 and self.shape[1] == 1 and 1 <= self.shape[2] <= 1024):
            raise ValueError('invalid state shape')
        for name, identity in m['files'].items():
            file = self.root / name
            if file.is_symlink() or not file.is_file() or type(identity['bytes']) is not int or not 0 < identity['bytes'] <= 128 * 1024 * 1024 or file.stat().st_size != identity['bytes'] or _hash(file) != identity['sha256']:
                raise ValueError('file identity mismatch')
        if type(threads) is not int or not 1 <= threads <= 16: raise ValueError('thread bound')
        self.identity = expected_manifest_sha256
        self.options = ort.SessionOptions(); self.options.intra_op_num_threads = threads
        self.session = None; self.chunk = None

    def advance(self, ids, state=None):
        ids = np.asarray(ids)
        if ids.dtype != np.int64 or ids.ndim != 1 or len(ids) not in (1,8,32) or np.any((ids < 0) | (ids >= 256)):
            raise ValueError('expected 1, 8 or 32 byte token IDs as int64')
        if state is None:
            state = CoreState(self.identity, 0, np.zeros(self.shape, np.float32), np.zeros(self.shape, np.float32))
        if not isinstance(state, CoreState) or state.manifest_sha256 != self.identity or type(state.position) is not int or state.position < 0:
            raise ValueError('foreign or invalid state')
        for value in (state.fast,state.slow):
            if value.shape != self.shape or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError('invalid recurrent state')
        if self.chunk != len(ids):
            self.session = None  # One graph session resident, including when switching mobile budgets.
            self.session = ort.InferenceSession(str(self.root / self.manifest['graphs'][str(len(ids))]),
                                                self.options, providers=['CPUExecutionProvider'])
            self.chunk = len(ids)
        logits, fast, slow = self.session.run(None, {'input_ids':ids[None], 'fast_state':state.fast, 'slow_state':state.slow})
        if logits.shape != (1,len(ids),256) or fast.shape != self.shape or slow.shape != self.shape or not all(np.isfinite(v).all() for v in (logits,fast,slow)):
            raise ValueError('invalid core outputs')
        return logits, CoreState(self.identity, state.position + len(ids), fast, slow)

    def run(self, ids, state=None, max_chunk=8):
        ids = np.asarray(ids)
        if ids.dtype != np.int64 or ids.ndim != 1 or not 1 <= len(ids) <= 4096 or type(max_chunk) is not int or max_chunk not in (1,8,32):
            raise ValueError('invalid sequence or chunk budget')
        if np.any((ids < 0) | (ids >= 256)): raise ValueError('invalid byte token')
        outputs = []; offset = 0
        while offset < len(ids):
            size = max(n for n in (1,8,32) if n <= max_chunk and n <= len(ids)-offset)
            output, state = self.advance(ids[offset:offset+size], state)
            outputs.append(output); offset += size
        return np.concatenate(outputs, axis=1), state
