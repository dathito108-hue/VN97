"""Research-only ONNX lowering of the existing adaptive core, without training."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import torch
from torch import nn
from torch.nn import functional as F
from language_candidate import NativeLanguageCandidate, load_candidate

SCHEMA = "VN97ASCORT1"
CHUNKS = (1, 8, 32)


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for data in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(data)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


class PortableGraph(nn.Module):
    """Same learned projections/rates and tied output, standard ONNX operators only."""
    def __init__(self, candidate, chunk):
        super().__init__()
        self.core = candidate.backbone
        self.chunk = chunk

    @staticmethod
    def transition(drive, elapsed, rate):
        p = elapsed * rate
        decay = torch.exp(-p)
        # ONNX has Exp but no standard Expm1. Stable small-p ZOH limit avoids
        # subtracting nearly equal FP32 numbers; polynomial error is O(p^5).
        series = 1 - p / 2 + p * p / 6 - p * p * p / 24 + p ** 4 / 120
        gain = torch.where(p < 1e-3, elapsed * series, (1 - decay) / rate)
        return decay, gain * drive

    def scan(self, a, b, initial):
        offset = 1
        while offset < self.chunk:
            a, b = (torch.cat((a[:, :offset], a[:, offset:] * a[:, :-offset]), 1),
                    torch.cat((b[:, :offset], a[:, offset:] * b[:, :-offset] + b[:, offset:]), 1))
            offset *= 2
        return a * initial.unsqueeze(1) + b

    def forward(self, input_ids, fast_state, slow_state):
        x = self.core.embedding(input_ids)
        fast_out, slow_out = [], []
        for index, layer in enumerate(self.core.layers):
            cell = layer.cell
            drive, duration, write, read = cell.projection(layer.norm(x)).chunk(4, -1)
            dt = torch.sigmoid(duration) * cell.config.max_dt
            write, read = torch.sigmoid(write), torch.sigmoid(read)
            fast_rate, slow_rate = cell.rates()
            a, b = self.transition(drive, dt, fast_rate)
            fast = self.scan(a, b, fast_state[index])
            a, b = self.transition(drive, dt * write, slow_rate)
            slow = self.scan(a, b, slow_state[index])
            x = x + layer.out_proj((1 - read) * fast + read * slow)
            fast_out.append(fast[:, -1]); slow_out.append(slow[:, -1])
        logits = F.linear(self.core.final_norm(x), self.core.embedding.weight)
        return logits, torch.stack(fast_out), torch.stack(slow_out)


def export_core(candidate, destination):
    import onnx
    from onnx import external_data_helper, numpy_helper
    if not isinstance(candidate, NativeLanguageCandidate):
        raise ValueError('expected the existing VN97 adaptive candidate')
    if any(p.device.type != 'cpu' or p.dtype != torch.float32 or not bool(torch.isfinite(p).all())
           for p in candidate.parameters()):
        raise ValueError('export requires finite CPU FP32 weights')
    destination = Path(destination)
    if destination.exists():
        raise ValueError('destination already exists; refusing to replace an existing bundle')
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.vn97-core-', dir=destination.parent))
    training = candidate.training
    try:
        candidate.eval()
        config = candidate.config
        weight_identity = hashlib.sha256()
        for name, value in sorted(candidate.state_dict().items()):
            weight_identity.update(canonical([name, list(value.shape), str(value.dtype)]))
            weight_identity.update(value.detach().contiguous().numpy().tobytes())
        graphs = {}
        payloads = {}
        with (stage / 'weights.bin').open('wb') as weights:
            for chunk in CHUNKS:
                name = f'chunk-{chunk}.onnx'
                path = stage / name
                shape = (config.n_layers, 1, config.d_inner)
                torch.onnx.export(PortableGraph(candidate, chunk),
                    (torch.zeros((1, chunk), dtype=torch.long), torch.zeros(shape), torch.zeros(shape)),
                    str(path), opset_version=18, dynamo=False,
                    input_names=['input_ids', 'fast_state', 'slow_state'],
                    output_names=['logits', 'next_fast_state', 'next_slow_state'])
                graph = onnx.load(str(path))
                for tensor in graph.graph.initializer:
                    if not tensor.HasField('raw_data'):
                        tensor.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(tensor), tensor.name))
                    raw = tensor.raw_data
                    key = hashlib.sha256(raw).hexdigest()
                    if key not in payloads:
                        padding = (-weights.tell()) % 64
                        weights.write(b'\0' * padding)
                        payloads[key] = (weights.tell(), len(raw))
                        weights.write(raw)
                    offset, length = payloads[key]
                    external_data_helper.set_external_data(tensor, location='weights.bin', offset=offset, length=length)
                    tensor.ClearField('raw_data')
                onnx.save_model(graph, str(path))
                graphs[str(chunk)] = name
        for name in graphs.values():
            onnx.checker.check_model(str(stage / name))
        files = {p.name: {'bytes': p.stat().st_size, 'sha256': digest_file(p)}
                 for p in sorted(stage.iterdir())}
        body = {'schema': SCHEMA, 'architecture': config.fingerprint(), 'weights_id': weight_identity.hexdigest(),
                'config': config.__dict__, 'parameters': config.parameter_count(), 'dtype': 'float32',
                'tokenizer': 'utf8-bytes-256', 'state_shape': [config.n_layers, 1, config.d_inner],
                'graphs': graphs, 'files': files, 'production_activation_authorized': False}
        manifest = dict(body, bundle_id=hashlib.sha256(canonical(body)).hexdigest())
        (stage / 'manifest.json').write_bytes(canonical(manifest))
        stage.rename(destination)
        return manifest
    finally:
        candidate.train(training)
        if stage.exists():
            shutil.rmtree(stage)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = export_core(load_candidate(args.checkpoint), args.output)
    print(json.dumps({'bundle_id': result['bundle_id'], 'parameters': result['parameters'],
                      'production_activation_authorized': False, 'training_performed': False}))
