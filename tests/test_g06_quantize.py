import sys
from dataclasses import asdict
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from g06_quantize import quantize_graph
from test_r2_mamba2_g05_parallel import _model
from vn97.r2.mamba2_parallel_onnx import export_parallel_recurrent_onnx


@pytest.mark.parametrize('mode', ['int8', 'mixed-int4'])
def test_quantized_ssm_graph_runs_continuation(tmp_path, mode):
    model = _model().half()
    export_parallel_recurrent_onnx(model, tmp_path / 'source', capsule_id='a'*64,
                                  capsule_manifest_sha256='b'*64, source_weight_sha256='c'*64)
    raw = onnx.load(str(tmp_path / 'source/recurrent-8.onnx'))
    quant, counts = quantize_graph(raw, asdict(model.config), mode)
    assert counts == ({'8': 4} if mode == 'int8' else {'4': 2, '8': 2})
    target = tmp_path / 'quant.onnx'
    onnx.save(quant, target)
    onnx.checker.check_model(str(target))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(target), sess_options=options, providers=['CPUExecutionProvider'])
    conv, ssm = (x.numpy() for x in model.initial_state())
    for length in (8, 1, 3):
        logits, conv, ssm = session.run(None, {'input_ids': np.ones((1, 8), dtype=np.int64),
            'valid_length': np.array([length], dtype=np.int64), 'conv_state': conv, 'ssm_state': ssm})
        assert all(np.isfinite(x).all() for x in (logits, conv, ssm))
        assert conv.dtype == ssm.dtype == np.float16
        assert np.count_nonzero(logits[:, length:]) == 0


def test_rejects_missing_projection_coverage():
    empty = onnx.helper.make_model(onnx.helper.make_graph([], 'empty', [], []))
    with pytest.raises(ValueError, match='two projection'):
        quantize_graph(empty, asdict(_model().config), 'int8')
