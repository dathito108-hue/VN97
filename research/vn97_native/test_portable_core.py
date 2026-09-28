import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import onnx
import torch
from language_candidate import CandidateConfig, NativeLanguageCandidate
from portable_core import PortableGraph, export_core, digest_file
from portable_runtime import CoreState, PortableCore


class PortableCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(97); torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name) / 'core'
        cls.model = NativeLanguageCandidate(CandidateConfig(d_model=8, d_inner=12, n_layers=2)).eval()
        cls.original = {k:v.clone() for k,v in cls.model.state_dict().items()}
        cls.manifest = export_core(cls.model, cls.root)
        cls.identity = digest_file(cls.root / 'manifest.json')
        cls.ids = np.random.default_rng(97).integers(0,256,73,dtype=np.int64)

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def runner(self): return PortableCore(self.root, self.identity)

    def test_pc_mobile_chunk_profiles_match_original_logits_and_final_state(self):
        with torch.no_grad():
            expected, reference = self.model(torch.from_numpy(self.ids)[None])
        expected_fast = torch.stack([layer.ssm[...,0] for layer in reference.recurrent.layers]).numpy()
        expected_slow = torch.stack([layer.ssm[...,1] for layer in reference.recurrent.layers]).numpy()
        for budget in (1,8,32):
            actual, state = self.runner().run(self.ids, max_chunk=budget)
            np.testing.assert_allclose(actual, expected.numpy(), rtol=3e-4, atol=3e-5)
            np.testing.assert_allclose(state.fast, expected_fast, rtol=3e-4, atol=3e-5)
            np.testing.assert_allclose(state.slow, expected_slow, rtol=3e-4, atol=3e-5)
            self.assertEqual(state.position, len(self.ids))
            print(f'ORT_PARITY budget={budget} max_abs_logits={np.abs(actual-expected.numpy()).max():.8g}', flush=True)

    def test_switching_device_budget_preserves_nonzero_state(self):
        runner = self.runner()
        first, state = runner.run(self.ids[:32], max_chunk=32)
        second, state = self.runner().run(self.ids[32:64], state, max_chunk=8)
        third, state = runner.run(self.ids[64:], state, max_chunk=1)
        expected, end = self.runner().run(self.ids, max_chunk=32)
        np.testing.assert_allclose(np.concatenate((first,second,third),1), expected, rtol=3e-4, atol=3e-5)
        np.testing.assert_allclose(state.fast, end.fast, rtol=3e-4, atol=3e-5)
        self.assertEqual(state.position,73)

    def test_weights_unchanged_and_shared_external_payload(self):
        for name, param in self.model.state_dict().items():
            torch.testing.assert_close(param, self.original[name], rtol=0, atol=0)
        for name in self.manifest['graphs'].values():
            graph = onnx.load(str(self.root/name), load_external_data=False)
            self.assertTrue(graph.graph.initializer)
            for tensor in graph.graph.initializer:
                self.assertEqual(tensor.data_location, onnx.TensorProto.EXTERNAL)
                self.assertEqual(dict((x.key,x.value) for x in tensor.external_data)['location'], 'weights.bin')
        self.assertFalse(self.manifest['production_activation_authorized'])
        self.assertEqual(self.manifest['parameters'], self.model.config.parameter_count())

    def test_stable_zoh_limit_and_gradients_without_optimizer(self):
        elapsed = torch.tensor([0.,1e-9,1e-6,1e-4,1e-2,1.,4.],dtype=torch.float64,requires_grad=True)
        drive = torch.ones_like(elapsed)
        rate = torch.tensor(.05,dtype=torch.float64,requires_grad=True)
        a,b = PortableGraph.transition(drive,elapsed,rate)
        expected = -torch.expm1(-elapsed*rate)/rate
        torch.testing.assert_close(b,expected,rtol=1e-11,atol=1e-14)
        (a.sum()+b.sum()).backward()
        self.assertTrue(torch.isfinite(elapsed.grad).all())
        self.assertTrue(torch.isfinite(rate.grad))

    def test_bad_tokens_and_foreign_state_are_rejected(self):
        runner = self.runner(); _,state = runner.advance(self.ids[:1])
        bad = CoreState('0'*64,state.position,state.fast,state.slow)
        with self.assertRaises(ValueError): runner.advance(self.ids[:1],bad)
        for ids in (np.array([-1],np.int64),np.array([256],np.int64),np.array([1.],np.float32),self.ids[:2]):
            with self.assertRaises(ValueError): runner.advance(ids,state)
        fast=state.fast.copy();fast.flat[0]=np.nan
        with self.assertRaises(ValueError):runner.advance(self.ids[:1],CoreState(self.identity,1,fast,state.slow))

    def test_manifest_and_weight_tamper_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'identity'): PortableCore(self.root,'0'*64)
        weights=self.root/'weights.bin'
        original=weights.read_bytes()
        try:
            weights.write_bytes(original[:-1]+bytes([original[-1]^1]))
            with self.assertRaisesRegex(ValueError,'identity'): self.runner()
        finally:weights.write_bytes(original)

    def test_export_does_not_overwrite_existing_bundle(self):
        with self.assertRaisesRegex(ValueError,'already exists'): export_core(self.model,self.root)
        self.assertEqual(digest_file(self.root/'manifest.json'),self.identity)


if __name__=='__main__': unittest.main(verbosity=2)
