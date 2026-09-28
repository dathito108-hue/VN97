import json
from pathlib import Path
import tempfile
import unittest
import torch
from language_candidate import CandidateConfig, NativeLanguageCandidate, save_candidate, load_candidate


class LanguageCandidateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(97)
        torch.set_num_threads(1)
        self.model = NativeLanguageCandidate(CandidateConfig(d_model=8, d_inner=12, n_layers=2))
        self.ids = torch.randint(0, 256, (2, 11))

    def test_parameter_budget_matches_real_model(self):
        self.assertEqual(self.model.config.parameter_count(), sum(p.numel() for p in self.model.parameters()))
        pilot = CandidateConfig(d_model=512, d_inner=768, n_layers=5)
        self.assertEqual(pilot.parameter_count(), 9_979_914)
        with self.assertRaises(ValueError):
            CandidateConfig(d_model=1024, d_inner=1024, n_layers=8)

    def test_language_logits_token_and_chunk_parity(self):
        expected, _ = self.model(self.ids)
        first, state = self.model(self.ids[:, :4])
        last, _ = self.model(self.ids[:, 4:], state)
        torch.testing.assert_close(expected, torch.cat((first, last), dim=1))
        outputs, state = [], None
        for t in range(11):
            out, state = self.model(self.ids[:, t], state, token=True)
            outputs.append(out)
        torch.testing.assert_close(expected, torch.stack(outputs, dim=1))
        self.assertEqual(expected.shape, (2, 11, 256))

    def test_cross_entropy_reaches_every_parameter(self):
        out, _ = self.model(self.ids[:, :-1])
        torch.nn.functional.cross_entropy(out.reshape(-1, 256), self.ids[:, 1:].reshape(-1)).backward()
        for name, param in self.model.named_parameters():
            self.assertIsNotNone(param.grad, name)
            self.assertTrue(torch.isfinite(param.grad).all(), name)
            self.assertGreater(param.grad.abs().sum().item(), 0, name)

    def test_stream_rejects_foreign_owner_and_updated_weights(self):
        _, state = self.model(self.ids)
        other = NativeLanguageCandidate(self.model.config)
        with self.assertRaisesRegex(ValueError, "stale"):
            other(self.ids, state)
        with torch.no_grad():
            next(self.model.parameters()).add_(.01)
        with self.assertRaisesRegex(ValueError, "stale"):
            self.model(self.ids, state)

    def test_checkpoint_roundtrip_and_integrity_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'candidate'
            save_candidate(self.model, root)
            restored = load_candidate(root)
            torch.testing.assert_close(self.model(self.ids)[0], restored(self.ids)[0])
            manifest = json.loads((root / 'manifest.json').read_text())
            manifest['production_activation_authorized'] = True
            (root / 'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                load_candidate(root)
            manifest['production_activation_authorized'] = False
            (root / 'manifest.json').write_text(json.dumps(manifest))
            with (root / 'weights.pt').open('ab') as handle:
                handle.write(b'tampered')
            with self.assertRaisesRegex(ValueError, 'hash'):
                load_candidate(root)


if __name__ == '__main__':
    unittest.main(verbosity=2)

class PilotDataTests(unittest.TestCase):
    def test_splits_and_answer_only_loss(self):
        from pilot_10m import dataset, batch
        splits = dataset()
        self.assertEqual([len(v) for v in splits.values()], [512, 64, 64])
        prompts = [set(x['prompt'] for x in rows) for rows in splits.values()]
        self.assertTrue(all(not prompts[a] & prompts[b] for a in range(3) for b in range(a)))
        for rows in splits.values():
            x, y = batch(rows[:4])
            for i, row in enumerate(rows[:4]):
                values = y[i][y[i] != -100].tolist()
                self.assertEqual(bytes(values), row['answer'].encode())
                self.assertEqual(x[i, :len(row['prompt'].encode())].tolist(), list(row['prompt'].encode()))
