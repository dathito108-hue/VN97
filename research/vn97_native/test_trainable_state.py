import copy
import io
import unittest
import torch
from adaptive_state import AdaptiveStateCell, Frame, State, StateConfig
from trainable_state import TensorState, TrainableAdaptiveState


class TrainableStateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(97)
        torch.set_num_threads(1)
        self.cell = TrainableAdaptiveState(2, StateConfig(3)).double()
        self.x = torch.randn(2, 9, 2, dtype=torch.float64)

    def test_tensor_matches_independent_scalar_reference(self):
        cell, x = self.cell, self.x
        fast, slow = (r.item() for r in cell.rates())
        oracle = AdaptiveStateCell(StateConfig(3, fast_rate=fast, slow_rate=slow))
        initial = TensorState(cell.architecture, torch.randn(2, 3).double(), torch.randn(2, 3).double())
        actual, end = cell(x, initial)
        drive, dt, write, read = [v.detach().tolist() for v in cell.selectors(x)]
        for batch in range(2):
            frames = [Frame(tuple(drive[batch][t]), tuple(dt[batch][t]), tuple(write[batch][t]), tuple(read[batch][t])) for t in range(9)]
            seed = State(oracle.identity, tuple(initial.fast[batch].tolist()), tuple(initial.slow[batch].tolist()))
            expected, final = oracle.sequence(frames, seed, execution="sequential")
            torch.testing.assert_close(actual[batch], torch.tensor(expected, dtype=torch.float64), rtol=1e-11, atol=1e-11)
            torch.testing.assert_close(end.slow[batch], torch.tensor(final.slow), check_dtype=False)

    def test_scan_and_sequential_parameter_gradients_match(self):
        other = copy.deepcopy(self.cell)
        x1, x2 = self.x.clone().requires_grad_(), self.x.clone().requires_grad_()
        y1, s1 = self.cell(x1)
        y2, s2 = other(x2, execution="sequential")
        (y1.square().mean() + s1.slow.square().mean()).backward()
        (y2.square().mean() + s2.slow.square().mean()).backward()
        torch.testing.assert_close(y1, y2, rtol=1e-11, atol=1e-11)
        torch.testing.assert_close(x1.grad, x2.grad, rtol=1e-10, atol=1e-11)
        for a, b in zip(self.cell.parameters(), other.parameters()):
            self.assertTrue(torch.isfinite(a.grad).all())
            self.assertGreater(a.grad.abs().sum().item(), 0)
            torch.testing.assert_close(a.grad, b.grad, rtol=1e-10, atol=1e-11)

    def test_gradcheck_inputs_and_all_parameters(self):
        cell = TrainableAdaptiveState(1, StateConfig(1)).double()
        params = dict(cell.named_parameters())
        x = torch.randn(1, 3, 1, dtype=torch.float64, requires_grad=True)
        def function(value, *weights):
            y, state = torch.func.functional_call(cell, dict(zip(params, weights)), (value,))
            return y, state.fast, state.slow
        self.assertTrue(torch.autograd.gradcheck(function, (x, *params.values()), eps=1e-6, atol=1e-5, rtol=1e-3))

    def test_chunked_backprop_matches_whole_sequence(self):
        other = copy.deepcopy(self.cell)
        full, _ = self.cell(self.x)
        first, state = other(self.x[:, :4])
        second, _ = other(self.x[:, 4:], state)
        joined = torch.cat((first, second), dim=1)
        torch.testing.assert_close(full, joined)
        full.square().sum().backward()
        joined.square().sum().backward()
        for a, b in zip(self.cell.parameters(), other.parameters()):
            torch.testing.assert_close(a.grad, b.grad)
        self.assertFalse(state.detach().fast.requires_grad)

    def test_checkpoint_and_token_continuation(self):
        data = io.BytesIO()
        torch.save(self.cell.state_dict(), data)
        data.seek(0)
        restored = TrainableAdaptiveState(2, StateConfig(3)).double()
        restored.load_state_dict(torch.load(data, weights_only=True))
        expected, _ = restored(self.x)
        state, outputs = None, []
        for t in range(self.x.shape[1]):
            y, state = restored.step(self.x[:, t], state)
            outputs.append(y)
        torch.testing.assert_close(expected, torch.stack(outputs, dim=1))
        empty, same = restored(self.x[:, :0], state)
        self.assertEqual(empty.shape, (2, 0, 3))
        self.assertIs(same, state)

    def test_invalid_state_and_inputs_fail(self):
        with self.assertRaises(ValueError):
            self.cell(self.x.float())
        bad = self.x.clone()
        bad[0, 0, 0] = float('nan')
        with self.assertRaises(ValueError):
            self.cell(bad)
        foreign = TrainableAdaptiveState(2, StateConfig(3, slow_rate=.2)).double().initial_state(2)
        with self.assertRaises(ValueError):
            self.cell(self.x, foreign)
        with self.assertRaises(ValueError):
            self.cell(self.x, self.cell.initial_state(1))

    def test_optimizer_reduces_synthetic_memory_task_loss(self):
        # Optimization wiring only: this batch is not a held-out language benchmark.
        torch.manual_seed(97)
        cell = TrainableAdaptiveState(1, StateConfig(1))
        x = torch.randn(8, 12, 1)
        carry = torch.zeros(8, 1)
        targets = []
        for t in range(12):
            carry = .8 * carry + .2 * x[:, t]
            targets.append(carry)
        target = torch.stack(targets, dim=1)
        optimizer = torch.optim.Adam(cell.parameters(), lr=.03)
        before = ((cell(x)[0] - target) ** 2).mean().item()
        for _ in range(120):
            optimizer.zero_grad()
            loss = ((cell(x)[0] - target) ** 2).mean()
            loss.backward()
            optimizer.step()
        after = ((cell(x)[0] - target) ** 2).mean().item()
        print(f'SYNTHETIC_OPTIMIZER_SMOKE seed=97 initial_mse={before:.9g} final_mse={after:.9g}', flush=True)
        self.assertLess(after, before * .5)
        fast, slow = cell.rates()
        self.assertGreater(fast.item(), slow.item())
        self.assertGreater(slow.item(), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
