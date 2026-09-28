import math
import random
import unittest
from adaptive_state import AdaptiveStateCell, Affine, Frame, State, StateConfig, compose


class AdaptiveStateTests(unittest.TestCase):
    def assertVectorClose(self, a, b):
        self.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            self.assertTrue(math.isclose(x, y, rel_tol=1e-11, abs_tol=1e-11), (x, y))

    def frames(self, width, length):
        rng = random.Random(97)
        return [Frame(tuple(rng.uniform(-2, 2) for _ in range(width)),
                      tuple(rng.random() for _ in range(width)),
                      tuple(rng.choice((0., .2, 1.)) for _ in range(width)),
                      tuple(rng.random() for _ in range(width))) for _ in range(length)]

    def test_scan_and_step_agree_from_nonzero_state(self):
        for length in (0, 1, 2, 3, 8, 17, 65):
            cell = AdaptiveStateCell(StateConfig(3))
            initial = State(cell.identity, (1., -2., 3.), (-1., 4., 2.))
            frames = self.frames(3, length)
            scan, a = cell.sequence(frames, initial)
            serial, b = cell.sequence(frames, initial, execution="sequential")
            for x, y in zip(scan, serial):
                self.assertVectorClose(x, y)
            self.assertVectorClose(a.fast + a.slow, b.fast + b.slow)

    def test_chunk_boundaries_preserve_state(self):
        cell = AdaptiveStateCell(StateConfig(3))
        frames = self.frames(3, 33)
        expected, terminal = cell.sequence(frames)
        for split in (0, 1, 8, 16, 32, 33):
            first, state = cell.sequence(frames[:split])
            second, state = cell.sequence(frames[split:], state)
            for a, b in zip(first + second, expected):
                self.assertVectorClose(a, b)
            self.assertVectorClose(state.fast + state.slow, terminal.fast + terminal.slow)

    def test_closed_write_holds_slow_state_but_fast_updates(self):
        cell = AdaptiveStateCell(StateConfig(1))
        initial = State(cell.identity, (2.,), (7.,))
        output, state = cell.step(Frame((10.,), (.5,), (0.,), (1.,)), initial)
        self.assertEqual(state.slow, initial.slow)
        self.assertEqual(output, (7.,))
        self.assertNotEqual(state.fast, initial.fast)

    def test_exact_zoh_matches_constant_drive_solution(self):
        cell = AdaptiveStateCell(StateConfig(1))
        _, state = cell.sequence([Frame((2.,), (.1,), (1.,), (.5,))] * 100)
        self.assertVectorClose(state.fast, (2 * (1 - math.exp(-10)),))
        self.assertVectorClose(state.slow, (2 / .05 * (1 - math.exp(-.5)),))

    def test_tiny_dt_does_not_cancel_integral(self):
        cell = AdaptiveStateCell(StateConfig(1))
        _, state = cell.step(Frame((1.,), (1e-18,), (1.,), (.5,)))
        self.assertGreater(state.fast[0], 0)
        self.assertAlmostEqual(state.fast[0] / 1e-18, 1.)

    def test_zero_dt_is_identity(self):
        cell = AdaptiveStateCell(StateConfig(1))
        state = State(cell.identity, (3.,), (9.,))
        _, next_state = cell.step(Frame((100.,), (0.,), (1.,), (.5,)), state)
        self.assertEqual(state, next_state)

    def test_composition_is_associative_and_ordered(self):
        a, b, c = Affine((.2,), (1.,)), Affine((.3,), (2.,)), Affine((.4,), (3.,))
        left, right = compose(c, compose(b, a)), compose(compose(c, b), a)
        self.assertVectorClose(left.scale + left.bias, right.scale + right.bias)
        self.assertNotEqual(compose(a, b).bias, compose(b, a).bias)

    def test_rejects_foreign_state_and_invalid_inputs(self):
        cell = AdaptiveStateCell(StateConfig(1))
        frame = Frame((1.,), (.1,), (.5,), (.5,))
        with self.assertRaises(ValueError):
            cell.step(frame, AdaptiveStateCell(StateConfig(1, slow_rate=.1)).initial_state())
        for invalid in (Frame((math.nan,), (.1,), (.5,), (.5,)),
                        Frame((1.,), (-1.,), (.5,), (.5,)),
                        Frame((1.,), (.1,), (2.,), (.5,)),
                        Frame((1.,), (.1,), (.5,), (math.inf,)),
                        Frame((1., 2.), (.1,), (.5,), (.5,))):
            with self.assertRaises(ValueError):
                cell.step(invalid)
        with self.assertRaises(ValueError):
            StateConfig(True)

    def test_memory_budget_is_fixed_and_identity_covers_configuration(self):
        cfg = StateConfig(128)
        self.assertEqual(cfg.recurrent_bytes(), 512)
        self.assertEqual(cfg.recurrent_bytes(4), 1024)
        self.assertEqual(cfg.fingerprint(), StateConfig(128, fast_rate=1).fingerprint())
        self.assertNotEqual(cfg.fingerprint(), StateConfig(128, slow_rate=.1).fingerprint())


if __name__ == "__main__":
    unittest.main(verbosity=2)
