import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location('audit', Path(__file__).resolve().parents[1] / 'tools/g06_numeric_audit.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class NumericalMetricsTests(unittest.TestCase):
    def test_exact_and_known_error(self):
        value = np.array([0.0, 1.0, -2.0], dtype=np.float16)
        self.assertEqual(audit.difference(value, value), 0)
        changed = value.copy()
        changed[0] = 0.25
        self.assertEqual(audit.difference(value, changed), 0.25)

    def test_nonfinite_rejected_on_either_side(self):
        for bad in (np.nan, np.inf, -np.inf):
            value = np.array([bad], dtype=np.float32)
            good = np.zeros(1, dtype=np.float32)
            for a, b in ((value, good), (good, value)):
                with self.assertRaisesRegex(ValueError, 'non-finite'):
                    audit.difference(a, b)

    def test_shape_and_dtype_are_part_of_contract(self):
        with self.assertRaisesRegex(ValueError, 'shape/dtype'):
            audit.difference(np.zeros(2, dtype=np.float16), np.zeros((1, 2), dtype=np.float16))
        with self.assertRaisesRegex(ValueError, 'shape/dtype'):
            audit.difference(np.zeros(2, dtype=np.float16), np.zeros(2, dtype=np.float32))

    def test_maximum_at_chunk_boundary(self):
        a = np.zeros(1024 * 1024 + 2, dtype=np.float16)
        b = a.copy()
        b[-1] = 3
        self.assertEqual(audit.difference(a, b), 3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
