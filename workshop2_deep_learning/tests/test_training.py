import importlib.util
import unittest
from pathlib import Path

import numpy as np
import torch


MODULE_PATH = Path(__file__).parents[1] / "src" / "train.py"
SPEC = importlib.util.spec_from_file_location("housing_train", MODULE_PATH)
train = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(train)


class TrainingTests(unittest.TestCase):
    def test_standardizer_uses_supplied_training_statistics(self):
        training = np.array(
            [[1.0, 10.0], [3.0, 14.0]], dtype=np.float32
        )
        validation = np.array([[5.0, 18.0]], dtype=np.float32)
        mean, scale = train.fit_standardizer(training)

        np.testing.assert_allclose(mean, [2.0, 12.0])
        np.testing.assert_allclose(scale, [1.0, 2.0])
        np.testing.assert_allclose(
            train.standardize(validation, mean, scale), [[3.0, 3.0]]
        )

    def test_target_round_trip_recovers_original_prices(self):
        prices = np.array([50_000.0, 180_000.0, 750_000.0])
        target_log = np.log1p(prices).astype(np.float32)
        mean, scale = train.fit_standardizer(target_log)
        scaled = train.standardize(target_log, mean, scale)

        recovered_log, recovered_prices = train.inverse_target(
            scaled, float(mean), float(scale)
        )
        np.testing.assert_allclose(recovered_log, target_log, rtol=1e-6)
        np.testing.assert_allclose(recovered_prices, prices, rtol=2e-6)

    def test_model_output_shape_and_finite_values(self):
        model = train.HousingMLP(261, hidden_dims=(64, 32), dropout=0.05)
        output = model(torch.randn(8, 261))

        self.assertEqual(output.shape, (8,))
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
