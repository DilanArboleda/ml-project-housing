"""Reproducibility tests for the frozen final MLP checkpoint."""

import json
import unittest

import numpy as np
import pandas as pd

import evaluate


class FinalModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = evaluate.verify_selection_manifest()
        cls.data = evaluate.load_partitions_and_preprocess()
        cls.checkpoint, cls.model = evaluate.load_checkpoint(
            evaluate.SELECTED_CHECKPOINT
        )

    def test_selection_was_validation_only_and_frozen(self):
        self.assertFalse(self.manifest["protocol"]["test_partition_used"])
        self.assertEqual(
            self.manifest["selected_configuration"]["config_name"],
            "e4_wide_128_64",
        )
        self.assertEqual(self.checkpoint["seed"], 42)
        self.assertEqual(tuple(self.checkpoint["config"]["hidden_dims"]), (128, 64))

    def test_checkpoint_scalers_match_training_only_statistics(self):
        evaluate.verify_checkpoint_preprocessing(self.checkpoint, self.data)

    def test_fixed_test_observations_match_splits_file(self):
        splits = json.loads(
            (evaluate.ROOT / "data" / "splits.json").read_text(encoding="utf-8")
        )
        np.testing.assert_array_equal(
            self.data["test_indices"], np.asarray(splits["test"])
        )
        self.assertEqual(len(self.data["y_test_raw"]), 440)

    def test_checkpoint_reload_reproduces_identical_predictions(self):
        log_first, raw_first = evaluate.predict_deep_learning(
            self.checkpoint, self.model, self.data["X_test"]
        )
        checkpoint_second, model_second = evaluate.load_checkpoint(
            evaluate.SELECTED_CHECKPOINT
        )
        log_second, raw_second = evaluate.predict_deep_learning(
            checkpoint_second, model_second, self.data["X_test"]
        )
        np.testing.assert_array_equal(log_first, log_second)
        np.testing.assert_array_equal(raw_first, raw_second)

    def test_target_inverse_and_metrics_are_on_expected_scales(self):
        prediction_log, prediction_raw = evaluate.predict_deep_learning(
            self.checkpoint, self.model, self.data["X_test"]
        )
        np.testing.assert_allclose(
            prediction_raw, np.expm1(prediction_log), rtol=2e-6
        )
        metrics = evaluate.compute_metrics(
            self.data["y_test_raw"], prediction_raw,
            self.data["y_test_log"], prediction_log,
        )
        expected_rmse_log = np.sqrt(np.mean(
            (self.data["y_test_log"] - prediction_log) ** 2
        ))
        self.assertAlmostEqual(metrics["RMSE_log"], expected_rmse_log, places=12)
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))

    def test_saved_final_metrics_reproduce_selected_checkpoint(self):
        result_path = evaluate.RESULTS_DIR / "final_test_metrics.json"
        self.assertTrue(result_path.exists())
        saved = json.loads(result_path.read_text(encoding="utf-8"))
        prediction_log, prediction_raw = evaluate.predict_deep_learning(
            self.checkpoint, self.model, self.data["X_test"]
        )
        current = evaluate.compute_metrics(
            self.data["y_test_raw"], prediction_raw,
            self.data["y_test_log"], prediction_log,
        )
        expected = saved["metrics"]["Deep Learning MLP"]
        for metric_name, metric_value in current.items():
            self.assertAlmostEqual(metric_value, expected[metric_name], places=10)

    def test_saved_predictions_use_the_same_ordered_observations(self):
        prediction_path = evaluate.RESULTS_DIR / "test_predictions.csv"
        self.assertTrue(prediction_path.exists())
        predictions = pd.read_csv(prediction_path)
        np.testing.assert_array_equal(
            predictions["split_index"].to_numpy(), self.data["test_indices"]
        )
        np.testing.assert_array_equal(
            predictions["SalePrice_actual"].to_numpy(),
            self.data["y_test_raw"],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
