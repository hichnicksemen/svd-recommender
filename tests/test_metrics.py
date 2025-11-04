"""
Tests for evaluation metrics.
"""
import unittest
import numpy as np
from recommender.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    f1_score_at_k,
    ndcg_at_k,
    map_at_k,
    mrr_at_k,
    hit_rate_at_k,
    coverage,
    diversity,
    rmse,
    mae,
    mse,
    r_squared
)


class TestRankingMetrics(unittest.TestCase):
    """Test ranking metrics."""
    
    def setUp(self):
        # Perfect recommendations
        self.recommendations_perfect = {
            1: [(10, 0.9), (20, 0.8), (30, 0.7)],
            2: [(15, 0.95), (25, 0.85)]
        }
        
        self.relevant_items_perfect = {
            1: {10, 20, 30},
            2: {15, 25}
        }
        
        # Partial match
        self.recommendations_partial = {
            1: [(10, 0.9), (99, 0.8), (30, 0.7)],  # 99 is not relevant
            2: [(15, 0.95), (88, 0.85)]  # 88 is not relevant
        }
        
        self.relevant_items_partial = {
            1: {10, 20, 30},
            2: {15, 25}
        }
    
    def test_precision_at_k_perfect(self):
        """Test precision with perfect recommendations."""
        precision = precision_at_k(
            self.recommendations_perfect,
            self.relevant_items_perfect,
            k=3
        )
        
        # User 1: 3/3 = 1.0, User 2: 2/2 = 1.0
        self.assertAlmostEqual(precision, 1.0)
    
    def test_precision_at_k_partial(self):
        """Test precision with partial match."""
        precision = precision_at_k(
            self.recommendations_partial,
            self.relevant_items_partial,
            k=3
        )
        
        # User 1: 2/3 = 0.667, User 2: 1/2 = 0.5
        # Average: (0.667 + 0.5) / 2 = 0.583
        self.assertAlmostEqual(precision, 0.583, places=2)
    
    def test_recall_at_k_perfect(self):
        """Test recall with perfect recommendations."""
        recall = recall_at_k(
            self.recommendations_perfect,
            self.relevant_items_perfect,
            k=3
        )
        
        # User 1: 3/3 = 1.0, User 2: 2/2 = 1.0
        self.assertAlmostEqual(recall, 1.0)
    
    def test_recall_at_k_partial(self):
        """Test recall with partial match."""
        recall = recall_at_k(
            self.recommendations_partial,
            self.relevant_items_partial,
            k=3
        )
        
        # User 1: 2/3 = 0.667, User 2: 1/2 = 0.5
        self.assertAlmostEqual(recall, 0.583, places=2)
    
    def test_f1_score_at_k(self):
        """Test F1 score."""
        f1 = f1_score_at_k(
            self.recommendations_partial,
            self.relevant_items_partial,
            k=3
        )
        
        # Should be harmonic mean of precision and recall
        precision = precision_at_k(
            self.recommendations_partial,
            self.relevant_items_partial,
            k=3
        )
        recall = recall_at_k(
            self.recommendations_partial,
            self.relevant_items_partial,
            k=3
        )
        expected_f1 = 2 * (precision * recall) / (precision + recall)
        
        self.assertAlmostEqual(f1, expected_f1, places=5)
    
    def test_ndcg_at_k(self):
        """Test NDCG."""
        ndcg = ndcg_at_k(
            self.recommendations_perfect,
            self.relevant_items_perfect,
            k=3
        )
        
        # Perfect recommendations should have NDCG = 1.0
        self.assertAlmostEqual(ndcg, 1.0)
    
    def test_map_at_k(self):
        """Test MAP (Mean Average Precision)."""
        map_score = map_at_k(
            self.recommendations_perfect,
            self.relevant_items_perfect,
            k=3
        )
        
        # Perfect recommendations should have MAP = 1.0
        self.assertAlmostEqual(map_score, 1.0)
    
    def test_mrr_at_k(self):
        """Test MRR (Mean Reciprocal Rank)."""
        recommendations = {
            1: [(99, 0.9), (10, 0.8), (20, 0.7)],  # First relevant at position 2
            2: [(15, 0.95), (25, 0.85)]  # First relevant at position 1
        }
        
        relevant_items = {
            1: {10, 20},
            2: {15, 25}
        }
        
        mrr = mrr_at_k(recommendations, relevant_items, k=3)
        
        # User 1: 1/2 = 0.5, User 2: 1/1 = 1.0
        # Average: (0.5 + 1.0) / 2 = 0.75
        self.assertAlmostEqual(mrr, 0.75)
    
    def test_hit_rate_at_k(self):
        """Test Hit Rate."""
        hit_rate = hit_rate_at_k(
            self.recommendations_partial,
            self.relevant_items_partial,
            k=3
        )
        
        # Both users have at least one hit
        self.assertAlmostEqual(hit_rate, 1.0)
    
    def test_coverage(self):
        """Test catalog coverage."""
        recommendations = {
            1: [(10, 0.9), (20, 0.8)],
            2: [(15, 0.95), (20, 0.85)],  # 20 appears again
            3: [(30, 0.9)]
        }
        
        n_items = 100
        cov = coverage(recommendations, n_items)
        
        # Unique items: {10, 15, 20, 30} = 4
        # Coverage: 4 / 100 = 0.04
        self.assertAlmostEqual(cov, 0.04)
    
    def test_diversity(self):
        """Test diversity metric."""
        recommendations = {
            1: [(10, 0.9), (20, 0.8), (30, 0.7)],
            2: [(10, 0.95), (20, 0.85), (40, 0.75)]  # 2 items overlap with user 1
        }
        
        div = diversity(recommendations)
        
        # Should be between 0 and 1
        self.assertGreaterEqual(div, 0.0)
        self.assertLessEqual(div, 1.0)


class TestRatingPredictionMetrics(unittest.TestCase):
    """Test rating prediction metrics."""
    
    def setUp(self):
        self.y_true = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        self.y_pred = np.array([4.8, 4.2, 2.8, 2.1, 1.2])
    
    def test_rmse(self):
        """Test RMSE calculation."""
        rmse_score = rmse(self.y_true, self.y_pred)
        
        # Manual calculation
        mse = np.mean((self.y_true - self.y_pred) ** 2)
        expected_rmse = np.sqrt(mse)
        
        self.assertAlmostEqual(rmse_score, expected_rmse, places=5)
    
    def test_mae(self):
        """Test MAE calculation."""
        mae_score = mae(self.y_true, self.y_pred)
        
        # Manual calculation
        expected_mae = np.mean(np.abs(self.y_true - self.y_pred))
        
        self.assertAlmostEqual(mae_score, expected_mae, places=5)
    
    def test_mse(self):
        """Test MSE calculation."""
        mse_score = mse(self.y_true, self.y_pred)
        
        # Manual calculation
        expected_mse = np.mean((self.y_true - self.y_pred) ** 2)
        
        self.assertAlmostEqual(mse_score, expected_mse, places=5)
    
    def test_r_squared(self):
        """Test R² calculation."""
        r2_score = r_squared(self.y_true, self.y_pred)
        
        # R² should be between -inf and 1 (1 is perfect)
        self.assertLessEqual(r2_score, 1.0)
        
        # For reasonably good predictions, should be positive
        self.assertGreater(r2_score, 0.8)
    
    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = y_true.copy()
        
        self.assertAlmostEqual(rmse(y_true, y_pred), 0.0)
        self.assertAlmostEqual(mae(y_true, y_pred), 0.0)
        self.assertAlmostEqual(mse(y_true, y_pred), 0.0)
        self.assertAlmostEqual(r_squared(y_true, y_pred), 1.0)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases in metrics."""
    
    def test_empty_recommendations(self):
        """Test metrics with empty recommendations."""
        recommendations = {
            1: [],
            2: []
        }
        
        relevant_items = {
            1: {10, 20},
            2: {15, 25}
        }
        
        precision = precision_at_k(recommendations, relevant_items, k=5)
        recall = recall_at_k(recommendations, relevant_items, k=5)
        
        # Should be 0 when no recommendations
        self.assertEqual(precision, 0.0)
        self.assertEqual(recall, 0.0)
    
    def test_no_relevant_items(self):
        """Test metrics when no relevant items."""
        recommendations = {
            1: [(10, 0.9), (20, 0.8)],
        }
        
        relevant_items = {
            1: set()  # No relevant items
        }
        
        precision = precision_at_k(recommendations, relevant_items, k=5)
        
        # Should handle gracefully
        self.assertIsInstance(precision, float)
    
    def test_k_larger_than_recommendations(self):
        """Test when k is larger than number of recommendations."""
        recommendations = {
            1: [(10, 0.9), (20, 0.8)],
        }
        
        relevant_items = {
            1: {10, 20, 30}
        }
        
        recall = recall_at_k(recommendations, relevant_items, k=100)
        
        # Recall should be 2/3
        self.assertAlmostEqual(recall, 2.0 / 3.0, places=5)


if __name__ == '__main__':
    unittest.main(verbosity=2)

