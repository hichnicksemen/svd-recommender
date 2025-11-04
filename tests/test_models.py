"""
Comprehensive tests for all recommender models.
"""
import unittest
import tempfile
import os
import numpy as np
from recommender import (
    EASERecommender,
    SLIMRecommender,
    SVDRecommender,
    SVDPlusPlusRecommender,
    ALSRecommender,
    InteractionDataset,
    create_synthetic_dataset
)


class TestEASERecommender(unittest.TestCase):
    """Test EASE recommender."""
    
    def setUp(self):
        self.df = create_synthetic_dataset(
            n_users=50,
            n_items=30,
            n_interactions=500,
            implicit=True,
            seed=42
        )
        self.dataset = InteractionDataset(self.df, implicit=True)
        self.train, self.test = self.dataset.split(test_size=0.2, random_state=42)
    
    def test_initialization(self):
        """Test model initialization."""
        model = EASERecommender(l2_reg=100.0)
        self.assertFalse(model.is_fitted)
        self.assertEqual(model.l2_reg, 100.0)
    
    def test_fit(self):
        """Test model training."""
        model = EASERecommender(l2_reg=100.0)
        model.fit(self.train.data)
        
        self.assertTrue(model.is_fitted)
        self.assertIsNotNone(model.item_sim_matrix)
        self.assertEqual(model.item_sim_matrix.shape[0], self.train.n_items)
    
    def test_recommend(self):
        """Test recommendation generation."""
        model = EASERecommender(l2_reg=100.0)
        model.fit(self.train.data)
        
        user_ids = list(self.train.data['user_id'].unique()[:5])
        recommendations = model.recommend(user_ids, k=10)
        
        self.assertEqual(len(recommendations), len(user_ids))
        for user_id in user_ids:
            self.assertIn(user_id, recommendations)
            self.assertLessEqual(len(recommendations[user_id]), 10)
            
            # Check format: list of (item_id, score) tuples
            for item_id, score in recommendations[user_id]:
                self.assertIsInstance(item_id, (int, np.integer))
                self.assertIsInstance(score, (float, np.floating))
    
    def test_exclude_seen(self):
        """Test excluding seen items."""
        model = EASERecommender(l2_reg=100.0)
        model.fit(self.train.data)
        
        user_id = self.train.data['user_id'].iloc[0]
        seen_items = set(self.train.data[self.train.data['user_id'] == user_id]['item_id'])
        
        recommendations = model.recommend([user_id], k=10, exclude_seen=True)
        recommended_items = {item_id for item_id, _ in recommendations[user_id]}
        
        # Check no overlap with seen items
        self.assertEqual(len(recommended_items & seen_items), 0)
    
    def test_save_load(self):
        """Test model serialization."""
        model = EASERecommender(l2_reg=100.0)
        model.fit(self.train.data)
        
        user_ids = [0, 1, 2]
        recs_before = model.recommend(user_ids, k=5)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            temp_path = f.name
        
        try:
            model.save(temp_path)
            
            loaded_model = EASERecommender()
            loaded_model.load(temp_path)
            
            self.assertTrue(loaded_model.is_fitted)
            self.assertEqual(loaded_model.l2_reg, 100.0)
            
            recs_after = loaded_model.recommend(user_ids, k=5)
            
            # Check consistency
            for user_id in user_ids:
                self.assertEqual(len(recs_before[user_id]), len(recs_after[user_id]))
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


class TestSLIMRecommender(unittest.TestCase):
    """Test SLIM recommender."""
    
    def setUp(self):
        self.df = create_synthetic_dataset(
            n_users=40,
            n_items=25,
            n_interactions=400,
            implicit=True,
            seed=42
        )
        self.dataset = InteractionDataset(self.df, implicit=True)
    
    def test_fit_recommend(self):
        """Test SLIM training and recommendation."""
        model = SLIMRecommender(l1_reg=0.1, l2_reg=0.1, max_iter=20)
        model.fit(self.dataset.data)
        
        self.assertTrue(model.is_fitted)
        
        user_ids = list(self.dataset.data['user_id'].unique()[:3])
        recommendations = model.recommend(user_ids, k=5)
        
        self.assertEqual(len(recommendations), len(user_ids))
    
    def test_get_similar_items(self):
        """Test similar items retrieval."""
        model = SLIMRecommender(l1_reg=0.1, l2_reg=0.1, max_iter=20)
        model.fit(self.dataset.data)
        
        item_id = 5
        similar_items = model.get_similar_items(item_id, k=5)
        
        self.assertLessEqual(len(similar_items), 5)
        for sim_item_id, score in similar_items:
            self.assertIsInstance(sim_item_id, (int, np.integer))
            self.assertIsInstance(score, (float, np.floating))
            self.assertNotEqual(sim_item_id, item_id)  # Should not include itself


class TestSVDRecommender(unittest.TestCase):
    """Test SVD recommender."""
    
    def setUp(self):
        self.df = create_synthetic_dataset(
            n_users=50,
            n_items=30,
            n_interactions=500,
            rating_range=(1, 5),
            implicit=False,
            seed=42
        )
        self.dataset = InteractionDataset(self.df, implicit=False)
        self.train, self.test = self.dataset.split(test_size=0.2, random_state=42)
    
    def test_fit_predict(self):
        """Test SVD training and prediction."""
        model = SVDRecommender(n_components=10)
        model.fit(self.train.data)
        
        self.assertTrue(model.is_fitted)
        
        # Test prediction
        user_ids = self.test.data['user_id'].values[:10]
        item_ids = self.test.data['item_id'].values[:10]
        predictions = model.predict(user_ids, item_ids)
        
        self.assertEqual(len(predictions), 10)
        self.assertTrue(np.all(np.isfinite(predictions)))
    
    def test_recommend(self):
        """Test recommendation generation."""
        model = SVDRecommender(n_components=10)
        model.fit(self.train.data)
        
        user_ids = list(self.train.data['user_id'].unique()[:5])
        recommendations = model.recommend(user_ids, k=10)
        
        self.assertEqual(len(recommendations), len(user_ids))


class TestSVDPlusPlusRecommender(unittest.TestCase):
    """Test SVD++ recommender."""
    
    def setUp(self):
        self.df = create_synthetic_dataset(
            n_users=40,
            n_items=25,
            n_interactions=400,
            rating_range=(1, 5),
            implicit=False,
            seed=42
        )
        self.dataset = InteractionDataset(self.df, implicit=False)
        self.train, self.test = self.dataset.split(test_size=0.2, random_state=42)
    
    def test_fit_predict(self):
        """Test SVD++ training and prediction."""
        model = SVDPlusPlusRecommender(
            n_factors=10,
            n_epochs=10,
            lr=0.01,
            reg=0.02
        )
        model.fit(self.train.data)
        
        self.assertTrue(model.is_fitted)
        
        # Test prediction
        user_ids = self.test.data['user_id'].values[:5]
        item_ids = self.test.data['item_id'].values[:5]
        predictions = model.predict(user_ids, item_ids)
        
        self.assertEqual(len(predictions), 5)
        self.assertTrue(np.all(np.isfinite(predictions)))


class TestALSRecommender(unittest.TestCase):
    """Test ALS recommender."""
    
    def setUp(self):
        self.df = create_synthetic_dataset(
            n_users=50,
            n_items=30,
            n_interactions=500,
            implicit=True,
            seed=42
        )
        self.dataset = InteractionDataset(self.df, implicit=True)
        self.train, self.test = self.dataset.split(test_size=0.2, random_state=42)
    
    def test_fit_recommend(self):
        """Test ALS training and recommendation."""
        model = ALSRecommender(n_factors=10, n_iterations=5, reg=0.01)
        model.fit(self.train.data)
        
        self.assertTrue(model.is_fitted)
        
        user_ids = list(self.train.data['user_id'].unique()[:5])
        recommendations = model.recommend(user_ids, k=10)
        
        self.assertEqual(len(recommendations), len(user_ids))
        for user_id in user_ids:
            self.assertLessEqual(len(recommendations[user_id]), 10)
    
    def test_convergence(self):
        """Test that model converges."""
        model = ALSRecommender(n_factors=10, n_iterations=20, reg=0.01)
        model.fit(self.train.data)
        
        # Check that factors are not NaN or Inf
        self.assertTrue(np.all(np.isfinite(model.user_factors)))
        self.assertTrue(np.all(np.isfinite(model.item_factors)))


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def test_predict_before_fit(self):
        """Test that prediction before fit raises error."""
        model = EASERecommender()
        
        with self.assertRaises(Exception):
            model.recommend([1, 2, 3], k=5)
    
    def test_empty_recommendations(self):
        """Test handling of users with no interactions."""
        df = create_synthetic_dataset(
            n_users=20,
            n_items=10,
            n_interactions=50,
            implicit=True,
            seed=42
        )
        dataset = InteractionDataset(df, implicit=True)
        
        model = EASERecommender(l2_reg=100.0)
        model.fit(dataset.data)
        
        # Try to get recommendations for a new user
        new_user_id = 9999
        recommendations = model.recommend([new_user_id], k=5)
        
        # Should return empty or cold start recommendations
        self.assertIn(new_user_id, recommendations)
    
    def test_small_k(self):
        """Test recommendation with k=1."""
        df = create_synthetic_dataset(
            n_users=30,
            n_items=20,
            n_interactions=200,
            implicit=True,
            seed=42
        )
        dataset = InteractionDataset(df, implicit=True)
        
        model = EASERecommender(l2_reg=100.0)
        model.fit(dataset.data)
        
        user_id = dataset.data['user_id'].iloc[0]
        recommendations = model.recommend([user_id], k=1)
        
        self.assertEqual(len(recommendations[user_id]), 1)
    
    def test_large_k(self):
        """Test recommendation with k larger than available items."""
        df = create_synthetic_dataset(
            n_users=30,
            n_items=20,
            n_interactions=200,
            implicit=True,
            seed=42
        )
        dataset = InteractionDataset(df, implicit=True)
        
        model = EASERecommender(l2_reg=100.0)
        model.fit(dataset.data)
        
        user_id = dataset.data['user_id'].iloc[0]
        recommendations = model.recommend([user_id], k=1000)
        
        # Should return at most n_items recommendations
        self.assertLessEqual(len(recommendations[user_id]), dataset.n_items)


if __name__ == '__main__':
    unittest.main(verbosity=2)

