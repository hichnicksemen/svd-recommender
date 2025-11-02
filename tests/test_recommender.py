import unittest
import pandas as pd
import numpy as np
from recommender import (
    EASERecommender,
    SLIMRecommender,
    SVDRecommender,
    ALSRecommender,
    InteractionDataset,
    Evaluator,
    create_synthetic_dataset
)


class TestDataset(unittest.TestCase):
    """Test InteractionDataset functionality."""
    
    def setUp(self):
        self.df = create_synthetic_dataset(
            n_users=100,
            n_items=50,
            n_interactions=1000,
            implicit=True,
            seed=42
        )
    
    def test_dataset_creation(self):
        """Test dataset creation."""
        dataset = InteractionDataset(self.df, implicit=True)
        self.assertGreater(dataset.n_users, 0)
        self.assertGreater(dataset.n_items, 0)
        self.assertGreater(len(dataset), 0)
    
    def test_dataset_split(self):
        """Test train/test split."""
        dataset = InteractionDataset(self.df, implicit=True)
        train, test = dataset.split(test_size=0.2, strategy='random')
        
        self.assertGreater(len(train), 0)
        self.assertGreater(len(test), 0)
        self.assertAlmostEqual(len(test) / len(dataset), 0.2, delta=0.05)
    
    def test_to_csr_matrix(self):
        """Test sparse matrix conversion."""
        dataset = InteractionDataset(self.df, implicit=True)
        matrix = dataset.to_csr_matrix()
        
        self.assertEqual(matrix.shape, (dataset.n_users, dataset.n_items))


class TestSimpleModels(unittest.TestCase):
    """Test EASE and SLIM models."""
    
    def setUp(self):
        self.df = create_synthetic_dataset(
            n_users=50,
            n_items=30,
            n_interactions=500,
            implicit=True,
            seed=42
        )
        self.dataset = InteractionDataset(self.df, implicit=True)
        self.train, self.test = self.dataset.split(test_size=0.2)
    
    def test_ease_fit_recommend(self):
        """Test EASE training and recommendations."""
        model = EASERecommender(l2_reg=100.0)
        model.fit(self.train.data)
        
        self.assertTrue(model.is_fitted)
        
        # Test recommendations
        user_ids = self.train.data['user_id'].unique()[:3]
        recommendations = model.recommend(user_ids, k=5)
        
        self.assertEqual(len(recommendations), len(user_ids))
        for user_id in user_ids:
            self.assertLessEqual(len(recommendations[user_id]), 5)
    
    def test_slim_fit_recommend(self):
        """Test SLIM training and recommendations."""
        model = SLIMRecommender(l1_reg=0.1, l2_reg=0.1, max_iter=20)
        model.fit(self.train.data)
        
        self.assertTrue(model.is_fitted)
        
        # Test recommendations
        user_ids = self.train.data['user_id'].unique()[:3]
        recommendations = model.recommend(user_ids, k=5)
        
        self.assertEqual(len(recommendations), len(user_ids))


class TestMatrixFactorization(unittest.TestCase):
    """Test matrix factorization models."""
    
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
        self.train, self.test = self.dataset.split(test_size=0.2)
    
    def test_svd_fit_predict(self):
        """Test SVD training and prediction."""
        model = SVDRecommender(n_components=10)
        model.fit(self.train.data)
        
        self.assertTrue(model.is_fitted)
        
        # Test prediction
        user_ids = self.test.data['user_id'].values[:5]
        item_ids = self.test.data['item_id'].values[:5]
        predictions = model.predict(user_ids, item_ids)
        
        self.assertEqual(len(predictions), 5)
        self.assertTrue(np.all(np.isfinite(predictions)))
    
    def test_als_fit_recommend(self):
        """Test ALS training and recommendations."""
        model = ALSRecommender(n_factors=10, n_iterations=5)
        model.fit(self.train.data)
        
        self.assertTrue(model.is_fitted)
        
        # Test recommendations
        user_ids = self.train.data['user_id'].unique()[:3]
        recommendations = model.recommend(user_ids, k=5)
        
        self.assertEqual(len(recommendations), len(user_ids))


class TestEvaluation(unittest.TestCase):
    """Test evaluation functionality."""
    
    def setUp(self):
        self.df = create_synthetic_dataset(
            n_users=50,
            n_items=30,
            n_interactions=500,
            implicit=True,
            seed=42
        )
        self.dataset = InteractionDataset(self.df, implicit=True)
        self.train, self.test = self.dataset.split(test_size=0.2)
        
        # Train a simple model
        self.model = EASERecommender(l2_reg=100.0)
        self.model.fit(self.train.data)
    
    def test_evaluator(self):
        """Test evaluator."""
        evaluator = Evaluator(
            metrics=['precision', 'recall', 'ndcg'],
            k_values=[5, 10]
        )
        
        results = evaluator.evaluate(
            self.model,
            self.test,
            task='ranking',
            exclude_train=True,
            train_data=self.train
        )
        
        # Check that metrics are present
        self.assertIn('precision@5', results)
        self.assertIn('recall@10', results)
        self.assertIn('ndcg@5', results)
        
        # Check that values are reasonable
        for metric, value in results.items():
            self.assertTrue(0 <= value <= 1, f"{metric} = {value} is out of range")


class TestSaveLoad(unittest.TestCase):
    """Test model serialization."""
    
    def setUp(self):
        self.df = create_synthetic_dataset(
            n_users=30,
            n_items=20,
            n_interactions=300,
            implicit=True,
            seed=42
        )
        self.dataset = InteractionDataset(self.df, implicit=True)
    
    def test_ease_save_load(self):
        """Test EASE save/load."""
        import tempfile
        import os
        
        model = EASERecommender(l2_reg=100.0)
        model.fit(self.dataset.data)
        
        # Generate recommendations before saving
        user_ids = [0, 1, 2]
        recs_before = model.recommend(user_ids, k=5)
        
        # Save and load
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            temp_path = f.name
        
        try:
            model.save(temp_path)
            
            # Create new model and load
            model_loaded = EASERecommender()
            model_loaded.load(temp_path)
            
            # Generate recommendations after loading
            recs_after = model_loaded.recommend(user_ids, k=5)
            
            # Compare
            self.assertEqual(len(recs_before), len(recs_after))
            for user_id in user_ids:
                self.assertEqual(len(recs_before[user_id]), len(recs_after[user_id]))
        
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


def suite():
    """Create test suite."""
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(TestDataset))
    test_suite.addTest(unittest.makeSuite(TestSimpleModels))
    test_suite.addTest(unittest.makeSuite(TestMatrixFactorization))
    test_suite.addTest(unittest.makeSuite(TestEvaluation))
    test_suite.addTest(unittest.makeSuite(TestSaveLoad))
    return test_suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())
