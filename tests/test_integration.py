"""
Integration tests for complete workflows.
"""
import unittest
import tempfile
import os
import pandas as pd
from recommender import (
    EASERecommender,
    SLIMRecommender,
    SVDRecommender,
    ALSRecommender,
    InteractionDataset,
    Evaluator,
    create_synthetic_dataset
)


class TestCompleteWorkflow(unittest.TestCase):
    """Test complete recommendation workflow."""
    
    def test_implicit_feedback_workflow(self):
        """Test complete workflow for implicit feedback."""
        # 1. Create data
        df = create_synthetic_dataset(
            n_users=100,
            n_items=50,
            n_interactions=1000,
            implicit=True,
            seed=42
        )
        
        # 2. Create dataset
        dataset = InteractionDataset(
            df,
            implicit=True,
            min_user_interactions=5,
            min_item_interactions=3
        )
        
        self.assertGreater(dataset.n_users, 0)
        self.assertGreater(dataset.n_items, 0)
        
        # 3. Split data
        train, test = dataset.split(test_size=0.2, strategy='random', random_state=42)
        
        self.assertGreater(len(train), 0)
        self.assertGreater(len(test), 0)
        
        # 4. Train model
        model = EASERecommender(l2_reg=500.0)
        model.fit(train.data)
        
        self.assertTrue(model.is_fitted)
        
        # 5. Generate recommendations
        user_ids = list(train.data['user_id'].unique()[:10])
        recommendations = model.recommend(user_ids, k=10, exclude_seen=True)
        
        self.assertEqual(len(recommendations), len(user_ids))
        
        # 6. Evaluate
        evaluator = Evaluator(
            metrics=['precision', 'recall', 'ndcg'],
            k_values=[5, 10]
        )
        
        results = evaluator.evaluate(
            model,
            test,
            task='ranking',
            exclude_train=True,
            train_data=train
        )
        
        self.assertIn('precision@5', results)
        self.assertIn('recall@10', results)
        self.assertIn('ndcg@5', results)
        
        # 7. Save model
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            temp_path = f.name
        
        try:
            model.save(temp_path)
            self.assertTrue(os.path.exists(temp_path))
            
            # 8. Load model
            loaded_model = EASERecommender()
            loaded_model.load(temp_path)
            
            self.assertTrue(loaded_model.is_fitted)
            
            # 9. Test loaded model
            recs_loaded = loaded_model.recommend(user_ids[:3], k=5)
            self.assertEqual(len(recs_loaded), 3)
        
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_explicit_feedback_workflow(self):
        """Test complete workflow for explicit feedback."""
        # 1. Create data
        df = create_synthetic_dataset(
            n_users=80,
            n_items=40,
            n_interactions=800,
            rating_range=(1, 5),
            implicit=False,
            seed=42
        )
        
        # 2. Create dataset
        dataset = InteractionDataset(
            df,
            implicit=False,
            min_user_interactions=3,
            min_item_interactions=3
        )
        
        # 3. Split data
        train, test = dataset.split(test_size=0.2, strategy='random', random_state=42)
        
        # 4. Train model
        model = SVDRecommender(n_components=10)
        model.fit(train.data)
        
        self.assertTrue(model.is_fitted)
        
        # 5. Predict ratings
        user_ids = test.data['user_id'].values[:20]
        item_ids = test.data['item_id'].values[:20]
        predictions = model.predict(user_ids, item_ids)
        
        self.assertEqual(len(predictions), 20)
        
        # 6. Evaluate
        evaluator = Evaluator(
            metrics=['rmse', 'mae'],
            k_values=[10]
        )
        
        results = evaluator.evaluate(
            model,
            test,
            task='rating_prediction',
            train_data=train
        )
        
        self.assertIn('rmse', results)
        self.assertIn('mae', results)


class TestModelComparison(unittest.TestCase):
    """Test comparing multiple models."""
    
    def test_compare_models(self):
        """Test comparing different recommender models."""
        # Create data
        df = create_synthetic_dataset(
            n_users=80,
            n_items=40,
            n_interactions=800,
            implicit=True,
            seed=42
        )
        
        dataset = InteractionDataset(df, implicit=True, min_user_interactions=5)
        train, test = dataset.split(test_size=0.2, random_state=42)
        
        # Models to compare
        models = {
            'EASE': EASERecommender(l2_reg=500.0),
            'SLIM': SLIMRecommender(l1_reg=0.1, l2_reg=0.1, max_iter=20),
            'ALS': ALSRecommender(n_factors=10, n_iterations=5)
        }
        
        # Evaluator
        evaluator = Evaluator(
            metrics=['precision', 'recall', 'ndcg'],
            k_values=[5, 10]
        )
        
        results = {}
        
        for model_name, model in models.items():
            # Train
            model.fit(train.data)
            self.assertTrue(model.is_fitted, f"{model_name} failed to fit")
            
            # Evaluate
            model_results = evaluator.evaluate(
                model,
                test,
                task='ranking',
                exclude_train=True,
                train_data=train
            )
            
            results[model_name] = model_results
            
            # Check that we got results
            self.assertIn('ndcg@10', model_results)
            self.assertGreater(model_results['ndcg@10'], 0.0)
        
        # Check that we have results for all models
        self.assertEqual(len(results), len(models))


class TestColdStart(unittest.TestCase):
    """Test cold start scenarios."""
    
    def test_new_user_recommendations(self):
        """Test recommendations for new users."""
        df = create_synthetic_dataset(
            n_users=50,
            n_items=30,
            n_interactions=500,
            implicit=True,
            seed=42
        )
        
        dataset = InteractionDataset(df, implicit=True)
        model = EASERecommender(l2_reg=100.0)
        model.fit(dataset.data)
        
        # Try to get recommendations for a user not in training data
        new_user_id = 9999
        recommendations = model.recommend([new_user_id], k=10)
        
        # Should handle gracefully (either empty or popular items)
        self.assertIn(new_user_id, recommendations)
        self.assertIsInstance(recommendations[new_user_id], list)
    
    def test_mixed_known_and_new_users(self):
        """Test recommendations for mix of known and new users."""
        df = create_synthetic_dataset(
            n_users=50,
            n_items=30,
            n_interactions=500,
            implicit=True,
            seed=42
        )
        
        dataset = InteractionDataset(df, implicit=True)
        model = EASERecommender(l2_reg=100.0)
        model.fit(dataset.data)
        
        # Mix of known and new users
        known_user = dataset.data['user_id'].iloc[0]
        new_user = 9999
        user_ids = [known_user, new_user]
        
        recommendations = model.recommend(user_ids, k=5)
        
        # Should return recommendations for both
        self.assertEqual(len(recommendations), 2)
        self.assertIn(known_user, recommendations)
        self.assertIn(new_user, recommendations)


class TestDataQuality(unittest.TestCase):
    """Test handling of data quality issues."""
    
    def test_duplicate_interactions(self):
        """Test handling of duplicate interactions."""
        df = create_synthetic_dataset(
            n_users=50,
            n_items=30,
            n_interactions=500,
            implicit=True,
            seed=42
        )
        
        # Add some duplicates
        duplicates = df.head(10).copy()
        df_with_dups = pd.concat([df, duplicates], ignore_index=True)
        
        # Dataset should handle duplicates
        dataset = InteractionDataset(df_with_dups, implicit=True)
        model = EASERecommender(l2_reg=100.0)
        model.fit(dataset.data)
        
        self.assertTrue(model.is_fitted)
    
    def test_sparse_data(self):
        """Test with very sparse data."""
        df = create_synthetic_dataset(
            n_users=100,
            n_items=100,
            n_interactions=200,  # Very sparse: 2% density
            implicit=True,
            seed=42
        )
        
        dataset = InteractionDataset(
            df,
            implicit=True,
            min_user_interactions=2,
            min_item_interactions=2
        )
        
        if len(dataset) > 0:
            model = EASERecommender(l2_reg=100.0)
            model.fit(dataset.data)
            
            self.assertTrue(model.is_fitted)


if __name__ == '__main__':
    unittest.main(verbosity=2)

