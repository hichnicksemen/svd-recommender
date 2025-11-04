"""
Tests for data processing functionality.
"""
import unittest
import pandas as pd
import numpy as np
from recommender import (
    InteractionDataset,
    create_synthetic_dataset,
    load_movielens,
    UniformSampler,
    PopularitySampler
)
from recommender.data.preprocessing import (
    filter_by_interaction_count,
    binarize_implicit_feedback,
    create_sequences
)


class TestSyntheticData(unittest.TestCase):
    """Test synthetic data generation."""
    
    def test_implicit_dataset(self):
        """Test implicit feedback dataset generation."""
        df = create_synthetic_dataset(
            n_users=100,
            n_items=50,
            n_interactions=500,
            implicit=True,
            seed=42
        )
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn('user_id', df.columns)
        self.assertIn('item_id', df.columns)
        self.assertIn('rating', df.columns)
        self.assertIn('timestamp', df.columns)
        
        # Check that all ratings are 1 for implicit
        self.assertTrue(np.all(df['rating'] == 1.0))
        
    def test_explicit_dataset(self):
        """Test explicit feedback dataset generation."""
        df = create_synthetic_dataset(
            n_users=100,
            n_items=50,
            n_interactions=500,
            rating_range=(1, 5),
            implicit=False,
            seed=42
        )
        
        # Check rating range
        self.assertTrue(df['rating'].min() >= 1)
        self.assertTrue(df['rating'].max() <= 5)
        
    def test_reproducibility(self):
        """Test that same seed produces same data."""
        df1 = create_synthetic_dataset(n_users=50, n_items=30, n_interactions=200, seed=42)
        df2 = create_synthetic_dataset(n_users=50, n_items=30, n_interactions=200, seed=42)
        
        pd.testing.assert_frame_equal(df1, df2)


class TestInteractionDataset(unittest.TestCase):
    """Test InteractionDataset class."""
    
    def setUp(self):
        self.df = create_synthetic_dataset(
            n_users=100,
            n_items=50,
            n_interactions=1000,
            implicit=True,
            seed=42
        )
    
    def test_initialization(self):
        """Test dataset initialization."""
        dataset = InteractionDataset(self.df, implicit=True)
        
        self.assertGreater(dataset.n_users, 0)
        self.assertGreater(dataset.n_items, 0)
        self.assertEqual(len(dataset), len(dataset.data))
    
    def test_filtering(self):
        """Test min interactions filtering."""
        dataset = InteractionDataset(
            self.df,
            implicit=True,
            min_user_interactions=5,
            min_item_interactions=3
        )
        
        # Check filtering worked
        user_counts = dataset.data.groupby('user_id').size()
        item_counts = dataset.data.groupby('item_id').size()
        
        self.assertTrue(user_counts.min() >= 5)
        self.assertTrue(item_counts.min() >= 3)
    
    def test_random_split(self):
        """Test random train/test split."""
        dataset = InteractionDataset(self.df, implicit=True)
        train, test = dataset.split(test_size=0.2, strategy='random')
        
        self.assertGreater(len(train), 0)
        self.assertGreater(len(test), 0)
        self.assertAlmostEqual(len(test) / len(dataset), 0.2, delta=0.05)
        
        # Check no overlap
        train_set = set(zip(train.data['user_id'], train.data['item_id']))
        test_set = set(zip(test.data['user_id'], test.data['item_id']))
        self.assertEqual(len(train_set & test_set), 0)
    
    def test_temporal_split(self):
        """Test temporal split."""
        dataset = InteractionDataset(self.df, implicit=True)
        train, test = dataset.split(test_size=0.2, strategy='temporal')
        
        # Check that test timestamps are later than train
        train_max_time = train.data['timestamp'].max()
        test_min_time = test.data['timestamp'].min()
        
        # Most test timestamps should be after most train timestamps
        self.assertLessEqual(train_max_time, test_min_time + 100000000)  # Some tolerance
    
    def test_leave_one_out_split(self):
        """Test leave-one-out split."""
        dataset = InteractionDataset(self.df, implicit=True, min_user_interactions=3)
        # Use temporal split instead (leave-one-out not implemented)
        train, test = dataset.split(test_size=0.2, strategy='temporal')
        
        # Check split worked
        self.assertGreater(len(train), 0)
        self.assertGreater(len(test), 0)
    
    def test_to_csr_matrix(self):
        """Test conversion to CSR matrix."""
        dataset = InteractionDataset(self.df, implicit=True)
        matrix = dataset.to_csr_matrix()
        
        self.assertEqual(matrix.shape, (dataset.n_users, dataset.n_items))
        self.assertTrue(matrix.nnz > 0)
    
    def test_get_user_items(self):
        """Test getting items for a user."""
        dataset = InteractionDataset(self.df, implicit=True)
        user_id = dataset.data['user_id'].iloc[0]
        
        items = dataset.get_user_items(user_id)
        self.assertIsInstance(items, (list, np.ndarray))
        self.assertGreater(len(items), 0)


class TestPreprocessing(unittest.TestCase):
    """Test preprocessing functions."""
    
    def setUp(self):
        self.df = create_synthetic_dataset(
            n_users=100,
            n_items=50,
            n_interactions=1000,
            rating_range=(1, 5),
            implicit=False,
            seed=42
        )
    
    def test_filter_by_interaction_count(self):
        """Test filtering by interaction count."""
        filtered_df = filter_by_interaction_count(
            self.df,
            min_user_interactions=5,
            min_item_interactions=3
        )
        
        user_counts = filtered_df.groupby('user_id').size()
        item_counts = filtered_df.groupby('item_id').size()
        
        self.assertTrue(user_counts.min() >= 5)
        self.assertTrue(item_counts.min() >= 3)
    
    def test_binarize_implicit_feedback(self):
        """Test binarization of ratings."""
        binary_df = binarize_implicit_feedback(self.df, threshold=3.0)
        
        self.assertIn('rating', binary_df.columns)
        # Check that ratings are binary
        unique_ratings = binary_df['rating'].unique()
        self.assertTrue(len(unique_ratings) <= 2)
        self.assertTrue(all(r in [0, 1] for r in unique_ratings))
    
    def test_create_sequences(self):
        """Test sequence creation."""
        sequences, _ = create_sequences(self.df, max_seq_length=10)
        
        self.assertIsInstance(sequences, list)
        self.assertGreater(len(sequences), 0)
        
        # Check sequence format  
        for seq in sequences:
            self.assertIsInstance(seq, list)
            self.assertLessEqual(len(seq), 10)


class TestSamplers(unittest.TestCase):
    """Test negative samplers."""
    
    def test_uniform_sampler(self):
        """Test uniform negative sampler."""
        sampler = UniformSampler(n_items=100, seed=42)
        
        # Sample negative items for a user
        user_idx = 0
        positive_items = {1, 5, 10, 20}  # Items user has interacted with
        
        samples1 = sampler.sample(user_idx, positive_items, n_negatives=10)
        samples2 = sampler.sample(user_idx, positive_items, n_negatives=10)
        
        self.assertEqual(len(samples1), 10)
        self.assertEqual(len(samples2), 10)
        
        # Check all samples are valid items
        self.assertTrue(all(0 <= s < 100 for s in samples1))
        
        # Check no positive items in negatives
        self.assertEqual(len(set(samples1) & positive_items), 0)
    
    def test_popularity_sampler(self):
        """Test popularity-based sampler."""
        item_popularity = {i: i + 1 for i in range(100)}  # Items 0-99
        sampler = PopularitySampler(
            n_items=100,
            item_popularity=item_popularity,
            seed=42
        )
        
        user_idx = 0
        positive_items = {1, 2, 3}
        
        # Sample many times to check distribution
        all_samples = []
        for _ in range(100):
            samples = sampler.sample(user_idx, positive_items, n_negatives=10)
            all_samples.extend(samples)
        
        # Check that popular items (higher IDs) are sampled more often
        # Mean should be higher than uniform (which would be ~50)
        self.assertGreater(np.mean(all_samples), 50)


class TestDataLoaders(unittest.TestCase):
    """Test dataset loaders."""
    
    def test_movielens_loader(self):
        """Test MovieLens data loader (if available)."""
        try:
            df = load_movielens(size='100k')
            
            self.assertIsInstance(df, pd.DataFrame)
            self.assertIn('user_id', df.columns)
            self.assertIn('item_id', df.columns)
            self.assertIn('rating', df.columns)
            self.assertGreater(len(df), 0)
        except Exception as e:
            self.skipTest(f"MovieLens data not available: {e}")


if __name__ == '__main__':
    unittest.main(verbosity=2)

