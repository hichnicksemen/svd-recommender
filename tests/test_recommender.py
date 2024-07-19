import unittest
import pandas as pd
from recommender.recommender import CollaborativeRecommender

class TestCollaborativeRecommender(unittest.TestCase):
    def setUp(self):
        ratings_data = {
            'user_id': [1, 1, 1, 2, 2, 3, 3, 3],
            'item_id': [1, 2, 3, 1, 3, 1, 2, 3],
            'rating': [5, 4, 3, 4, 2, 3, 5, 4]
        }
        self.df = pd.DataFrame(ratings_data)
        self.recommender = CollaborativeRecommender(n_components=2)

    def test_fit_predict(self):
        self.recommender.fit(self.df)
        predictions = self.recommender.predict()
        self.assertEqual(predictions.shape, (3, 3))
        print("Predicted Ratings Matrix:")
        print(predictions)

if __name__ == '__main__':
    unittest.main()
